from dolfin import *
import math
from fenics import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
from ufl import nabla_div
from tqdm import tqdm

num_points = 20
mesh = UnitCubeMesh(num_points, num_points, num_points)

W = FunctionSpace(mesh, 'P', 1)
V = VectorFunctionSpace(mesh, "CG", 2)
TS = TensorFunctionSpace(mesh, 'P', 1,symmetry=True)

P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)

# Make mixed spaces
phasespace_element = P1 * P1
polarityspace_element = P2 * P2
flowspace_element = P2 * P1

phasespace = FunctionSpace(mesh, phasespace_element)
polarityspace = FunctionSpace(mesh, polarityspace_element)
flowspace = FunctionSpace(mesh, flowspace_element)

# Set fenics parameters
parameters["form_compiler"]["quadrature_degree"] = 3
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"]=True
parameters["krylov_solver"]["nonzero_initial_guess"] = True

class ZeroTensor(UserExpression):
    def init(self,**kwargs):
        super().init(**kwargs)
    def eval(self,value,x):
        value[0] = 0
        value[1] = 0
        value[2] = 0
        value[3] = 0
        value[4] = 0
        value[5] = 0
        value[6] = 0
        value[7] = 0
        value[8] = 0
        
    def value_shape(self):
        return (3,3)
        
eta = 5/3
w_sa = 0.0015*3600
gamma = 0.04
zeta = 0.01
E=1
#Initialize polarity and phi

class phiIC(UserExpression):
    def eval(self,value,x):
        if 0.02 < (x[0]-0.5)**2 + (x[1]-0.5)**2 < 0.9 and x[2] < 0.1:
            value[:] = 1
        else:
            value[:] = 0
    def value_shape(self):
        return ()
    
class polarIC(UserExpression):
    def eval(self,value,x):
        if 0.02 < (x[0]-0.5)**2 + (x[1]-0.5)**2 < 0.9 and x[2] < 0.1:
            value[:] = [0.5-x[0],0.5-x[1],0]
        else:
            value[:] = [0,0,0]
    def value_shape(self):
        return (3,)
    
class vIC(UserExpression):
    def eval(self,value,x):
        value[:] = [0,0,0]
    def value_shape(self):
        return (3,)
        
class NSSolver:
    def __init__(self):
        # Define the boundaries
        self.boundary = 'near(x[0],0) || near(x[1], 0) || near(x[2], 0) || near(x[0],1) || near(x[1], 1) || near(x[2], 1)'
        self.n = FacetNormal(mesh)
        
        # Assign initial conditions for the viscous stress
        self.str_old = interpolate(ZeroTensor(),TS)
        
        # Assign initial conditions for velocity and pressure
        self.v_old = interpolate(vIC(), V)
        zero = Expression(('0.0'), degree=2)
        self.pr_old = interpolate(zero,W)

        # Assign initial conditions for polarity fields
        self.p_old = interpolate(polarIC(),V)
        
        #Assign initial condition for the phi field
        self.phi_old = interpolate(phiIC(),W)

    def E(self, u):
        return sym(nabla_grad(u))

    def W(self, u):
        return skew(nabla_grad(u))

    def advance_one_step(self, t):
        # Load objects from previous time step
        str_old = self.str_old
        v_old = self.v_old
        pr_old = self.pr_old
        p_old = self.p_old
        phi_old = self.phi_old
        
        p_new = Function(V)
        str_new = Function(TS)
        vpr_new = Function(flowspace)
        phi_new = Function(W)

        #Assign guesses to next iteration
        # str_new.assign(str_old)
        # p_new.assign(p_old)
        # assigner = FunctionAssigner(flowspace.sub(0),V)
        # assigner.assign(vpr_new.sub(0),v_old)
        # phi_new.assign(phi_old)

        #POLARITY EVOLUTION #
        #Define variational formulation
        print('Entering polarity problem')
        y = TestFunction(V)
        u = TrialFunction(V)
        #Fp = (1./dt)*inner(p_new-p_old,y)*dx - inner(nabla_grad(p_old)*(v_old + w_sa*p_old),y)*dx
        a = (1./dt)*inner(u,y)*dx
        L = (1./dt)*inner(p_old,y)*dx + inner(nabla_grad(p_old)*(v_old + w_sa*p_old),y)*dx
        A = assemble(a)
        b = assemble(L)

        solver = KrylovSolver("gmres","ilu")
        solver.set_operator(A)
        #p_new.assign(p_old)
        assigner = FunctionAssigner(V, V)
        assigner.assign(p_new, p_old)
        #Solve variational problem
        solver.solve(p_new.vector(),b)

        #STRESS TENSOR 
        #Define variational formulation
        print('Entering stress problem')
        u = TrialFunction(TS)
        z = TestFunction(TS)

        # Fstr = (1+eta/(E*dt))*inner(str_new,z)*dx - eta*inner(self.E(v_old),z)*dx -\
        #         (eta/E*dt)*inner(str_old,z)*dx

        a = (1+eta/(E*dt))*inner(u,z)*dx
        L = eta*inner(self.E(v_old),z)*dx + (eta/E*dt)*inner(str_old,z)*dx
        A = assemble(a)
        b = assemble(L)

        solver = KrylovSolver("gmres","ilu")
        solver.set_operator(A)
        #p_new.assign(p_old)
        assigner = FunctionAssigner(TS, TS)
        assigner.assign(str_new, str_old)
        solver.solve(str_new.vector(),b)

        # FLOW PROBLEM#
        print('Entering flow problem')
        yw = TestFunction(flowspace)
        y,w=split(yw)
        v_new, pr_new = split(vpr_new)

        assigner = FunctionAssigner(flowspace.sub(0), V)
        assigner.assign(vpr_new.sub(0),v_old)

        assigner = FunctionAssigner(flowspace.sub(1), W)
        assigner.assign(vpr_new.sub(1), pr_old)

        dU = TrialFunction(flowspace)
        (du1, du2) = split(dU)

        a = inner(nabla_grad(du1),nabla_grad(y))*dx +\
            inner(nabla_grad(du2),y)*dx +\
            inner(div(du1),w)*dx
        L = -zeta*inner(div(outer(p_new,p_new)),y)*dx - gamma*inner(v_old,y)*dx

        A = assemble(a)
        b = assemble(L)
        solver = KrylovSolver("gmres", "ilu")
        solver.set_operator(A)
        solver.solve(vpr_new.vector(), b)
        
        #PHASE FIELD PROBLEM#
        print('Entered phase field problem')
        phi_new = Function(W)
        u = TrialFunction(W)
        w1 = TestFunction(W)
        #phi evolution       
        #F_phi = (1./dt)*(phi_new-phi_old)*w1*dx - dot(v_new,nabla_grad(phi_old))*w1*dx
        a = (1./dt)*u*w1*dx
        L = (1./dt)*phi_old*w1*dx + dot(v_new,nabla_grad(phi_old))*w1*dx

        A = assemble(a)
        b = assemble(L)

        solver = KrylovSolver("gmres", "ilu")
        solver.set_operator(A)
        assigner = FunctionAssigner(W, W)
        assigner.assign(phi_new, phi_old)
        solver.solve(phi_new.vector(), b)

        #ASSIGN ALL VARIABLES FOR NEW STEP
        #Flow problem variables
        self.str_old.assign(str_new)
        self.p_old.assign(p_new)
        assigner = FunctionAssigner(V,flowspace.sub(0))
        assigner.assign(self.v_old,vpr_new.sub(0))
        self.phi_old.assign(phi_new)
        self.pr_old.assign(pr_new)
# Defining the problem
system_solver = NSSolver()
set_log_level(20)
numSteps=1

dt = 0.01
for i in tqdm(range(numSteps)):
    t = i*dt
    # Advance one time step in the simulation
    system_solver.advance_one_step(t)