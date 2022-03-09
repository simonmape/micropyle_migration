from dolfin import *
import math
from fenics import *
from scipy.integrate import odeint
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
w_sa = 0
gamma = 1
zeta = 1
E=1
dt = 0.01

class phiIC(UserExpression):
    def eval(self,value,x):
        if 0.02 < (x[0]-0.5)**2 + (x[1]-0.5)**2 < 0.24 and x[2] < 0.1:
            value[:] = 1
        else:
            value[:] = 0
    def value_shape(self):
        return ()
    
class polarIC(UserExpression):
    def eval(self,value,x):
        if 0.02 < (x[0]-0.5)**2 + (x[1]-0.5)**2 < 0.24 and x[2] < 0.1:
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
        
        # Assign initial conditions
        self.str_old = interpolate(ZeroTensor(),TS)
        self.v_old = interpolate(vIC(), V)
        zero = Expression(('0.0'), degree=2)
        self.pr_old = interpolate(zero,W)
        self.p_old = interpolate(polarIC(),V)
        self.phi_old = interpolate(phiIC(),W)

        self.polarity_assigner = FunctionAssigner(V, V)
        self.stress_assigner = FunctionAssigner(TS, TS)
        self.velocity_assigner = FunctionAssigner(flowspace.sub(0), V)
        self.pressure_assigner = FunctionAssigner(flowspace.sub(1), W)
        self.phi_assigner = FunctionAssigner(W, W)

        #Define variational forms
        self.y = TestFunction(V)
        u = TrialFunction(V)
        a = (1./dt)*inner(u,self.y)*dx
        self.A_pol = assemble(a)

        u = TrialFunction(TS)
        self.z = TestFunction(TS)
        a = (1+eta/(E*dt))*inner(u,self.z)*dx
        self.A_str = assemble(a)

        yw = TestFunction(flowspace)
        self.y1,self.w1=split(yw)
        dU = TrialFunction(flowspace)
        (du1, du2) = split(dU)
        a = eta*inner(nabla_grad(du1),nabla_grad(self.y1))*dx +\
            inner(nabla_grad(du2),self.y1)*dx +\
            inner(div(du1),self.w1)*dx
        self.A_flow = assemble(a)

        u = TrialFunction(W)
        self.w2 = TestFunction(W)
        a = (1./dt)*u*self.w2*dx
        self.A_phi = assemble(a)
    def E(self, u):
        return sym(nabla_grad(u))

    def W(self, u):
        return skew(nabla_grad(u))

    def advance_one_step(self, t):
        # Load objects from previous time step and create new
        str_old = self.str_old
        v_old = self.v_old
        pr_old = self.pr_old
        p_old = self.p_old
        phi_old = self.phi_old
        p_new = Function(V)
        str_new = Function(TS)
        vpr_new = Function(flowspace)
        phi_new = Function(W)
        print('Initializing t=',t)
        print('polarity', p_old.vector().get_local().min(),p_old.vector().get_local().max())
        print('stress', str_old.vector().get_local().min(), str_old.vector().get_local().max())
        print('velocity', vpr_old.sub(0).vector().get_local().min(), vpr_old.sub(0).vector().get_local().max())

        #POLARITY EVOLUTION #
        y = self.y
        L = (1. / dt) * inner(p_old, y) * dx + inner(nabla_grad(p_old) * (v_old + w_sa * p_old), y) * dx
        b = assemble(L)
        solver = KrylovSolver("gmres","ilu")
        solver.set_operator(self.A_pol)
        #self.polarity_assigner.assign(p_new, p_old)
        solver.solve(p_new.vector(),b)
        print('polarity', p_new.vector().get_local().min(),p_new.vector().get_local().max())

        #STRESS TENSOR
        z = self.z
        L = eta * inner(self.E(v_old), z) * dx + (eta / E * dt) * inner(str_old, z) * dx
        b = assemble(L)
        solver = KrylovSolver("gmres","ilu")
        solver.set_operator(self.A_str)
        #self.stress_assigner.assign(str_new, str_old)
        solver.solve(str_new.vector(),b)
        print('stress',str_new.vector().get_local().min(),str_new.vector().get_local().max())

        # FLOW PROBLEM#
        # yw = TestFunction(flowspace)
        y,w=  self.y1, self.w1
        v_new, pr_new = split(vpr_new)
        #self.velocity_assigner.assign(vpr_new.sub(0),v_old)
        #self.pressure_assigner.assign(vpr_new.sub(1), pr_old)
        L = -zeta*inner(div(outer(p_new,p_new)),y)*dx - gamma*inner(v_old,y)*dx
        b = assemble(L)
        solver = KrylovSolver("gmres", "ilu")
        solver.set_operator(self.A_flow)
        solver.solve(vpr_new.vector(), b)
        print('velocity', vpr_new.sub(0).vector().get_local().min(), vpr_new.sub(0).vector().get_local().max())

        #PHASE FIELD PROBLEM#
        L = (1. / dt) * phi_old * self.w2 * dx + dot(v_new, nabla_grad(phi_old)) * self.w2 * dx
        b = assemble(L)
        solver = KrylovSolver("gmres", "ilu")
        solver.set_operator(self.A_phi)
        #self.phi_assigner.assign(phi_new, phi_old)
        solver.solve(phi_new.vector(), b)

        #ASSIGN ALL VARIABLES FOR NEW STEP
        self.str_old.assign(str_new)
        self.p_old.assign(p_new)
        assigner = FunctionAssigner(V, flowspace.sub(0))
        assigner.assign(self.v_old,vpr_new.sub(0))
        self.phi_old.assign(phi_new)
        assigner = FunctionAssigner(W, flowspace.sub(1))
        assigner.assign(self.pr_old, vpr_new.sub(1))

phi_file = File("results/phi.pvd")
p_file = File("results/p.pvd")
v_file = File("results/v.pvd")


system_solver = NSSolver()
set_log_level(20)
numSteps = 10

for i in tqdm(range(numSteps)):
    t = i*dt
    phi = system_solver.phi_old
    p = system_solver.p_old
    v = system_solver.v_old

    phi.rename("phi","phi")
    p.rename("p","p")
    v.rename("v","v")

    phi_file << phi
    p_file << p
    v_file << v

    # Advance one time step in the simulation
    system_solver.advance_one_step(t)