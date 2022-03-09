from dolfin import *
import math
from fenics import *
import numpy as np
from ufl import nabla_div
from tqdm import tqdm

num_points = 30
mesh = UnitCubeMesh(num_points, num_points, num_points)
W = FunctionSpace(mesh, 'P', 1)
V = VectorFunctionSpace(mesh, "CG", 2)
TS = TensorFunctionSpace(mesh, 'P', 1, symmetry=True)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)

# Make mixed spaces
phasespace_element = P1 * P1
polarityspace_element = P2 * P2
flowspace_element = P2 * P1
phasespace = FunctionSpace(mesh, phasespace_element)
polarityspace = FunctionSpace(mesh, polarityspace_element)
flowspace = FunctionSpace(mesh, flowspace_element)

#Define simulation parameters
eta = 5 / 3
w_sa = 1
gamma = 1
zeta = 1
E = 1
dt = 0.1
numSteps = 10

# Set fenics parameters
parameters["form_compiler"]["quadrature_degree"] = 3
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["krylov_solver"]["nonzero_initial_guess"] = True
set_log_level(20)

class ZeroTensor(UserExpression):
    def init(self, **kwargs):
        super().init(**kwargs)

    def eval(self, value, x):
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
        return (3, 3)

class phiIC(UserExpression):
    def eval(self, value, x):
        if 0.01 < (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 < 0.1 and 0.1 < x[2] < 0.2:
            value[:] = 1
        else:
            value[:] = 0

    def value_shape(self):
        return ()

class polarIC(UserExpression):
    def eval(self, value, x):
        if 0.01 < (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 < 0.1 and 0.1 < x[2] < 0.2:
            value[:] = [0.5 - x[0], 0.5 - x[1], 0]
        else:
            value[:] = [0, 0, 0]

    def value_shape(self):
        return (3,)

class vIC(UserExpression):
    def eval(self, value, x):
        value[:] = [0, 0, 0]

    def value_shape(self):
        return (3,)

boundary = 'near(x[0],0) || near(x[1], 0) || near(x[2], 0) || near(x[0],1) || near(x[1], 1) || near(x[2], 1)'

# Assign initial conditions
str_old = interpolate(ZeroTensor(), TS)
v_old = interpolate(vIC(), V)
zero = Expression(('0.0'), degree=2)
pr_old = interpolate(zero, W)
p_old = interpolate(polarIC(), V)
phi_old = interpolate(phiIC(), W)

polarity_assigner = FunctionAssigner(V, V)
stress_assigner = FunctionAssigner(TS, TS)
velocity_assigner = FunctionAssigner(flowspace.sub(0), V)
pressure_assigner = FunctionAssigner(flowspace.sub(1), W)
phi_assigner = FunctionAssigner(W, W)
velocity_assigner_inv = FunctionAssigner(V, flowspace.sub(0))
pressure_assigner_inv = FunctionAssigner(W, flowspace.sub(1))

#Define candidate functions
p_new = Function(V)
str_new = Function(TS)
vpr_new = Function(flowspace)
v_new, pr_new = split(vpr_new)
phi_new = Function(W)

#Define trial functions
uV = TrialFunction(V)
uT = TrialFunction(TS)
uphi = TrialFunction(W)
dU = TrialFunction(flowspace)
(du1, du2) = split(dU)

#Define test functions
y = TestFunction(V)
z = TestFunction(TS)
yw = TestFunction(flowspace)
y1, w1 = split(yw)
w2 = TestFunction(W)

#Define operator for polarity
a_pol = (1. / dt) * dot(u, y) * dx
A_pol = assemble(a_pol)
solver_pol = KrylovSolver("gmres", "ilu")
solver_pol.set_operator(A_pol)

#Define operator for stress
a_str = (1 + eta / (E * dt)) * inner(u, z) * dx
A_str = assemble(a_str)
solver_str = KrylovSolver("gmres", "ilu")
solver_str.set_operator(A_str)

#Define operator for flow problem
a_flow = eta * inner(nabla_grad(du1), nabla_grad(y1)) * dx +\
    dot(nabla_grad(du2), y1) * dx +\
    dot(div(du1), w1) * dx +\
    gamma * dot(du1, y1) * dx
A_flow = assemble(a_flow)
solver_flow = KrylovSolver("gmres", "ilu")
solver_flow.set_operator(A_flow)

#Define operator for phase field
a_phi = (1. / dt) * u * w2 * dx
A_phi = assemble(a_phi)
solver_phi = KrylovSolver("gmres", "ilu")
solver_phi.set_operator(A_phi)

def E(self, u):
    return sym(nabla_grad(u))

def W(self, u):
    return skew(nabla_grad(u))

phi_file = File("results/phi.pvd")
p_file = File("results/p.pvd")
v_file = File("results/v.pvd")

phi_old.rename("phi", "phi")
p_old.rename("p", "p")
v_old.rename("v", "v")
phi_new.rename("phi", "phi")
p_new.rename("p", "p")
v_new.rename("v", "v")

phi_file << phi_old
p_file << p_old
v_file << v_old

for i in tqdm(range(numSteps)):
    print('Initializing i=', i)
    print('polarity', p_old.vector().get_local().min(), p_old.vector().get_local().max())
    print('stress', str_old.vector().get_local().min(), str_old.vector().get_local().max())
    print('velocity', v_old.vector().get_local().min(), v_old.vector().get_local().max())

    # POLARITY EVOLUTION #
    L = (1. / dt) * dot(p_old, y) * dx + dot(p_old, y) * dx - dot(nabla_grad(p_old) * (v_old + w_sa * p_old), y) * dx
    b = assemble(L)
    solver_pol.solve(p_new.vector(), b)

    # STRESS TENSOR
    L = eta * inner(E(v_old), z) * dx + (eta / E * dt) * inner(str_old, z) * dx
    b = assemble(L)
    solver_str.solve(str_new.vector(), b)

    # FLOW PROBLEM#
    L = - zeta * dot(div(outer(p_new, p_new)), y1) * dx
    b = assemble(L)
    solver_flow.solve(vpr_new.vector(), b)

    # PHASE FIELD PROBLEM#
    L = (1. / dt) * phi_old * w2 * dx + dot(v_new, nabla_grad(phi_old)) * w2 * dx
    b = assemble(L)
    solver_phi.solve(phi_new.vector(), b)

    # ASSIGN ALL VARIABLES FOR NEW STEP
    str_old.assign(str_new)
    p_old.assign(p_new)
    velocity_assigner_inv.assign(v_old, vpr_new.sub(0))
    phi_old.assign(phi_new)
    pressure_assigner_inv.assign(pr_old, vpr_new.sub(1))

    phi_file << phi_new
    p_file << p_new
    v_file << v_new