from dolfin import *
import math
from fenics import *
from scipy.integrate import odeint
import numpy as np
from ufl import nabla_div
from tqdm import tqdm
import sys
import os

#Define computational domain
num_points = 30
mesh = UnitCubeMesh(num_points, num_points, num_points)

W = FunctionSpace(mesh, 'P', 1)
V = VectorFunctionSpace(mesh, "CG", 2)
TS = TensorFunctionSpace(mesh, 'P', 1)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)

# Make mixed space for the flow problem
flowspace_element = P2 * P1
flowspace = FunctionSpace(mesh, flowspace_element)
velocity_assigner_inv = FunctionAssigner(V, flowspace.sub(0))
pressure_assigner_inv = FunctionAssigner(W, flowspace.sub(1))

# Set fenics parameters
parameters["form_compiler"]["quadrature_degree"] = 3
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["krylov_solver"]["nonzero_initial_guess"] = True
set_log_level(16)

# Set fixed simulation parameters
k = 1
eta = 5. / 3.
D = 0.007
a = 1
M = D / a
zeta = 0.01
dt = 0.01
numSteps = int(3 / dt)

# Define main expressions
class phiIC(UserExpression):
    def eval(self, value, x):
        value[:] = 0
        if 0.25 < x[0] < 0.75 and 0.25 < x[1] < 0.75 and 0.25 < x[2] < 0.75:
            value[:] = 1
    def value_shape(self):
        return ()

boundary = 'near(x[0],0) || near(x[0],1) || near(x[1], 0) || near(x[1],1) || near(x[2], 0) || near(x[2],1)'

#Set initial conditions
phi_old = interpolate(phiIC(), W)
v_old = Function(V)

#FLOW PROBLEM
vpr_new = Function(flowspace)
v_new, pr_new = split(vpr_new)
yw = TestFunction(flowspace)
y, w = split(yw)
dU = TrialFunction(flowspace)
(du1, du2) = split(dU)
#stress is - isotropic pressure?
a_v = eta*inner(nabla_grad(du1), nabla_grad(y)) * dx - \
      du2 * nabla_div(y) * dx + \
      div(du1) * w * dx
zero = Expression(('0.0', '0.0', '0.0', '0.0'), degree=2)
bcs_flow = DirichletBC(flowspace, zero, boundary)

# PHASE FIELD PROBLEM#
phi_new = Function(W)
w1 = TestFunction(W)
dphi = TrialFunction(W)

a_phi = (1. / dt) * dphi * w1 * dx + M * dot(nabla_grad(dphi), nabla_grad(w1)) * dx
zero = Expression(('0.0'), degree=2)
bcs_phi = DirichletBC(W, zero, boundary)

#VELOCITY RHS
#Define activity
act = Expression('1/(1+exp(-20*(x[2]-0.5)))', degree =2)
L_v =  -100*act * inner((1./3.)*Identity(3) - outer(nabla_grad(phi_old), nabla_grad(phi_old))/(inner(nabla_grad(phi_old),nabla_grad(phi_old))+1e-2) ,nabla_grad(y)) * dx

# PHASE FIELD PROBLEM#
L_phi = (1. / dt) * phi_old * w1 * dx - div(phi_old * vpr_new.sub(0)) * w1 * dx - \
        phi_old*(1-phi_old)*(1-2*phi_old) * w1 * dx

#Create output files for saving
phi_file = File("results/phi.pvd", "compressed")
v_file = File("results/v.pvd", "compressed")

for i in tqdm(range(numSteps)):
    t = i * dt
    print('velocity')
    solve(a_v == L_v, vpr_new, bcs_flow, solver_parameters=dict(linear_solver='gmres',
                                                                preconditioner='ilu'))

    print('Phi')
    solve(a_phi == L_phi, phi_new, bcs_phi, solver_parameters=dict(linear_solver='gmres',
                                                                    preconditioner='ilu'))

    # ASSIGN ALL VARIABLES FOR NEW STEP
    phi_old.assign(phi_new)
    velocity_assigner_inv.assign(v_old, vpr_new.sub(0))
    phi_new.rename("phi","phi")
    vpr_new.sub(0).rename("v","v")
    phi_file << phi_new
    v_file << vpr_new.sub(0)