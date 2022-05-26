# Finite differencing scheme to solve a horizontal spring-damper system
# DE is described as 0 = m*x'' + k2*x' + k1*x

import numpy as np
import matplotlib.pyplot as plt

h = 0.1
T_max = 10

N = round(T_max / h)

# T array just gets used so the x-axis has appropriate units at the end instead of "steps"
T = np.linspace(0, T_max, N)

# create empty matrices
A = np.zeros((N, N))
B = np.zeros(N)

# system definition
m = 1
k1 = 1
k2 = 1

# define initial conditions for the problem
v0 = 0
x0 = 1

# initial condition, Dirichlet BC
B[0] = x0

# Neumann BC
B[1] = x0 + v0 * h

A[0][0] = 1
A[1][1] = 1

# co-efficients for central second-order finite difference
I = (m / h / h) - (k2 / (2 * h))
J = (-2 * m / h / h) + k1
K = (m / h / h) + (k2 / (2 * h))

# put coefficients into matrix
for i in range(2, N - 1):
    A[i][i - 1] = I  # X_{N - 1}
    A[i][i + 0] = J  # X_{N}
    A[i][i + 1] = K  # X_{N + 1}

# last row can't use central scheme so use backwards instead
A[N - 1][N - 3] = m / h / h
A[N - 1][N - 2] = (-2 * m / h / h) - (k2 / h)
A[N - 1][N - 1] = (m / h / h) + (k2 / h) + k1

X = np.linalg.solve(A, B)

plt.plot(T, X)
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.show()
