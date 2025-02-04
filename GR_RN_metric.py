import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
M = 1.0  # Mass of the black hole
Q = 0.5  # Charge of the black hole

# Reissner-Nordström metric function
def f(r):
    return 1 - (2 * M) / r + (Q**2) / (r**2)

# Geodesic equations
def geodesic_eq(t, y):
    r, phi, pr, pphi = y
    drdt = pr
    dphidt = pphi / (r**2)
    dprdt = - (M / r**2) + (Q**2 / r**3) + (pphi**2) / (r**3) - (2 * M * pphi**2) / (r**4)
    dpphidt = 0  # Conserved angular momentum
    return [drdt, dphidt, dprdt, dpphidt]

# Initial conditions
r0 = 10.0
phi0 = 0.0
pr0 = 0.0
pphi0 = 4.0

# Time span
t_span = (0, 100)
t_eval = np.linspace(*t_span, 10000)

# Solve the geodesic equations
sol = solve_ivp(geodesic_eq, t_span, [r0, phi0, pr0, pphi0], t_eval=t_eval)

# Extract results
r = sol.y[0]
phi = sol.y[1]

# Convert to Cartesian coordinates for plotting
x = r * np.cos(phi)
y = r * np.sin(phi)

# Plotting
fig, ax = plt.subplots()
ax.plot(x, y, label='Reissner-Nordström Orbit')
ax.plot(0, 0, 'ko', label='Black Hole')  # Black hole at the center

# Adjust legend to prevent errors
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.set_title('Reissner-Nordström Geodesic Simulation')
plt.show()
