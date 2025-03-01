# %%test
from matplotlib.patches import Circle
from FileHandling import *
import numpy as np

alpha = 0.015
x_initial = 0  # Starting value of x
iterations = 60  # Number of iterations to simulate
alphas = [0.5, 0.3, 0.2, 0.1, 0.05]
plt.figure(figsize=(8, 6))
# Initialize variables
for alpha in alphas:
    x_values = [x_initial]  # List to store x values

    # Simulate the movement of x
    x = x_initial
    for _ in range(iterations):
        x = x + (5 - x) * alpha
        x_values.append(x)

    # Plot the results

    plt.plot(np.linspace(0, 1, 61), x_values, marker="o", linestyle="-", label=alpha)

plt.title(r"Movement of $x$: $x_{\text{new}} = x + 10(1-x)\alpha$")
plt.xlabel("Iteration")
plt.ylabel("x")
plt.grid(True)
plt.legend()
plt.show()
