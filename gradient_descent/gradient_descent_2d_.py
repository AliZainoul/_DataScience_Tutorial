"""
1D Gradient Descent with Local Minima Highlighting
--------------------------------------------------

This script visualizes gradient descent on a quartic function:
    f(x) = 3x⁴ - 4x³ - 12x² - 3

Features:
- Epsilon-based convergence test
- MSE-based error evaluation
- Animated gradient descent
- Highlights all critical points (local minima and maxima)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


# --------------------------------------------------------------------------- #
# Type Alias
# --------------------------------------------------------------------------- #

Vector_N_d = float | NDArray[np.float64]


# --------------------------------------------------------------------------- #
# Mathematical Functions
# --------------------------------------------------------------------------- #

def function(x: Vector_N_d) -> Vector_N_d:
    """Return f(x) = 3x⁴ - 4x³ - 12x² - 3."""
    return 3 * x**4 - 4 * x**3 - 12 * x**2 - 3


def gradient_function(x: Vector_N_d) -> Vector_N_d:
    """Return derivative f'(x) = 12x³ - 12x² - 24x."""
    return 12 * x**3 - 12 * x**2 - 24 * x


def mse_error(x: float, target: float) -> float:
    """Compute squared L2 distance to a given target."""
    return float((x - target) ** 2)


# --------------------------------------------------------------------------- #
# Gradient Descent Visualization
# --------------------------------------------------------------------------- #

def animate_gradient_descent(
    learning_rate: float = 0.015,
    epsilon: float = 1e-9,
    max_iterations: int = 150,
) -> None:
    """Animate gradient descent and highlight local minima/maxima."""

    # Prepare function curve
    x_vals: Vector_N_d = np.arange(-3.0, 4.0, 0.1, dtype=np.float64)
    y_vals: Vector_N_d = function(x_vals)

    plt.figure(figsize=(9, 6))
    plt.plot(x_vals, y_vals, label="f(x) = 3x⁴ - 4x³ - 12x² - 3", linewidth=2)
    plt.title("Quartic Gradient Descent with Local Minima/Maxima")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()

    # Critical points
    critical_points = [-1, 0, 2]  # -1, 0, 2 from f'(x)=0
    colors = ["green", "blue", "purple"]
    labels = ["Local Min (-1)", "Local Max (0)", "Global Min (2)"]

    for cp, color, label in zip(critical_points, colors, labels):
        plt.scatter(cp, function(cp), color=color, s=80, label=label)
    plt.legend()

    # Random starting point
    x: float = float(np.random.randint(-3, 4) + np.random.rand())

    for iteration in range(max_iterations):

        # Plot current descent point
        plt.scatter(x, function(x), color="red")
        plt.pause(0.4)

        grad: float = float(gradient_function(x))
        # Compute MSE to global minimum at x=2
        err: float = mse_error(x, 2.0)

        print(
            f"Iteration {iteration:3d} | x = {x:+10.9f} "
            f"| grad = {grad:+10.9f} | MSE = {err:10.9f}"
        )

        # Convergence check
        if abs(grad) <= epsilon:
            print("\n Converged: gradient below epsilon threshold.")
            print(
                f"Optimal Solution:\nIteration {iteration:3d} | x = {x:+10.9f} "
                f"| grad = {grad:+10.9f} | MSE = {err:10.9f}"
            )
            break

        # Gradient descent update
        x = float(x - learning_rate * grad)

    plt.show()


# --------------------------------------------------------------------------- #
# Main Execution
# --------------------------------------------------------------------------- #

def main() -> None:
    """Run the gradient descent animation."""
    animate_gradient_descent()


if __name__ == "__main__":
    main()
