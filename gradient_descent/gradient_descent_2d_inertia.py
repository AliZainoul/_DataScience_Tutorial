"""
1D Gradient Descent with Momentum and Local Minima Highlighting
----------------------------------------------------------------

This script visualizes momentum-based gradient descent on a quartic function:
    f(x) = 3x⁴ - 4x³ - 12x² - 3

Features:
- Momentum-based gradient descent (correction term)
- Animated descent
- Highlights all critical points (local minima and maxima)
- Convergence criterion via gradient magnitude
"""

from __future__ import annotations

from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


# --------------------------------------------------------------------------- #
# Type Alias
# --------------------------------------------------------------------------- #

Vector_N_d = Union[float, NDArray[np.float64]]


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
# Gradient Descent with Momentum
# --------------------------------------------------------------------------- #

def animate_gradient_descent_momentum(
    learning_rate: float = 0.015,
    momentum: float = 0.3,
    epsilon: float = 1e-6,
    max_iterations: int = 150,
) -> None:
    """
    Animate momentum-based gradient descent on the quartic function.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent.
    momentum : float
        Momentum coefficient for the correction term.
    epsilon : float
        Convergence threshold based on gradient magnitude.
    max_iterations : int
        Maximum number of iterations to avoid infinite loops.
    """

    # Prepare function curve
    x_vals: Vector_N_d = np.arange(-3.0, 4.0, 0.1, dtype=np.float64)
    y_vals: Vector_N_d = function(x_vals)

    plt.figure(figsize=(9, 6))
    plt.plot(x_vals, y_vals, label="f(x) = 3x⁴ - 4x³ - 12x² - 3", linewidth=2)
    plt.title("Momentum Gradient Descent with Local Minima")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)

    # Highlight critical points
    critical_points = [-1, 0, 2]  # From f'(x) = 0
    colors = ["green", "blue", "purple"]
    labels = ["Local Min (-1)", "Local Max (0)", "Global Min (2)"]

    for cp, color, label in zip(critical_points, colors, labels):
        plt.scatter(cp, function(cp), color=color, s=80, label=label)
    plt.legend()

    # Random starting point
    x: float = float(np.random.randint(-3, 4) + np.random.rand())
    correction: float = 0.0  # momentum correction term
    
    for iteration in range(max_iterations):
        # Plot current point
        plt.scatter(x, function(x), color="red")
        plt.pause(0.4)

        grad: float = float(gradient_function(x))
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

        # Momentum-based gradient descent update
        correction = momentum * correction - learning_rate * grad
        x = x + correction

    plt.show()


# --------------------------------------------------------------------------- #
# Main Execution
# --------------------------------------------------------------------------- #

def main() -> None:
    """Run the momentum-based gradient descent animation."""
    animate_gradient_descent_momentum()


if __name__ == "__main__":
    main()
