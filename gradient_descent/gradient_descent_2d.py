"""
1D Gradient Descent with Convergence Criterion and Error Tracking
-----------------------------------------------------------------

This script visualizes gradient descent on a quadratic function.
It includes:

- Epsilon-based convergence test
- MSE-based error evaluation
- Clean Pylance/mypy-safe type annotations

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
    """Return f(x) = x² + 3x - 2."""
    return x ** 2 + 3 * x - 2


def gradient_function(x: Vector_N_d) -> Vector_N_d:
    """Return derivative f'(x) = 2x + 3."""
    return 2 * x + 3


def mse_error(x: float) -> float:
    """
    Compute the squared L2-distance to the optimal value x* = -1.5.

    MSE = (x - x_target)^2
    """
    x_target: float = -1.5  # Minimum of the function
    return float((x - x_target) ** 2)


# --------------------------------------------------------------------------- #
# Gradient Descent Visualization
# --------------------------------------------------------------------------- #

def animate_gradient_descent(
    learning_rate: float = 0.2,
    epsilon: float = 1e-9,
    max_iterations: int = 150,
) -> None:
    """
    Plot the function and animate the gradient descent process.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent.

    epsilon : float
        Convergence threshold on |gradient|.

    max_iterations : int
        Maximum iterations to avoid infinite loops.
    """

    # Prepare curve
    x_vals: Vector_N_d = np.arange(-5.0, 3.0, 0.1, dtype=np.float64)
    y_vals: Vector_N_d = function(x_vals)

    plt.figure(figsize=(9, 6))
    plt.plot(x_vals, y_vals, label="f(x) = x² + 3x - 2", linewidth=2)
    plt.title("Gradient Descent with Error and Convergence Test")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()

    # Random starting point
    x: float = float(np.random.randint(-4, 4) + np.random.rand())

    for iteration in range(max_iterations):

        # Plot current point
        plt.scatter(x, function(x), color="red")
        plt.pause(0.4)

        grad : float = float(gradient_function(x))
        err : float = mse_error(x)

        print(
            f"Iteration {iteration:3d} | x = {x:+10.9f} "
            f"| grad = {grad:+10.9f} | MSE = {err:10.9f}"
        )

        # Convergence check
        if abs(grad) <= epsilon:
            print("\n Converged: gradient below epsilon threshold.")
            print(f"Optimal Solution : \n"
            f"Iteration {iteration:3d} | x = {x:+10.9f} "
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
    """Execute the gradient descent animation."""
    animate_gradient_descent()


if __name__ == "__main__":
    main()
