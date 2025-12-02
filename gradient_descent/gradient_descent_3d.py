"""
2D Gradient Descent with Momentum on a Smooth Surface
-----------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union
from numpy.typing import NDArray

# --------------------------------------------------------------------------- #
# Type Alias
# --------------------------------------------------------------------------- #

Vector = Union[float, NDArray[np.float64]]


# --------------------------------------------------------------------------- #
# Mathematical Functions
# --------------------------------------------------------------------------- #

def function(x: Vector, y: Vector) -> Vector:
    """Compute f(x, y) = x e^{-x²-y²} + (x² + y²)/20."""
    return x * np.exp(-x**2 - y**2) + (x**2 + y**2) / 20


def gradient_function(x: Vector, y: Vector) -> Tuple[Vector, Vector]:
    """
    Compute gradient ∂f/∂x and ∂f/∂y.
    """
    exp_term = np.exp(-x**2 - y**2)

    g_x = exp_term * (1 - 2 * x**2) + x / 10
    g_y = exp_term * (-2 * x * y) + y / 10

    return g_x, g_y


# --------------------------------------------------------------------------- #
# Gradient Descent with Momentum
# --------------------------------------------------------------------------- #

def animate_gradient_descent_momentum(
    learning_rate: float = 0.25,
    momentum: float = 0.6,
    epsilon: float = 1e-6,
    max_iterations: int = 200,
) -> None:

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=49, azim=-29)

    # Surface mesh (arrays)
    X = np.arange(-3, 3.2, 0.2)
    Y = np.arange(-3, 3.2, 0.2)
    X, Y = np.meshgrid(X, Y)
    Z : Vector = function(X, Y)

    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color="gray", alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.set_title("Gradient Descent with Momentum (2D Function)")

    # Scalars
    x: float = float(np.random.randint(-2, 3) + np.random.rand())
    y: float = float(np.random.randint(-2, 3) + np.random.rand())

    corr_x: float = 0.0
    corr_y: float = 0.0

    for iteration in range(max_iterations):

        ax.scatter(x, y, function(x, y), color="red", s=40)
        plt.pause(0.15)

        g_x, g_y = gradient_function(x, y)
        grad_norm = float(np.sqrt(g_x**2 + g_y**2))

        print(
            f"Iteration {iteration:3d} | "
            f"x = {x:+8.5f} | y = {y:+8.5f} | "
            f"||grad|| = {grad_norm:8.5f}"
        )

        if grad_norm <= epsilon:
            print("\nConverged: gradient norm below epsilon.")
            break

        # Momentum update
        corr_x = momentum * corr_x - learning_rate * float(g_x)
        corr_y = momentum * corr_y - learning_rate * float(g_y)

        x += corr_x
        y += corr_y

    plt.show()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    animate_gradient_descent_momentum()


if __name__ == "__main__":
    main()
