"""
2D Gradient Descent with Momentum on a 3D Surface
-------------------------------------------------

Function:
    f(x, y) = x * exp(-x² - y²) + (x² + y²)/20

Features:
- 3D wireframe + contour plot
- Momentum-based gradient descent
- Animated 3D descent
- Fully typed (float or NDArray)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union
from numpy.typing import NDArray

# --------------------------------------------------------------------------- #
# Type alias
# --------------------------------------------------------------------------- #

Vector = Union[float, NDArray[np.float64]]


# --------------------------------------------------------------------------- #
# Mathematical Functions
# --------------------------------------------------------------------------- #

def function(x: Vector, y: Vector) -> Vector:
    """Return f(x, y) = x e^{-(x²+y²)} + (x²+y²)/20."""
    return x * np.exp(-x**2 - y**2) + (x**2 + y**2) / 20


def gradient_function(x: Vector, y: Vector) -> Tuple[Vector, Vector]:
    """
    Return gradient of f(x, y):

        ∂f/∂x = exp(-x²-y²)*(1 - 2x²) + x/10
        ∂f/∂y = -2xy * exp(-x²-y²) + y/10
    """
    exp_term = np.exp(-x**2 - y**2)

    g_x = exp_term * (1 - 2 * x**2) + x / 10
    g_y = exp_term * (-2 * x * y) + y / 10

    return g_x, g_y


# --------------------------------------------------------------------------- #
# Gradient Descent with Momentum + Animation
# --------------------------------------------------------------------------- #

def animate_gradient_descent_momentum(
    learning_rate: float = 0.2,
    momentum: float = 0.9,
    epsilon: float = 1e-6,
    max_iterations: int = 300,
) -> None:
    """Animate the momentum-based gradient descent on the 3D function."""

    # Figure + 3D AXIS
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=49, azim=-29)

    # Meshgrid for surface
    X = np.arange(-3, 3.2, 0.2)
    Y = np.arange(-3, 3.2, 0.2)
    X, Y = np.meshgrid(X, Y)
    Z = function(X, Y)

    # Wireframe + contour
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color="gray", alpha=0.7)
    ax.contour(X, Y, Z, 70, cmap="plasma", offset=None)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.set_title("Momentum Gradient Descent on 2D Function")

    # Random starting point
    x: float = float(np.random.randint(-2, 3) + np.random.rand())
    y: float = float(np.random.randint(-2, 3) + np.random.rand())

    # Momentum terms
    corr_x: float = 0.0
    corr_y: float = 0.0

    # Main optimization loop
    for i in range(max_iterations):

        # Plot current point
        ax.scatter(x, y, function(x, y), s=20, color="#FF0000")
        plt.pause(0.05)

        # Compute gradient
        g_x, g_y = gradient_function(x, y)
        grad_norm = float(np.sqrt(g_x**2 + g_y**2))

        print(f"Iteration {i:3d} -> x={x:+7.5f}  y={y:+7.5f}  |grad|={grad_norm:.5f}")

        # Convergence
        if grad_norm <= epsilon:
            print("\nConverged: gradient norm below threshold.")
            break

        # Momentum update
        corr_x = momentum * corr_x - learning_rate * float(g_x)
        corr_y = momentum * corr_y - learning_rate * float(g_y)

        x += corr_x
        y += corr_y

    plt.show()


# --------------------------------------------------------------------------- #
# Main Execution
# --------------------------------------------------------------------------- #

def main() -> None:
    animate_gradient_descent_momentum()


if __name__ == "__main__":
    main()
