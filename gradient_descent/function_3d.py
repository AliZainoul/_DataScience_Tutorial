"""
3D Surface Plot with Gradient Field
-----------------------------------

This script visualizes a 3D function and overlays its gradient field
on a plane, using modern Matplotlib APIs (3.7+).
"""

# from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.figure import Figure
from matplotlib import colormaps
from matplotlib.colors import Colormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the function f(x, y)."""
    return x * np.exp(-x**2 - y**2) + (x**2 + y**2) / 20


def gradient_function(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the gradient ∇f."""
    exp_term = np.exp(-(x**2 + y**2))
    df_dx = exp_term - 2 * x**2 * exp_term + x / 10
    df_dy = -2 * x * y * exp_term + y / 10
    return df_dx, df_dy


def plot_surface_and_gradient() -> None:
    """Plot the function surface and its gradient field."""
    # --- Meshgrid ---
    x_vals: np.ndarray = np.arange(-3, 3, 0.2)
    y_vals: np.ndarray = np.arange(-3, 3, 0.2)
    x: np.ndarray
    y: np.ndarray
    x, y = np.meshgrid(x_vals, y_vals)
    z: np.ndarray = function(x, y)

    # --- Figure ---
    fig: Figure = plt.figure(figsize=(9, 7))
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=-29, elev=49)

    # --- Surface Plot (new API) ---
    cmap: Colormap = colormaps["viridis"]
    ax.plot_surface(
        x,
        y,
        z,
        cmap=cmap,
        edgecolor="none",
        rstride=1,
        cstride=1,
        alpha=0.9,
    )

    ax.set_xlabel("Parameter 1 (x)")
    ax.set_ylabel("Parameter 2 (y)")
    ax.set_zlabel("f(x, y)")

    # --- Gradient Field ---
    df_dx: np.ndarray
    df_dy: np.ndarray
    df_dx, df_dy = gradient_function(x, y)
    z_plane: np.ndarray = np.full_like(x, -1.0)
    zeros: np.ndarray = np.zeros_like(df_dx)

    # Pylance expects 'length: int' but Matplotlib accepts float.
    LENGTH: int | float = float(0.15)

    quiver_obj: Line3DCollection = ax.quiver(
        x,
        y,
        z_plane,
        df_dx,
        df_dy,
        zeros,
        length=LENGTH,
        normalize=True,
        color="#333333",
    )

    plt.tight_layout()
    plt.show()


def main() -> None:
    plot_surface_and_gradient()


if __name__ == "__main__":
    main()
