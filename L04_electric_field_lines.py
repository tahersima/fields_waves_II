# -*- coding: utf-8 -*-
"""
@author: tahersima
all rights reserved 
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_electric_field(charges, x_lim=(-5, 5), y_lim=(-5, 5), resolution=500, title="Electric Field"):
    """
    Computes and plots electric field lines for a given distribution of point charges.

    charges (list of tuples): A list where each element is (q, x, y)
                              q = magnitude of charge (Coulombs/arbitrary units)
                              x, y = coordinates of the charge
                              x_lim, y_lim (tuple): The spatial range of the plot (min, max).
    resolution (int): The number of grid points for calculation.
    """
    
    # grid of points of the simulation space
    x = np.linspace(x_lim[0], x_lim[1], resolution)
    y = np.linspace(y_lim[0], y_lim[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Initialize Electric Field Vector Components (Ex, Ey)
    Ex = np.zeros(X.shape)
    Ey = np.zeros(Y.shape)

    k = 8.99e9 # Coulomb's Constant

    # Compute Electric Field magnitudes at every grid point via Superposition
    for q, cx, cy in charges:
        # Vector from charge (cx, cy) to grid point (X, Y)
        dx = X - cx
        dy = Y - cy
        
        # Distance squared
        r2 = dx**2 + dy**2
        # Avoid division by zero at the exact location of the charge
        r2[r2 == 0] = 1e-12 
        # electric field: E_vec = k * q * (dx, dy) / r^3
        r3 = r2 ** 1.5
        
        Ex += k * q * dx / r3
        Ey += k * q * dy / r3

    # 3. Visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Calculate magnitude for line density normalization (optional but helps visualization)
    E_mag = np.sqrt(Ex**2 + Ey**2)
    print(E_mag.shape)
    
    # Plot Field Lines (Streamplot)
    strm = ax.streamplot(x, y, Ex, Ey, color='black', linewidth=1, 
                         density=1.5, arrowstyle='->', arrowsize=0.5,  
                         broken_streamlines=False)

    # Plot the Point Charges
    for q, cx, cy in charges:
        if q > 0:
            color = 'red'
            marker = 'o'
            sign = '+'
        else:
            color = 'blue'
            marker = 'o'
            sign = '-'
        
        # Size of marker roughly proportional to magnitude (log scale for visibility)
        size = 100 * (abs(q)**0.5) if abs(q) > 0 else 100
        
        ax.scatter(cx, cy, s=size, c=color, edgecolors='black', zorder=10)
        # Add sign text
        ax.text(cx, cy, sign, color='white', ha='center', va='center', 
                fontweight='bold', fontsize=9, zorder=11)

    ax.set_aspect('equal')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.set_title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()

# ==========================================
# Example 1: Point Charges
# ==========================================
charges_ex1 = [
    (1, -2, 0),   
    (1, 2, 0),    
    (-1, 0, 2),
    (-1.5, -1, -1)    
]

plot_electric_field(charges_ex1, 
                    title="Example 1: Two Positive, One Negative Charge")


# ==========================================
# Example 2: Parallel Plate Capacitor Approximation
# Approximating continuous plates using arrays of point charges
# ==========================================
charges_ex2 = []
plate_length = 4
num_points = 100 # Density of charges to simulate a continuous line
charge_density = 1.0

# Generate Top Plate (Positive) at y = 1.5
x_positions = np.linspace(-plate_length/2, plate_length/2, num_points)
for x in x_positions:
    charges_ex2.append((charge_density, x, 1.5))

# Generate Bottom Plate (Negative) at y = -1.5
for x in x_positions:
    charges_ex2.append((-charge_density, x, -1.5))

plot_electric_field(charges_ex2, 
                    title="Example 2: Parallel Plate Capacitor (Linear Distributions)")
