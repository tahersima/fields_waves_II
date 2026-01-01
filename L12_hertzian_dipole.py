# -*- coding: utf-8 -*-
"""
Animates Hertzian Dipole

Created on Wed Dec 31 20:49:55 2025
@author: MOHAMMAD H. TAHERSIMA
ALL RIGHTS RESERVED
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
os.chdir(r"...\testfolder")

# Constants
range_of_simulation = 4 * np.pi
d = range_of_simulation / 50 + 0.001
number_of_frames = 30  # Number of frames

# Setup Grid
coords = np.arange(-range_of_simulation, range_of_simulation, d)
x, y = np.meshgrid(coords, coords)
r = np.sqrt(x**2 + y**2)

# Define Contour Levels
v = [-1, -0.9, -0.8, -0.75, -0.5, -0.25, -0.1, -0.05, 0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 1]

# Figure Setup
fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
ax.set_facecolor('black')

def update(frame):
    ax.clear()
    # Time variable based on frame
    t = frame * 2 * np.pi / number_of_frames
    
    # The Physics Formula
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = (x / r)**2
        term2 = -np.sin(r - t) - (np.cos(r - t) / r)
        z = term1 * term2

    # Plotting
    cp = ax.contour(x, y, z, levels=v, cmap='magma', linewidths=0.8)
    
    # Aesthetics
    ax.set_aspect('equal')
    ax.axis('off')
    return cp,

# Generate Animation
ani = FuncAnimation(fig, update, frames=number_of_frames, interval=50, blit=False)
plt.show()
ani.save('dipole.gif', writer='pillow', fps=20)
