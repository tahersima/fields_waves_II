# -*- coding: utf-8 -*-
"""
Propagation of an electromagentic wave in a lossy medium 
Created in Dec 2025

@author: MOHAMMAD H. TAHERSIMA
ALL RIGHTS RESERVED
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import os
os.chdir(r"...\testfolder")

# lossy medium settings
kr = 1.0
ki = 1.0
phi = np.pi / 4  # Phase shift between E and H

xmax = 15
xmin = 0
delx = 0.1
x = np.arange(xmin, xmax, delx)

# Animation settings
framemax = 100
fps = 30

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()  
    # Time variable based on frame
    t = 2 * np.pi * frame / framemax
    
    # Calculate fields
    attenuation = np.exp(-0.1 * x * ki)
    E_vals = attenuation * np.cos(kr * x - t)
    H_vals = attenuation * np.cos(kr * x - t - phi) # Note: subtraction is for the phase delay
    S_vals = E_vals * H_vals # Poynting vector (Power)

    # Plotting E-field (Vertical Plane)
    ax.plot(x, np.zeros_like(x), E_vals, 'r', lw=2, label='E-field (Electric)')
    # Add "combs" to visualize the wave vector
    for i in range(0, len(x), 1):
        ax.plot([x[i], x[i]], [0, 0], [0, E_vals[i]], 'r', alpha=0.3)

    # Plotting H-field (Horizontal Plane)
    ax.plot(x, H_vals, np.zeros_like(x), 'b', lw=2, label='H-field (Magnetic)')
    for i in range(0, len(x), 1):
        ax.plot([x[i], x[i]], [0, H_vals[i]], [0, 0], 'b', alpha=0.3)

    # Plotting Power Density (S)
    ax.plot(x, np.zeros_like(x), S_vals, 'k', lw=3, label='Power Density (S)')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    
    ax.set_xlabel('Propagation Direction (x)')
    ax.set_ylabel('H-axis')
    ax.set_zlabel('E-axis')
    
    ax.set_title(f'EM Wave in Lossy Media (Phase Shift $\phi$ = {phi/np.pi:.2f}$\pi$)', fontsize=14)
    ax.legend(loc='upper right')
    ax.view_init(elev=20, azim=-35) # Better perspective

# Generate animation
ani = FuncAnimation(fig, update, frames=framemax, interval=50)
ani.save('wave_propagation_lossy.gif', writer='pillow', fps=20)
