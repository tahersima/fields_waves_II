# -*- coding: utf-8 -*-
"""
movement of charge partciles in magentic fields due to lorentzian force 
Created in Dec 2025

@author: MOHAMMAD H. TAHERSIMA
ALL RIGHTS RESERVED
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import os
os.chdir(r"...\testfolder")
# ==========================================
# Configuration & Physics Parameters
# ==========================================
width, height, depth = 8, 6, 6
box_x_offset = 1.0      # Gap from center to the box [m]
box_width = 6.0         # Thickness of the magnetic region [m]
B_strength = 10.0       # Strength of Magnetic Field [Wb]
particle_speed = 5.0    # Initial speed [m/s]
mass = 1.0
charge = 1.0            # Positive charge [C]
dt = 0.001              # Time step for physics [s]
frames = 400            # Total animation frames
steps_per_frame = 5     # Physics sub-steps per frame (for smoothness)

# ==========================================
# Physics Engine
# ==========================================
def get_b_field(pos):
    """
    Returns the Magnetic Field Vector B at position (x, y, z).
    There are two magnetic regions:
    1. Right Box (x > box_x_offset)
    2. Left Box (x < -box_x_offset)
    Both have B directed towards +Y axis (0, 1, 0).
    """
    x, y, z = pos
    
    # Check boundaries for Right Box
    if (box_x_offset <= x <= box_x_offset + box_width) and \
       (-height/2 <= y <= height/2) and \
       (-depth/2 <= z <= depth/2):
        return np.array([0.0, B_strength, 0.0])
    
    # Check boundaries for Left Box
    elif (-(box_x_offset + box_width) <= x <= -box_x_offset) and \
         (-height/2 <= y <= height/2) and \
         (-depth/2 <= z <= depth/2):
        return np.array([0.0, B_strength, 0.0])
    
    return np.array([0.0, 0.0, 0.0])

def update_physics(state):
    """
    Updates particle state [x,y,z, vx,vy,vz] using Lorentz force.
    Uses Runge-Kutta 4 (RK4) integration for stability.
    """
    pos = state[:3]
    vel = state[3:]
    
    def acceleration(v, p):
        B = get_b_field(p)
        # Lorentz Force: F = q(v x B) -> a = (q/m)(v x B)
        F = charge * np.cross(v, B)
        return F / mass

    # RK4 Integration
    k1_v = acceleration(vel, pos)
    k1_p = vel

    k2_v = acceleration(vel + 0.5 * k1_v * dt, pos + 0.5 * k1_p * dt)
    k2_p = vel + 0.5 * k1_v * dt

    k3_v = acceleration(vel + 0.5 * k2_v * dt, pos + 0.5 * k2_p * dt)
    k3_p = vel + 0.5 * k2_v * dt

    k4_v = acceleration(vel + k3_v * dt, pos + k3_p * dt)
    k4_p = vel + k3_v * dt

    new_vel = vel + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    new_pos = pos + (dt / 6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
    
    return np.concatenate((new_pos, new_vel))

# ==========================================
# Visualization Helpers
# ==========================================
def draw_box(ax, center, size, color):
    """Draws a semi-transparent 3D box representing the B-field."""
    x, y, z = center
    dx, dy, dz = size
    
    # Vertices of the cube
    xx = [x-dx/2, x+dx/2]
    yy = [y-dy/2, y+dy/2]
    zz = [z-dz/2, z+dz/2]
    
    vertices = [
        [[xx[0], yy[0], zz[0]], [xx[1], yy[0], zz[0]], [xx[1], yy[1], zz[0]], [xx[0], yy[1], zz[0]]], # Bottom
        [[xx[0], yy[0], zz[1]], [xx[1], yy[0], zz[1]], [xx[1], yy[1], zz[1]], [xx[0], yy[1], zz[1]]], # Top
        [[xx[0], yy[0], zz[0]], [xx[0], yy[1], zz[0]], [xx[0], yy[1], zz[1]], [xx[0], yy[0], zz[1]]], # Left
        [[xx[1], yy[0], zz[0]], [xx[1], yy[1], zz[0]], [xx[1], yy[1], zz[1]], [xx[1], yy[0], zz[1]]], # Right
        [[xx[0], yy[0], zz[0]], [xx[1], yy[0], zz[0]], [xx[1], yy[0], zz[1]], [xx[0], yy[0], zz[1]]], # Front
        [[xx[0], yy[1], zz[0]], [xx[1], yy[1], zz[0]], [xx[1], yy[1], zz[1]], [xx[0], yy[1], zz[1]]]  # Back
    ]
    
    # Draw faces
    poly = Poly3DCollection(vertices, alpha=0.1, facecolor=color, edgecolor=color, linewidths=0.5)
    ax.add_collection3d(poly)
    
    # Draw grid of B-field arrows inside
    ax_x = np.linspace(xx[0]+0.5, xx[1]-0.5, 3)
    ax_y = np.linspace(yy[0]+0.5, yy[1]-0.5, 3)
    ax_z = np.linspace(zz[0]+0.5, zz[1]-0.5, 3)
    
    mesh_x, mesh_y, mesh_z = np.meshgrid(ax_x, ax_y, ax_z)
    u = np.zeros_like(mesh_x)
    v = np.ones_like(mesh_x) * 0.8 # Pointing Y
    w = np.zeros_like(mesh_x)
    
    ax.quiver(mesh_x, mesh_y, mesh_z, u, v, w, length=0.8, color=color, alpha=0.4, arrow_length_ratio=0.4)
    
    # Label
    ax.text(x, y + dy/2 + 0.5, z, "$\\vec{B}$", color=color, fontsize=12, fontweight='bold', ha='center')

# ==========================================
# Simulation Setup
# ==========================================
fig = plt.figure(figsize=(10, 8), facecolor='white')
ax = fig.add_subplot(111, projection='3d')

# Initial State: Start at center, moving +x
initial_state = np.array([0.0, 0.0, 0.0, particle_speed, 0.0, 0.0])
state = initial_state.copy()
history_pos = [state[:3]]

# -- Setup Scene --
ax.set_xlim(-width/2, width/2)
ax.set_ylim(-height/2, height/2)
ax.set_zlim(-depth/2, depth/2)
ax.set_axis_off() # Clean look, no axes
ax.set_title("Lorentz Force Trap: $\\vec{F} = q(\\vec{v} \\times \\vec{B})$", fontsize=14)

# Draw Magnetic Field Regions
# Right Box
draw_box(ax, (box_x_offset + box_width/2, 0, 0), (box_width, height-1, depth-1), 'red')
# Left Box
draw_box(ax, (-(box_x_offset + box_width/2), 0, 0), (box_width, height-1, depth-1), 'red')

# -- Graphical Elements --
# Particle
particle_plot, = ax.plot([], [], [], 'o', color='blue', markersize=10, markeredgecolor='white', zorder=10)
# Trail
trail_plot, = ax.plot([], [], [], '-', color='cornflowerblue', linewidth=1.5, alpha=0.6)

# Vectors (Quivers)
# Velocity (Green)
q_vel = ax.quiver([0], [0], [0], [0], [0], [0], color='green', length=1.5, arrow_length_ratio=0.3, label='Velocity')
# Force (Black)
q_force = ax.quiver([0], [0], [0], [0], [0], [0], color='black', length=1.5, arrow_length_ratio=0.3, label='Force')

# Labels for vectors
txt_vel = ax.text(0,0,0, "$\\vec{v}$", color='green', fontweight='bold')
txt_force = ax.text(0,0,0, "$\\vec{F}$", color='black', fontweight='bold')

# ==========================================
# Animation Loop
# ==========================================
def update(frame):
    global state, history_pos, q_vel, q_force
    
    # Physics Sub-stepping
    for _ in range(steps_per_frame):
        state = update_physics(state)
        history_pos.append(state[:3])
    
    # Limit history length to prevent slowdown
    if len(history_pos) > 500:
        history_pos.pop(0)
        
    x, y, z = state[:3]
    vx, vy, vz = state[3:]
    
    # Calculate Force for visualization
    B_curr = get_b_field((x, y, z))
    F = charge * np.cross([vx, vy, vz], B_curr)
    
    # Update Particle
    particle_plot.set_data([x], [y])
    particle_plot.set_3d_properties([z])
    
    # Update Trail
    hist_arr = np.array(history_pos)
    trail_plot.set_data(hist_arr[:,0], hist_arr[:,1])
    trail_plot.set_3d_properties(hist_arr[:,2])
    
    # Update Vector Quivers (remove old, add new)
    # Matplotlib 3D quiver doesn't have a simple set_uvw, so we remove and redraw slightly 
    q_vel.remove()
    q_force.remove()
    
    # Normalize for display consistency
    v_norm = np.linalg.norm([vx, vy, vz])
    v_dir = np.array([vx, vy, vz]) / (v_norm if v_norm > 0 else 1)
    
    f_norm = np.linalg.norm(F)
    f_dir = F / (f_norm if f_norm > 0 else 1)
    
    # Only draw Force if significant
    f_len = 0.0
    if f_norm > 0.1:
        f_len = 1.5
    
    q_vel = ax.quiver([x], [y], [z], [v_dir[0]], [v_dir[1]], [v_dir[2]], 
                      color='green', length=1.5, arrow_length_ratio=0.3)
    
    q_force = ax.quiver([x], [y], [z], [f_dir[0]], [f_dir[1]], [f_dir[2]], 
                        color='black', length=f_len, arrow_length_ratio=0.3)
    
    # Update Text Labels
    txt_vel.set_position((x + v_dir[0], y + v_dir[1]))
    txt_vel.set_3d_properties(z + v_dir[2], zdir=None)
    
    txt_force.set_position((x + f_dir[0], y + f_dir[1]))
    txt_force.set_3d_properties(z + f_dir[2], zdir=None)
    
    # Rotate camera slightly for 3D effect
    ax.view_init(elev=20, azim=-60 + frame * 0.1)

    return particle_plot, trail_plot, q_vel, q_force, txt_vel, txt_force

ani = animation.FuncAnimation(fig, update, frames=frames, interval=20, blit=False)

plt.tight_layout()
plt.show()

# To save the video, uncomment the line below (requires ffmpeg)
ani.save('lorentz_oscillation.mp4', writer='ffmpeg', fps=30, dpi=150)
