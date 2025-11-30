#!/usr/bin/env python3
"""
Simple visualization script for PFEM 2D Sloshing results.
Reads VTK output files and creates plots.

Usage:
    python plot_results.py output/sloshing_*.vtk
    python plot_results.py output/sloshing_000100.vtk
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import re

def read_vtk(filename):
    """Read VTK file and extract mesh and field data."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    points = []
    cells = []
    velocity = []
    pressure = []
    boundary_type = []
    
    i = 0
    n_points = 0
    n_cells = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('POINTS'):
            parts = line.split()
            n_points = int(parts[1])
            i += 1
            for j in range(n_points):
                coords = list(map(float, lines[i+j].split()))
                points.append(coords[:2])  # Only x, y
            i += n_points
            
        elif line.startswith('CELLS') and not line.startswith('CELL_TYPES'):
            parts = line.split()
            n_cells = int(parts[1])
            i += 1
            for j in range(n_cells):
                cell_data = list(map(int, lines[i+j].split()))
                cells.append(cell_data[1:4])  # Skip first element (number of vertices)
            i += n_cells
            
        elif line.startswith('VECTORS velocity'):
            i += 1
            for j in range(n_points):
                vel = list(map(float, lines[i+j].split()))
                velocity.append(vel[:2])
            i += n_points
            
        elif line.startswith('SCALARS pressure'):
            i += 2  # Skip LOOKUP_TABLE line
            for j in range(n_points):
                p = float(lines[i+j].strip())
                pressure.append(p)
            i += n_points
            
        elif line.startswith('SCALARS boundary_type'):
            i += 2  # Skip LOOKUP_TABLE line
            for j in range(n_points):
                bt = int(lines[i+j].strip())
                boundary_type.append(bt)
            i += n_points
        else:
            i += 1
    
    return {
        'points': np.array(points),
        'cells': np.array(cells) if cells else None,
        'velocity': np.array(velocity) if velocity else None,
        'pressure': np.array(pressure) if pressure else None,
        'boundary_type': np.array(boundary_type) if boundary_type else None
    }

def plot_mesh(data, ax=None, color_by='velocity'):
    """Plot mesh with triangles colored by field."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    points = data['points']
    cells = data['cells']
    
    if cells is None or len(cells) == 0:
        ax.scatter(points[:, 0], points[:, 1], s=5, c='blue')
        return ax
    
    # Create triangles
    triangles = []
    for cell in cells:
        tri = points[cell]
        triangles.append(tri)
    
    # Color by field
    if color_by == 'velocity' and data['velocity'] is not None:
        vel_mag = np.sqrt(data['velocity'][:, 0]**2 + data['velocity'][:, 1]**2)
        # Average velocity magnitude per cell
        colors = []
        for cell in cells:
            colors.append(np.mean(vel_mag[cell]))
        colors = np.array(colors)
        cmap = 'viridis'
        label = 'Velocity magnitude [m/s]'
    elif color_by == 'pressure' and data['pressure'] is not None:
        colors = []
        for cell in cells:
            colors.append(np.mean(data['pressure'][cell]))
        colors = np.array(colors)
        cmap = 'coolwarm'
        label = 'Pressure [Pa]'
    else:
        colors = np.ones(len(cells)) * 0.5
        cmap = 'Blues'
        label = ''
    
    # Create polygon collection
    coll = PolyCollection(triangles, array=colors, cmap=cmap, 
                          edgecolors='black', linewidths=0.2)
    ax.add_collection(coll)
    
    ax.set_xlim(points[:, 0].min() - 0.05, points[:, 0].max() + 0.05)
    ax.set_ylim(points[:, 1].min() - 0.05, points[:, 1].max() + 0.05)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    
    if label:
        cbar = plt.colorbar(coll, ax=ax)
        cbar.set_label(label)
    
    return ax

def extract_time_step(filename):
    """Extract time step number from filename."""
    match = re.search(r'_(\d+)\.vtk$', filename)
    if match:
        return int(match.group(1))
    return 0

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <vtk_files>")
        print("Example: python plot_results.py output/sloshing_*.vtk")
        sys.exit(1)
    
    files = sorted(sys.argv[1:], key=extract_time_step)
    
    if len(files) == 1:
        # Single file - detailed plot
        filename = files[0]
        print(f"Reading {filename}...")
        data = read_vtk(filename)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        plot_mesh(data, axes[0], color_by='velocity')
        axes[0].set_title(f'Velocity magnitude\n{os.path.basename(filename)}')
        
        # Quiver plot
        points = data['points']
        velocity = data['velocity']
        if velocity is not None:
            # Subsample for clarity
            step = max(1, len(points) // 200)
            axes[1].quiver(points[::step, 0], points[::step, 1], 
                          velocity[::step, 0], velocity[::step, 1],
                          scale=50)
            axes[1].scatter(points[:, 0], points[:, 1], s=2, c='lightblue')
        axes[1].set_xlim(points[:, 0].min() - 0.05, points[:, 0].max() + 0.05)
        axes[1].set_ylim(points[:, 1].min() - 0.05, points[:, 1].max() + 0.05)
        axes[1].set_aspect('equal')
        axes[1].set_xlabel('x [m]')
        axes[1].set_ylabel('y [m]')
        axes[1].set_title('Velocity vectors')
        
        plt.tight_layout()
        plt.savefig('plot_single.png', dpi=150)
        print("Saved: plot_single.png")
        plt.show()
        
    else:
        # Multiple files - create overview
        n_files = min(6, len(files))
        indices = np.linspace(0, len(files)-1, n_files, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            filename = files[idx]
            print(f"Reading {filename}...")
            data = read_vtk(filename)
            
            plot_mesh(data, axes[i], color_by='velocity')
            step = extract_time_step(filename)
            axes[i].set_title(f'Step {step}')
        
        plt.suptitle('PFEM 2D Sloshing Simulation', fontsize=14)
        plt.tight_layout()
        plt.savefig('plot_overview.png', dpi=150)
        print("Saved: plot_overview.png")
        plt.show()

if __name__ == '__main__':
    main()
