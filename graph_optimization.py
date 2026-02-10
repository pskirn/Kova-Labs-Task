# ============================================================================
# 2D Pose Graph Optimization Interview Task - INSTRUCTIONS BELOW
#
# This file is provided as a template for your implementation.
#
# Instructions:
#   - Implement a simple 2D pose graph optimization backend (for SLAM)
#   - Read a g2o file of poses/edges, optimize using Gauss-Newton or LM, plot results.
#   - You may use NumPy/SciPy, but NOT any existing SLAM/simultaneous localization libraries.
#   - Fix the first pose.
#   - Visualize the trajectory before and after, report convergence/summary.
#   - Bonus: Robust loss (e.g. Huber), GN-vs-LM comparison, residual plots.
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt

def read_g2o_poses_edges(filename):
    """
    Reads 2D poses and edges from a G2O file in the SE2 format.

    Parameters
    ----------
    filename : str
        Path to the .g2o file containing the pose graph data.

    Returns
    -------
    poses : np.ndarray, shape (N, 3)
        Array of robot poses, where each pose is [x, y, theta].
    edges : list of tuples
        Each tuple corresponds to an edge (constraint) in the pose graph and has the form:
        (i, j, dx, dy, dtheta, info)
        where
            i : int
                Index of the first pose.
            j : int
                Index of the second pose.
            dx, dy, dtheta : float
                Relative pose measurement from i to j.
            info : ndarray, shape (3,3)
                Information matrix associated with this constraint.
    """
    poses = []
    edges = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('VERTEX_SE2'):
                _, i, x, y, theta = line.split()
                poses.append([float(x), float(y), float(theta)])
            elif line.startswith('EDGE_SE2'):
                parts = line.split()
                i, j = int(parts[1]), int(parts[2])
                dx, dy, dtheta = float(parts[3]), float(parts[4]), float(parts[5])
                # Parse upper triangle (6 elements) of info matrix
                info_elements = [float(val) for val in parts[6:12]]
                info = np.zeros((3,3))
                idx = 0
                for row in range(3):
                    for col in range(row, 3):
                        info[row, col] = info_elements[idx]
                        if row != col:
                            info[col, row] = info_elements[idx]
                        idx += 1
                edges.append((i, j, dx, dy, dtheta, info))
    return np.array(poses), edges

def display_poses_edges(poses, edges):
    """
    Visualizes a 2D robot trajectory (poses) and the associated edges (constraints) of a pose graph.

    Parameters
    ----------
    Same as what is output by the g2o reading function: 
        - poses: np.ndarray of shape (N, 3)
        - edges: list of (i, j, dx, dy, dtheta, info) tuples

    The function plots:
        - The robot trajectory as blue circles connected by lines.
        - Each edge as a thin, semi-transparent gray line between the connected poses.
    """
    plt.figure(figsize=(8,8))
    plt.plot(poses[:,0], poses[:,1], 'bo-', markersize=4, label="Trajectory")
    for (i, j, dx, dy, dtheta, info) in edges:
        plt.plot([poses[i,0], poses[j,0]], [poses[i,1], poses[j,1]], color='gray', alpha=0.3, linewidth=0.8)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Pose Trajectory and Edges from input_MITb_g2o.g2o")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage (to visualize unoptimized input):
poses, edges = read_g2o_poses_edges("input_MITb_g2o.g2o")
display_poses_edges(poses, edges)
