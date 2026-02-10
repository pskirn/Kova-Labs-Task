"""
2D Pose Graph SLAM Optimization using Levenberg-Marquardt
Implements robust optimization with Cauchy kernel for outlier rejection
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import math
import copy



# DATA STRUCTURES

class Pose2D:
    """Represents a 2D pose with position (x, y) and orientation (theta)"""
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def to_array(self):
        """Convert pose to numpy array [x, y, theta]"""
        return np.array([self.x, self.y, self.theta])


class Edge:
    """Represents a constraint between two poses"""
    def __init__(self, from_id, to_id, measurement, information):
        self.from_id = from_id          # Source pose ID
        self.to_id = to_id              # Target pose ID
        self.measurement = measurement   # Relative pose measurement [dx, dy, dtheta]
        self.information = information   # Information matrix (inverse covariance)



# UTILITY FUNCTIONS

def normalize_angle(angle):
    """
    Normalize angle to [-pi, pi] using atan2 for numerical stability
    
    Args:
        angle: Angle in radians
    Returns:
        Normalized angle in [-pi, pi]
    """
    return math.atan2(math.sin(angle), math.cos(angle))


def parse_g2o(filename):
    """
    Parse a g2o format pose graph file
    
    Args:
        filename: Path to .g2o file
    Returns:
        poses: Dictionary mapping pose_id -> Pose2D
        edges: List of Edge objects
    """
    poses = {}
    edges = []

    with open(filename, 'r') as file:
        for line in file:
            data = line.strip().split()
            
            if not data:
                continue

            if data[0] == "VERTEX_SE2":
                pose_id = int(data[1])
                x = float(data[2])
                y = float(data[3])
                theta = float(data[4])
                poses[pose_id] = Pose2D(x, y, theta)

            elif data[0] == "EDGE_SE2":
                from_id = int(data[1])
                to_id = int(data[2])
                
                # Measurement: relative pose transformation
                dx = float(data[3])
                dy = float(data[4])
                dtheta = float(data[5])
                measurement = np.array([dx, dy, dtheta])
                
                # Information matrix 
                info_11 = float(data[6])
                info_12 = float(data[7])
                info_13 = float(data[8])
                info_22 = float(data[9])
                info_23 = float(data[10])
                info_33 = float(data[11])
                
                information = np.array([
                    [info_11, info_12, info_13],
                    [info_12, info_22, info_23],
                    [info_13, info_23, info_33]
                ])
                
                edges.append(Edge(from_id, to_id, measurement, information))

    return poses, edges



# POSE GRAPH SLAM CORE FUNCTIONS

def compute_relative_pose(pose_i, pose_j):
    """
    Compute relative pose from pose_i to pose_j in pose_i's local frame
    
    This transforms the global pose difference into pose_i's coordinate frame
    using a 2D rotation matrix.
    
    Args:
        pose_i: Source pose (Pose2D or array [x, y, theta])
        pose_j: Target pose (Pose2D or array [x, y, theta])
    Returns:
        Relative pose [dx, dy, dtheta] in pose_i's frame
    """
    # Extract coordinates
    if isinstance(pose_i, Pose2D):
        x_i, y_i, theta_i = pose_i.x, pose_i.y, pose_i.theta
        x_j, y_j, theta_j = pose_j.x, pose_j.y, pose_j.theta
    else:
        x_i, y_i, theta_i = pose_i
        x_j, y_j, theta_j = pose_j

    # Global frame difference
    dx_global = x_j - x_i
    dy_global = y_j - y_i
    
    # Rotate into pose_i's local frame
    c_i = math.cos(theta_i)
    s_i = math.sin(theta_i)

    dx = c_i * dx_global + s_i * dy_global
    dy = -s_i * dx_global + c_i * dy_global
    
    # Angle difference
    dtheta = normalize_angle(theta_j - theta_i)

    return np.array([dx, dy, dtheta])


def compute_error(pose_i, pose_j, measurement):
    """
    Compute error between predicted and measured relative pose
    
    Args:
        pose_i: Source pose
        pose_j: Target pose
        measurement: Expected relative pose [dx, dy, dtheta]
    Returns:
        Error vector [e_x, e_y, e_theta]
    """
    predicted = compute_relative_pose(pose_i, pose_j)
    error = predicted - measurement
    
    # Normalize angular error to [-pi, pi]
    error[2] = normalize_angle(error[2])
    
    return error


def compute_jacobians(pose_i, pose_j):
    """
    Compute Jacobians of relative pose error w.r.t. poses i and j
    
    These are the partial derivatives ∂e/∂pose_i and ∂e/∂pose_j
    needed for the linearization in Gauss-Newton/LM optimization.
    
    Args:
        pose_i: Source pose
        pose_j: Target pose
    Returns:
        J_i: 3x3 Jacobian w.r.t. pose_i
        J_j: 3x3 Jacobian w.r.t. pose_j
    """
    theta_i = pose_i.theta
    c_i = math.cos(theta_i)
    s_i = math.sin(theta_i)
    
    dx_global = pose_j.x - pose_i.x
    dy_global = pose_j.y - pose_i.y

    # Jacobian w.r.t. pose_i [x_i, y_i, theta_i]
    J_i = np.array([
        [-c_i, -s_i, -s_i * dx_global + c_i * dy_global],
        [ s_i, -c_i, -c_i * dx_global - s_i * dy_global],
        [ 0,  0, -1]
    ])

    # Jacobian w.r.t. pose_j [x_j, y_j, theta_j]
    J_j = np.array([
        [ c_i,  s_i,  0],
        [-s_i,  c_i,  0],
        [ 0,  0,  1]
    ])

    return J_i, J_j


def build_linear_system_robust_cauchy(poses, edges, id2idx, fix_first_pose=True, delta=5.0):
    """
    Build the linear system H*Δx = -b for pose graph optimization
    with Cauchy robust kernel for outlier rejection
    
    The Cauchy kernel downweights large errors to reduce the influence
    of outliers (e.g., bad loop closures).
    
    Args:
        poses: Dictionary of current poses
        edges: List of edges (constraints)
        id2idx: Mapping from pose ID to index in parameter vector
        fix_first_pose: Whether to anchor the first pose
        delta: Cauchy kernel threshold parameter
    Returns:
        H: Approximate Hessian matrix (n_params x n_params)
        b: Gradient vector (n_params,)
    """
    n_poses = len(poses)
    n_params = n_poses * 3  # Each pose has 3 DOF: x, y, theta
    
    H = np.zeros((n_params, n_params))
    b = np.zeros(n_params)

    # Accumulate contributions from all edges
    for edge in edges:
        i = edge.from_id
        j = edge.to_id
        idx_i = id2idx[i] * 3
        idx_j = id2idx[j] * 3

        pose_i = poses[i]
        pose_j = poses[j]

        # Compute error and Jacobians
        e = compute_error(pose_i, pose_j, edge.measurement)
        J_i, J_j = compute_jacobians(pose_i, pose_j)
        omega = edge.information
        
        # Cauchy robust kernel: weight = 1 / (1 + chi^2 / delta^2)
        # This downweights edges with large errors (potential outliers)
        chi2 = e.T @ omega @ e
        weight = 1.0 / (1.0 + chi2 / (delta**2))
        
        # Weighted contributions (robust least squares)
        omega_J_i = weight * (omega @ J_i)
        omega_J_j = weight * (omega @ J_j)
        omega_e = weight * (omega @ e)

        # Update H (approximate Hessian = J^T * Ω * J)
        H[idx_i:idx_i+3, idx_i:idx_i+3] += J_i.T @ omega_J_i
        H[idx_i:idx_i+3, idx_j:idx_j+3] += J_i.T @ omega_J_j
        H[idx_j:idx_j+3, idx_i:idx_i+3] += J_j.T @ omega_J_i
        H[idx_j:idx_j+3, idx_j:idx_j+3] += J_j.T @ omega_J_j

        # Update b (gradient = J^T * Ω * e)
        b[idx_i:idx_i+3] += J_i.T @ omega_e
        b[idx_j:idx_j+3] += J_j.T @ omega_e

    # Gauge fixing: Add strong prior on first pose to remove gauge freedom
    # This prevents the entire graph from drifting
    if fix_first_pose:
        H[0:3, 0:3] += 1e9 * np.eye(3)

    return H, b


def compute_total_error(poses, edges):
    """
    Compute total chi-squared error across all edges
    
    Args:
        poses: Dictionary of poses
        edges: List of edges
    Returns:
        Total chi-squared error (scalar)
    """
    total_error = 0.0
    
    for edge in edges:
        pose_i = poses[edge.from_id]
        pose_j = poses[edge.to_id]
        
        error = compute_error(pose_i, pose_j, edge.measurement)
        chi2 = error.T @ edge.information @ error
        total_error += chi2
    
    return total_error


def optimize_pose_graph(poses, edges, max_iterations=50, lambda_init=0.01, 
                       tolerance=1e-6, delta_cauchy=5.0):
    """
    Optimize pose graph using Levenberg-Marquardt with Cauchy robust kernel
    
    LM adaptively interpolates between Gauss-Newton (fast near optimum) and
    gradient descent (stable far from optimum) using damping parameter λ.
    
    Args:
        poses: Initial poses dictionary
        edges: List of edges
        max_iterations: Maximum optimization iterations
        lambda_init: Initial LM damping parameter
        tolerance: Convergence tolerance on error reduction
        delta_cauchy: Cauchy kernel scale parameter
    Returns:
        poses: Optimized poses
        error_history: List of errors per iteration
    """
    print("\n" + "="*60)
    print("POSE GRAPH OPTIMIZATION - LEVENBERG-MARQUARDT")
    print("="*60)
    print(f"Poses: {len(poses)} | Edges: {len(edges)}")
    
    # Create pose ID to index mapping for consistent indexing
    id_list = sorted(poses.keys())
    id2idx = {pid: idx for idx, pid in enumerate(id_list)}

    # Compute initial error
    initial_error = compute_total_error(poses, edges)
    print(f"Initial error: {initial_error:.2f}")
    
    # LM parameters
    lambda_factor = 10.0
    current_lambda = lambda_init
    current_error = initial_error
    error_history = [initial_error]

    start_time = time.time()

    for iteration in range(max_iterations):
        # Build linear system with robust kernel
        H, b = build_linear_system_robust_cauchy(
            poses, edges, id2idx, 
            fix_first_pose=True, 
            delta=delta_cauchy
        )

        # Apply LM damping: H_lm = H + λ * diag(H)
        H_lm = H + current_lambda * np.diag(np.diag(H))

        # Solve linear system: H_lm * Δx = -b
        try:
            delta_x = np.linalg.solve(H_lm, -b)
        except np.linalg.LinAlgError:
            print(f"Iter {iteration:3d}: Singular matrix, increasing λ")
            current_lambda *= lambda_factor
            continue

        # Apply updates to create new pose estimates
        updated_poses = {}
        for pid, idx in id2idx.items():
            base = idx * 3
            p = poses[pid]
            dx, dy, dtheta = delta_x[base:base + 3]
            updated_poses[pid] = Pose2D(
                p.x + dx,
                p.y + dy,
                normalize_angle(p.theta + dtheta)
            )

        # Compute new error
        new_error = compute_total_error(updated_poses, edges)

        # LM update strategy
        if new_error < current_error:
            # Accept update
            poses = updated_poses
            error_reduction = current_error - new_error
            current_error = new_error
            error_history.append(current_error)
            
            # Decrease damping (move toward Gauss-Newton)
            current_lambda /= lambda_factor

            print(f"Iter {iteration:3d}: Error = {current_error:12.2f} | "
                  f"Reduction = {error_reduction:10.2f} | λ = {current_lambda:.2e}")

            # Check convergence
            if error_reduction < tolerance:
                print("\n✓ Converged: Error reduction below tolerance")
                break

        else:
            # Reject update, increase damping (move toward gradient descent)
            current_lambda *= lambda_factor
            print(f"Iter {iteration:3d}: Rejected (error increased) | λ = {current_lambda:.2e}")

            if current_lambda > 1e10:
                print("\n✗ Stopping: Damping parameter too large")
                break

    end_time = time.time()

    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Final error:      {current_error:.2f}")
    print(f"Error reduction:  {initial_error - current_error:.2f} "
          f"({100*(initial_error-current_error)/initial_error:.4f}%)")
    print(f"Iterations:       {iteration + 1}")
    print(f"Time:             {end_time - start_time:.3f} seconds")
    print(f"Avg error/edge:   {current_error / len(edges):.2f}")
    print(f"RMS error/edge:   {np.sqrt(current_error / len(edges)):.2f}")
    print("="*60)

    return poses, error_history



# VISUALIZATION AND ANALYSIS

def plot_trajectory(poses, edges=None, title="Trajectory", show_edges=False):
    """Plot 2D trajectory with optional edge visualization"""
    pose_ids = sorted(poses.keys())
    x_coords = [poses[i].x for i in pose_ids]
    y_coords = [poses[i].y for i in pose_ids]
    
    plt.figure(figsize=(9, 9))
    
    # Plot edges (constraints)
    if show_edges and edges is not None:
        for edge in edges:
            i, j = edge.from_id, edge.to_id
            plt.plot([poses[i].x, poses[j].x], 
                    [poses[i].y, poses[j].y], 
                    'gray', alpha=0.2, linewidth=0.5)
    
    # Plot trajectory
    plt.plot(x_coords, y_coords, 'b-', linewidth=2, label='Trajectory')
    plt.plot(x_coords[0], y_coords[0], 'go', markersize=12, label='Start', zorder=5)
    plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=12, label='End', zorder=5)
    
    plt.xlabel('x [m]', fontsize=12)
    plt.ylabel('y [m]', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()


def plot_comparison(poses_before, poses_after, error_history):
    """Create comprehensive before/after comparison plot"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    
    pose_ids = sorted(poses_before.keys())
    x_before = [poses_before[i].x for i in pose_ids]
    y_before = [poses_before[i].y for i in pose_ids]
    x_after = [poses_after[i].x for i in pose_ids]
    y_after = [poses_after[i].y for i in pose_ids]
    
    # Before optimization
    axes[0].plot(x_before, y_before, 'b-', linewidth=2)
    axes[0].plot(x_before[0], y_before[0], 'go', markersize=10)
    axes[0].set_xlabel('x [m]', fontsize=11)
    axes[0].set_ylabel('y [m]', fontsize=11)
    axes[0].set_title('Before Optimization', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # After optimization
    axes[1].plot(x_after, y_after, 'r-', linewidth=2)
    axes[1].plot(x_after[0], y_after[0], 'go', markersize=10)
    axes[1].set_xlabel('x [m]', fontsize=11)
    axes[1].set_ylabel('y [m]', fontsize=11)
    axes[1].set_title('After Optimization', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')
    
    # Convergence history
    axes[2].semilogy(error_history, 'b-', linewidth=2, marker='o', markersize=5)
    axes[2].set_xlabel('Iteration', fontsize=11)
    axes[2].set_ylabel('Total Chi-Squared Error (log scale)', fontsize=11)
    axes[2].set_title('Convergence History', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def analyze_results(poses_initial, poses_optimized, edges):
    """Analyze and print optimization results"""
    print("\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)
    
    # Edge type analysis
    sequential = sum(1 for e in edges if abs(e.to_id - e.from_id) == 1)
    loop_closures = len(edges) - sequential
    print(f"\nEdge Statistics:")
    print(f"  Sequential (odometry): {sequential}")
    print(f"  Loop closures:         {loop_closures}")
    print(f"  Total edges:           {len(edges)}")
    
    # Pose movement analysis
    distances = []
    angle_changes = []
    for pose_id in poses_initial.keys():
        dx = poses_optimized[pose_id].x - poses_initial[pose_id].x
        dy = poses_optimized[pose_id].y - poses_initial[pose_id].y
        dtheta = poses_optimized[pose_id].theta - poses_initial[pose_id].theta
        
        dist = np.sqrt(dx**2 + dy**2)
        distances.append(dist)
        angle_changes.append(abs(normalize_angle(dtheta)))
    
    print(f"\nPose Corrections:")
    print(f"  Max position change:  {max(distances):.2f} m")
    print(f"  Mean position change: {np.mean(distances):.2f} m")
    print(f"  Max angle change:     {np.rad2deg(max(angle_changes)):.2f}°")

    # Large error analysis
    threshold = 5.0
    large_errors = []
    for edge in edges:
        error = compute_error(poses_optimized[edge.from_id], 
                            poses_optimized[edge.to_id], 
                            edge.measurement)
        chi2 = error.T @ edge.information @ error
        if chi2 > threshold:
            large_errors.append((edge, chi2))

    print(f"\nLarge Errors (χ² > {threshold}):")
    print(f"  Found {len(large_errors)} edges with large errors")

    if large_errors:
        large_errors.sort(key=lambda x: x[1], reverse=True)
        print(f"  Top 10 largest errors:")
        for i, (edge, chi2) in enumerate(large_errors[:10], 1):
            edge_type = "odometry" if abs(edge.to_id - edge.from_id) == 1 else "loop"
            print(f"    {i:2d}. Edge {edge.from_id:3d}→{edge.to_id:3d} ({edge_type:8s}): χ² = {chi2:8.2f}")



# MAIN EXECUTION

def main():
    """Main execution function"""
    
    # Load data
    filename = "input_MITb_g2o.g2o"
    poses, edges = parse_g2o(filename)
    print(f"Loaded: {len(poses)} poses, {len(edges)} edges from {filename}")
    
    # Keep copy of initial poses for comparison
    poses_initial = copy.deepcopy(poses)
    
    # Run optimization
    poses_optimized, error_history = optimize_pose_graph(
        poses, 
        edges,
        max_iterations=50,
        lambda_init=0.01,
        tolerance=1e-6,
        delta_cauchy=5.0  # Cauchy robust kernel parameter
    )
    
    # Analyze results
    analyze_results(poses_initial, poses_optimized, edges)
    
    # Visualizations
    print("\nGenerating plots...")
    
    plot_trajectory(poses_initial, edges, "Initial Trajectory", show_edges=True)
    plt.savefig('trajectory_initial.png', dpi=150, bbox_inches='tight')
    
    plot_trajectory(poses_optimized, edges, "Optimized Trajectory", show_edges=True)
    plt.savefig('trajectory_optimized.png', dpi=150, bbox_inches='tight')
    
    plot_comparison(poses_initial, poses_optimized, error_history)
    plt.savefig('optimization_comparison.png', dpi=150, bbox_inches='tight')
    
    plt.show()
    
    print("✓ Plots saved successfully")
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()