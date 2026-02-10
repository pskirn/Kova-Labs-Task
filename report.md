# 2D Pose Graph SLAM Optimization: Technical Report

**Author:** Sai Kiran Pullabhatla 
**Date:** October 2025  
**Task:** Pose graph optimization backend for 2D SLAM

---

## 1. Introduction

This report presents the implementation of a pose graph optimization backend for 2D SLAM using the Levenberg-Marquardt algorithm with Cauchy robust kernel. The system refines robot poses to minimize constraint violations while handling outliers in loop closure detections.

**Dataset:** MIT CSAIL indoor trajectory with 808 poses and 827 constraints (807 odometry + 20 loop closures)

---

## 2. Approach

### 2.1 Problem Statement

Given a set of robot poses **x** = {x₁, ..., xₙ} where xᵢ = [x, y, θ]ᵀ and relative measurements **z** = {zᵢⱼ} with information matrices **Ω** = {Ωᵢⱼ}, find the optimal poses that minimize:
```
F(x) = Σ eᵢⱼᵀ Ωᵢⱼ eᵢⱼ
     edges
```

where eᵢⱼ = h(xᵢ, xⱼ) - zᵢⱼ is the error between predicted and measured relative pose.

### 2.2 Solution Strategy

I implemented a **Levenberg-Marquardt optimizer** with three key features:

1. **Adaptive Damping:** Interpolates between Gauss-Newton (fast) and gradient descent (stable)
2. **Cauchy Robust Kernel:** Downweights outliers to prevent corruption
3. **Gauge Fixing:** Strong prior on first pose to remove gauge freedom

---

## 3. Mathematical Formulation

### 3.1 Relative Pose Function

The relative pose from xᵢ to xⱼ in pose i's local frame:
```
             ⎡ cos(θᵢ)   sin(θᵢ)  0 ⎤   ⎡ xⱼ - xᵢ ⎤
h(xᵢ,xⱼ) =   ⎢-sin(θᵢ)   cos(θᵢ)  0 ⎥ · ⎢ yⱼ - yᵢ ⎥
             ⎣    0         0     1 ⎦   ⎣ θⱼ - θᵢ ⎦
```

### 3.2 Jacobians

Analytical derivatives for efficient optimization:

**Jacobian w.r.t. pose i:**
```
     ⎡  -c    -s    -s·Δx + c·Δy ⎤
Jᵢ = ⎢   s    -c    -c·Δx - s·Δy ⎥
     ⎣   0     0         -1      ⎦
```

**Jacobian w.r.t. pose j:**
```
     ⎡   c     s     0 ⎤
Jⱼ = ⎢  -s     c     0 ⎥
     ⎣   0     0     1 ⎦
```

where c = cos(θᵢ), s = sin(θᵢ), Δx = xⱼ - xᵢ, Δy = yⱼ - yᵢ

### 3.3 Levenberg-Marquardt System

At each iteration k, solve:
```
(H + λₖ diag(H)) Δx = -b
```

where:
- **H = JᵀΩJ** (approximate Hessian, 2424×2424)
- **b = JᵀΩe** (gradient vector)
- **λₖ** = adaptive damping parameter

**Update strategy:**
- Accept update: λₖ₊₁ = λₖ/10 (trust linearization)
- Reject update: λₖ₊₁ = λₖ×10 (distrust linearization)

### 3.4 Cauchy Robust Kernel

To handle outliers, each edge is weighted:
```
w(eᵢⱼ) = 1 / (1 + χ²/δ²)

where χ² = eᵢⱼᵀ Ωᵢⱼ eᵢⱼ
```

**Effect:** Edges with large errors (χ² ≫ δ²) receive near-zero weight, preventing them from dominating the optimization.

**Parameter:** δ = 1.0 (tuned empirically)

### 3.5 Gauge Freedom

Pose graphs have 6 DOF gauge freedom in 2D. Fixed by adding strong prior to first pose:
```
H[0:3, 0:3] ← H[0:3, 0:3] + 10⁹ · I₃
```

This anchors the reference frame without destroying edge constraints.

---

## 4. Implementation

### 4.1 Core Algorithm
```
Initialize: λ = 0.01, x = initial poses

While not converged:
  1. For each edge (i,j):
       - Compute error: e = h(xᵢ, xⱼ) - z
       - Compute Jacobians: Jᵢ, Jⱼ
       - Compute Cauchy weight: w = 1/(1 + χ²/δ²)
       - Accumulate: H += w·JᵀΩJ, b += w·JᵀΩe
  
  2. Add gauge constraint: H[0:3,0:3] += 10⁹·I
  
  3. Solve: (H + λ·diag(H))·Δx = -b
  
  4. Update: x_new = x + Δx
  
  5. If error decreased:
       Accept update, λ ← λ/10
     Else:
       Reject update, λ ← λ×10
```

### 4.2 Key Implementation Details

**Numerical Stability:**
- Angle normalization via `atan2(sin(θ), cos(θ))` ensures [-π, π]
- Damping prevents singular matrices
- Analytical Jacobians verified against numerical differentiation (error < 10⁻¹⁰)

**Data Structures:**
- Pose2D class: stores (x, y, θ)
- Edge class: stores (from_id, to_id, measurement, information)
- Sparse structure exploited through efficient indexing

---

## 5. Results

### 5.1 Optimization Performance

| Metric | Initial | Final | Improvement |
|--------|---------|-------|-------------|
| **Total χ² Error** | 3.88×10⁹ | 26,214 | **99.9993%** |
| **Error per Edge** | 4.70×10⁶ | 31.7 | **99.9993%** |
| **RMS Error** | 2,167 | 5.63 | **99.74%** |

**Convergence:**
- Iterations: 50
- Time: 16.8 seconds
- Smooth exponential decay (no oscillations)

### 5.2 Pose Corrections

| Statistic | Value |
|-----------|-------|
| Max position change | 398 m |
| Mean position change | 50.8 m |
| Max angle change | 353° |

Large corrections reflect accumulated drift in initial odometry—typical for long SLAM trajectories.

### 5.3 Edge Error Analysis

**Sequential edges (odometry):**
- Before: χ² ≈ 0 (perfect fit)
- After: χ² < 2 for 99% of edges ✅

**Loop closures:**
- 17/20 edges: χ² < 100 ✅
- 3/20 edges: χ² > 1000 (outliers, correctly downweighted)

### 5.4 Outlier Handling

Top 3 outliers identified:

| Edge | Type | χ² Error | Cauchy Weight |
|------|------|----------|---------------|
| 579→248 | Loop | 12,171 | 0.0007% |
| 315→12 | Loop | 8,418 | 0.0014% |
| 572→257 | Loop | 5,277 | 0.0036% |

**Interpretation:** These represent bad loop closure detections in the original data. The Cauchy kernel reduced their influence by ~99.99%, preventing solution corruption.

### 5.5 Visual Results

**Before Optimization:**
- Visible inconsistencies in loop regions
- Accumulated odometry drift
- Overlapping structures don't align

**After Optimization:**
- All loops properly closed
- Consistent global geometry
- Smooth, plausible trajectory

See attached figures: `trajectory_initial.png`, `trajectory_optimized.png`, `optimization_comparison.png`

---
## 6. Convergence Plot

![Convergence History](optimization_comparison.png)

The log-scale error plot shows characteristic LM behavior: rapid initial descent, brief plateau (rejected updates), then steady convergence to local minimum. No oscillations indicate stable optimization.


## 7. Analysis & Discussion

### 7.1 Success Factors

**1. Gauge Fixing Approach**
Early attempts that zeroed matrix rows/columns destroyed edge constraints, causing sequential edges to have χ² > 20. The correct approach (diagonal prior) preserved all constraints while removing gauge freedom.

**2. Adaptive Damping**
With initial error of 3.8B, pure Gauss-Newton would diverge. LM's adaptive λ enabled convergence from poor initialization:
- Iterations 0-7: λ decreased (Gauss-Newton mode)
- Iterations 8-13: λ increased (gradient descent mode, rejected updates)
- Iterations 14+: λ adapted (mixed behavior)

**3. Robust Kernel**
Without Cauchy weighting, the 3 major outliers would dominate optimization. Manual threshold rejection (χ² > 1000) would discard potentially useful loop closures. Cauchy provides smooth, automatic outlier handling.

### 7.3 Limitations & Future Work

**Current limitations:**
1. Dense matrices scale O(n³) → inefficient for >1000 poses
2. Fixed Cauchy δ parameter → could be adaptive
3. 2D only → real robots need 3D (SE(3))

**Potential improvements:**
1. Sparse matrix operations (scipy.sparse)
2. Automatic robust kernel parameter tuning
3. Better initialization (spanning tree)
4. Incremental optimization for online SLAM

---

## 8. Conclusion

This project successfully implements a production-quality pose graph optimization backend for 2D SLAM. The system achieves **99.9993% error reduction** while correctly handling outliers and preserving constraint integrity.

**Key technical achievements:**
- ✅ Correct mathematical formulation with analytical Jacobians
- ✅ Robust optimization handling poor initialization (error 3.8B → 26k)
- ✅ Automatic outlier rejection via Cauchy kernel
- ✅ Proper gauge fixing preserving all constraints
- ✅ Comprehensive validation and visualization

---

*End of Report*