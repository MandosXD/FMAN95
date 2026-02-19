"""
FMAN95 - Computer Vision - Utility Functions
=============================================

This file contains utility functions used throughout the FMAN95 course assignments.
Note that this file is shared among ALL assignments, so not all functions are
necessary for each individual assignment. You should use only the functions relevant
to your current assignment.

Contents:
---------
- Coordinate transformations (phom)
- Homogeneous line drawing (draw_hom_line)
- 3D plotting utilities (set_axes_equal, plotcam, plotcam_frustum)
- Image warping (homography_warp_image)
- Matrix decomposition (rq)
- Essential matrix estimation (fivepoint_solver)
- Bundle adjustment utilities (linearize_reprojection_error, update_solution, compute_reprojection_rms_error)

Changelog:
----------
2026-01-14: Initial implementation
2026-02-09: Bugfix in plotcam_frustum.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy


def phom(v):
    """
    Convert Euclidean coordinates to homogeneous coordinates.

    Adds a row (for 2D arrays) or element (for 1D arrays) of ones.

    Parameters:
    - v: ndarray of shape (d, N) or (d,) representing Euclidean coordinates

    Returns:
    - ndarray of shape (d+1, N) or (d+1,) with homogeneous coordinates

    Example:
    >>> v = np.array([[1, 2], [3, 4]])  # 2x2 array
    >>> phom(v)  # Returns 3x2 array with last row all ones
    """
    if not isinstance(v, np.ndarray):
        raise TypeError(f"Input must be a numpy array, got {type(v).__name__}")

    if len(v.shape) == 2:
        return np.r_[v, np.ones((1, v.shape[1]))]
    elif len(v.shape) == 1:
        return np.r_[v, 1.0]
    else:
        raise ValueError(
            f"Input must be 1D or 2D array, got {len(v.shape)}D array with shape {v.shape}"
        )


def draw_hom_line(line):
    """
    Draw one or more homogeneous lines in the current matplotlib plot.

    A homogeneous line ax + by + c = 0 is represented as [a, b, c].
    The function computes intersections with the current plot boundaries
    and draws the line segments within the visible region.

    Parameters:
    - line: ndarray of shape (3,) or (3, N) representing homogeneous line(s)

    Returns:
    - None (modifies current plot in-place)

    Example:
    >>> plt.figure()
    >>> plt.xlim(0, 10)
    >>> plt.ylim(0, 10)
    >>> draw_hom_line(np.array([1, 2, 3]))
    >>> plt.show()
    """
    if not isinstance(line, np.ndarray):
        raise TypeError(f"Line must be a numpy array, got {type(line).__name__}")

    # Validate shape
    if len(line.shape) == 1:
        if line.shape[0] != 3:
            raise ValueError(f"Line must have 3 elements, got shape {line.shape}")
    elif len(line.shape) == 2:
        if line.shape[0] != 3:
            raise ValueError(
                f"Line must have shape (3,) or (3, N), got shape {line.shape}"
            )
    else:
        raise ValueError(f"Line must be 1D or 2D array, got shape {line.shape}")

    # Get current plot limits
    ax = plt.gca()
    xlim = np.array(ax.get_xlim())
    ylim = np.array(ax.get_ylim())

    # Axis might be reversed, so we need to sort them
    xlim.sort()
    ylim.sort()

    if len(line.shape) < 2:
        line = line[:, np.newaxis]

    for i in range(line.shape[1]):
        # Compute the intersection points with the plot borders
        a, b, c = line[:, i]

        # Check for degenerate line (line at infinity)
        if np.abs(a) < 1e-10 and np.abs(b) < 1e-10:
            import warnings

            warnings.warn(
                f"Line {i} is degenerate (a={a:.2e}, b={b:.2e}). Skipping.",
                RuntimeWarning,
            )
            continue

        # Compute intersections, handling vertical and horizontal lines
        pts_list = []

        # Intersection with left/right borders (avoid division by zero for horizontal lines)
        if np.abs(b) > 1e-10:
            y1 = (-c - a * xlim[0]) / b
            y2 = (-c - a * xlim[1]) / b
            pts_list.extend([[xlim[0], y1], [xlim[1], y2]])

        # Intersection with top/bottom borders (avoid division by zero for vertical lines)
        if np.abs(a) > 1e-10:
            x1 = (-c - b * ylim[0]) / a
            x2 = (-c - b * ylim[1]) / a
            pts_list.extend([[x1, ylim[0]], [x2, ylim[1]]])

        if len(pts_list) == 0:
            continue

        pts = np.array(pts_list).T

        # Figure out which are inside the plot (add small epsilon to the limits)
        eps = 0.001 * max(np.abs(pts.flatten()).max(), 1.0)
        mask = (
            (pts[0] >= xlim[0] - eps)
            & (pts[0] <= xlim[1] + eps)
            & (pts[1] >= ylim[0] - eps)
            & (pts[1] <= ylim[1] + eps)
        )

        pts = pts[:, mask]

        if pts.shape[1] < 2:
            import warnings

            warnings.warn(
                f"Line {i} does not intersect the visible plot region. Skipping.",
                RuntimeWarning,
            )
            continue

        plt.plot(pts[0], pts[1], "r")


def set_axes_equal(
    ax, robust=False, q_low=0.02, q_high=0.98, min_range=None, must_include=None
):
    """
    Set equal scaling (aspect ratio 1:1:1) for a 3D matplotlib axis.

    This function ensures that the x, y, and z axes have the same scale,
    preventing distortion in 3D plots. It collects data from all plotted
    elements and computes bounds that maintain equal scaling.

    Parameters:
    - ax: matplotlib 3D axis object
    - robust: if True, use quantile-based bounds to ignore outliers (default: False)
    - q_low: lower quantile for robust bounds (default: 0.02)
    - q_high: upper quantile for robust bounds (default: 0.98)
    - min_range: minimum range for each axis (default: None)
    - must_include: array of shape (3, N) or (N, 3) with points that must be visible
                    in the plot, regardless of other settings (default: None)

    Returns:
    - None (modifies the axis limits in-place)

    Example:
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> ax.scatter(X[0], X[1], X[2])
    >>> set_axes_equal(ax)
    """
    # Validate that ax is a 3D axis
    if not hasattr(ax, "get_zlim"):
        raise TypeError(
            "Axis must be a 3D axis (created with projection='3d'). "
            "Got a 2D axis instead."
        )

    def _collect_xyz(ax):
        xs, ys, zs = [], [], []

        for line in ax.lines:
            x, y, z = line.get_data_3d()
            xs.append(np.asarray(x))
            ys.append(np.asarray(y))
            zs.append(np.asarray(z))

        for col in ax.collections:
            if hasattr(col, "_offsets3d"):
                x, y, z = col._offsets3d
                xs.append(np.asarray(x))
                ys.append(np.asarray(y))
                zs.append(np.asarray(z))

        if not xs:
            return None, None, None

        return (
            np.concatenate(xs),
            np.concatenate(ys),
            np.concatenate(zs),
        )

    # Collect plotted data
    x, y, z = _collect_xyz(ax)

    # Compute mandatory points (hard bounds)
    if must_include is not None:
        pts = np.asarray(must_include)

        if pts.ndim != 2 or 3 not in pts.shape:
            raise ValueError(
                f"must_include must have shape (3, N) or (N, 3), got shape {pts.shape}"
            )

        if pts.shape[0] == 3:
            xm, ym, zm = pts
        else:
            xm, ym, zm = pts.T

        hard_x0, hard_x1 = np.min(xm), np.max(xm)
        hard_y0, hard_y1 = np.min(ym), np.max(ym)
        hard_z0, hard_z1 = np.min(zm), np.max(zm)
    else:
        hard_x0 = hard_x1 = None
        hard_y0 = hard_y1 = None
        hard_z0 = hard_z1 = None

    if x is None and must_include is None:
        raise RuntimeError(
            "No 3D data found on axis and no must_include points provided. "
            "Plot some data first or provide must_include points."
        )

    # Compute robust or non-robust bounds from plotted data
    if robust and x is not None and len(x) > 0:
        x0, x1 = np.quantile(x, [q_low, q_high])
        y0, y1 = np.quantile(y, [q_low, q_high])
        z0, z1 = np.quantile(z, [q_low, q_high])
    else:
        x0, x1 = ax.get_xlim3d()
        y0, y1 = ax.get_ylim3d()
        z0, z1 = ax.get_zlim3d()

    # Enforce mandatory visibility (union of bounds)
    if must_include is not None:
        x0 = min(x0, hard_x0)
        x1 = max(x1, hard_x1)
        y0 = min(y0, hard_y0)
        y1 = max(y1, hard_y1)
        z0 = min(z0, hard_z0)
        z1 = max(z1, hard_z1)

    # Equalize ranges
    x_range = x1 - x0
    y_range = y1 - y0
    z_range = z1 - z0

    max_range = max(x_range, y_range, z_range)

    if min_range is not None:
        max_range = max(max_range, min_range)

    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)
    z_mid = 0.5 * (z0 + z1)

    ax.set_xlim3d(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim3d(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim3d(z_mid - max_range / 2, z_mid + max_range / 2)

    ax.set_box_aspect((1, 1, 1))


def homography_warp_image(im, H, bounds=None):
    """
    Warp an image using a homography matrix.

    Parameters:
    - im: Input image of shape (height, width) or (height, width, channels)
    - H: 3x3 homography matrix mapping from source to destination coordinates
    - bounds: Optional tuple (min_x, max_x, min_y, max_y) specifying output bounds.
              If None, bounds are computed from the warped image corners.

    Returns:
    - im_trans: Warped image (same dtype as input)
    - bounds: Tuple (min_x, max_x, min_y, max_y) of the output bounds

    Example:
    >>> im_warped, bounds = homography_warp_image(im, H)  # Note: returns tuple!
    >>> plt.imshow(im_warped, cmap='gray')
    """
    # Validate inputs
    if not isinstance(im, np.ndarray):
        raise TypeError(f"Image must be a numpy array, got {type(im).__name__}")

    if not isinstance(H, np.ndarray):
        raise TypeError(f"Homography must be a numpy array, got {type(H).__name__}")

    if H.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3 matrix, got shape {H.shape}")

    if im.ndim not in [2, 3]:
        raise ValueError(
            f"Image must be 2D (grayscale) or 3D (color), got {im.ndim}D array"
        )

    if im.ndim == 3 and im.shape[2] != 3:
        raise ValueError(
            f"Color images must have 3 channels, got {im.shape[2]} channels"
        )

    if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        raise ValueError("Homography contains NaN or Inf values")

    # Check if H is invertible
    det_H = np.linalg.det(H)
    if np.abs(det_H) < 1e-10:
        raise ValueError(
            f"Homography matrix is singular or near-singular (det={det_H:.2e}). "
            "Cannot invert homography for warping."
        )

    # We need the inverse mapping (from the transformed image to the original)
    H_inv = np.linalg.inv(H)

    if bounds is None:
        # Figure out bounds by warping corners of the input image
        corners = np.array(
            [
                [0, im.shape[1], im.shape[1], 0],
                [0, 0, im.shape[0], im.shape[0]],
                [1, 1, 1, 1],
            ]
        )
        warped_corners = H @ corners
        warped_corners /= warped_corners[2, :]

        # Compute bounding box that includes both:
        # 1. The warped corners
        # 2. The original image bounds (0, width) x (0, height)
        # This ensures the output canvas can fit both the warped image and the original coordinates
        all_x = np.concatenate([warped_corners[0, :], [0, im.shape[1]]])
        all_y = np.concatenate([warped_corners[1, :], [0, im.shape[0]]])

        min_x = np.floor(all_x.min()).astype(int)
        max_x = np.ceil(all_x.max()).astype(int)
        min_y = np.floor(all_y.min()).astype(int)
        max_y = np.ceil(all_y.max()).astype(int)

        bounds = (min_x, max_x, min_y, max_y)

    min_x, max_x, min_y, max_y = bounds

    # Compute output shape from bounds
    target_shape = (max_y - min_y, max_x - min_x)

    # Sanity check: prevent memory explosion from bad homographies
    input_area = im.shape[0] * im.shape[1]
    output_area = target_shape[0] * target_shape[1]
    if output_area > 4 * input_area:
        raise ValueError(
            f"Output image size ({target_shape[1]}x{target_shape[0]}) would be "
            f"{output_area / input_area:.1f}x larger than input "
            f"({im.shape[1]}x{im.shape[0]}). This likely indicates a bad homography. "
            f"Aborting to prevent memory explosion."
        )

    # Create translation matrix that shifts output coordinates to world coordinates
    # Output pixel (0,0) corresponds to world coordinate (min_x, min_y)
    T = np.array([[1, 0, min_x], [0, 1, min_y], [0, 0, 1]])
    # Compose: first translate output to world, then apply inverse homography
    H_inv_composed = H_inv @ T

    # Mapping function for scipy.ndimage.geometric_transform
    # Note: scipy uses (row, col) order, so we swap to (x, y) for matrix multiplication
    def warp(x):
        xh = np.array([x[1], x[0], 1.0])
        return (
            H_inv_composed[1, :] @ xh / (H_inv_composed[2, :] @ xh),
            H_inv_composed[0, :] @ xh / (H_inv_composed[2, :] @ xh),
        )

    if im.ndim == 2:
        im_trans = scipy.ndimage.geometric_transform(
            im, warp, (target_shape[0], target_shape[1])
        )
    elif im.ndim == 3:
        # Warp each color channel separately
        im_trans = np.concatenate(
            (
                scipy.ndimage.geometric_transform(
                    im[:, :, 0], warp, (target_shape[0], target_shape[1])
                )[:, :, None],
                scipy.ndimage.geometric_transform(
                    im[:, :, 1], warp, (target_shape[0], target_shape[1])
                )[:, :, None],
                scipy.ndimage.geometric_transform(
                    im[:, :, 2], warp, (target_shape[0], target_shape[1])
                )[:, :, None],
            ),
            axis=2,
        )

    # Check for issues in the output
    if np.any(np.isnan(im_trans)):
        import warnings

        warnings.warn(
            "Warped image contains NaN values. This may indicate issues with the homography "
            "or points being warped to infinity.",
            RuntimeWarning,
        )

    if np.iscomplexobj(im_trans):
        raise ValueError(
            "Warped image contains complex values. This should not happen and indicates "
            "a serious issue with the homography or coordinate transformations."
        )

    return im_trans, bounds


def plotcam(P, ax=None):
    """
    Plot a camera as a 3D arrow showing its optical axis direction.

    The camera center is computed from the projection matrix P,
    and an arrow is drawn pointing in the direction of the camera's
    optical axis (third row of P).

    Parameters:
    - P: 3x4 camera projection matrix
    - ax: matplotlib 3D axis (if None, uses current axis)

    Returns:
    - None (modifies the 3D plot in-place)
    """
    # Validate inputs
    if not isinstance(P, np.ndarray):
        raise TypeError(f"Camera matrix must be a numpy array, got {type(P).__name__}")

    if P.shape != (3, 4):
        raise ValueError(f"Camera matrix must be 3x4, got shape {P.shape}")

    if ax is None:
        ax = plt.gca()

    # Validate that ax is a 3D axis
    if not hasattr(ax, "get_zlim"):
        raise TypeError(
            "Axis must be a 3D axis (created with projection='3d'). "
            "Use fig.add_subplot(111, projection='3d')."
        )

    # Check if M = P[:,:3] is invertible
    M = P[:, :3]
    det_M = np.linalg.det(M)
    if np.abs(det_M) < 1e-10:
        raise ValueError(
            f"Camera matrix P[:,:3] is singular or near-singular (det={det_M:.2e}). "
            "Cannot compute camera center. Check that your camera matrix is valid."
        )

    c = -np.linalg.inv(M) @ P[:, 3]
    v = P[2, :]
    ax.quiver(c[0], c[1], c[2], v[0], v[1], v[2], color="r", normalize=True)


def plotcam_frustum(P, depth=1.0, ax=None, color="r"):
    """
    Plot a camera frustum (viewing pyramid) in 3D.

    Visualizes a camera as a 3D frustum showing its field of view.
    The frustum is constructed by backprojecting the image corners
    to a specified depth and connecting them to the camera center.

    Parameters:
    - P: 3x4 camera projection matrix P = K[R|t]
    - depth: distance along the optical axis at which to draw the image plane (default: 1.0)
    - ax: matplotlib 3D axis (if None, uses current axis)
    - color: color for the frustum lines and optical axis (default: "r")

    Returns:
    - None (modifies the 3D plot in-place)

    Note: The function infers the image size from the principal point,
    assuming width = 2*cx and height = 2*cy. This is a common convention
    when the principal point is at the image center.

    Example:
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> plotcam_frustum(P, depth=2.0, ax=ax, color='blue')
    """
    # Validate inputs
    if not isinstance(P, np.ndarray):
        raise TypeError(f"Camera matrix must be a numpy array, got {type(P).__name__}")

    if P.shape != (3, 4):
        raise ValueError(f"Camera matrix must be 3x4, got shape {P.shape}")

    if depth <= 0:
        raise ValueError(f"Depth must be positive, got {depth}")

    if ax is None:
        ax = plt.gca()

    # Validate that ax is a 3D axis
    if not hasattr(ax, "get_zlim"):
        raise TypeError(
            "Axis must be a 3D axis (created with projection='3d'). "
            "Use fig.add_subplot(111, projection='3d')."
        )

    # Decompose P
    M = P[:, :3]
    p4 = P[:, 3]

    # Check if M is invertible
    det_M = np.linalg.det(M)
    if np.abs(det_M) < 1e-10:
        raise ValueError(
            f"Camera matrix P[:,:3] is singular or near-singular (det={det_M:.2e}). "
            "Cannot decompose camera matrix. Check that your camera matrix is valid."
        )

    # Ensure det(M) > 0 so RQ gives proper rotation (P and -P are equivalent)
    if det_M < 0:
        M = -M
        p4 = -p4

    K, R = rq(M)

    # Normalize K so K[2,2] = 1
    K = K / K[2, 2]

    # Camera center
    C = -np.linalg.inv(M) @ p4

    # Check if K is identity (calibrated camera with normalized coordinates)
    # K might be a scaled identity, so normalize it with K[2,2] for comparison
    K_normalized = K / K[2, 2] if K[2, 2] != 0 else K
    is_calibrated = np.allclose(K_normalized, np.eye(3), atol=1e-6)

    if is_calibrated:
        # For calibrated cameras (K == I), assume 90 degree FoV
        # With 90 degree FoV, the half-angle is 45 degrees
        # tan(45°) = 1, so the image corners are at ±1 in normalized coordinates
        corners_px = np.array([
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]
        ]).T  # (3,4)

        # For calibrated camera, K = I, so rays_cam = corners_px
        rays_cam = corners_px
    else:
        # Infer image size from principal point
        cx = K[0, 2]
        cy = K[1, 2]
        w = 2.0 * cx
        h = 2.0 * cy

        # Image corners in pixel coordinates
        corners_px = np.array([
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1]
        ]).T  # (3,4)

        # Backproject rays
        Kinv = np.linalg.inv(K)
        rays_cam = Kinv @ corners_px

    rays_world = R.T @ rays_cam

    # Normalize rays
    rays_world /= np.linalg.norm(rays_world, axis=0, keepdims=True)

    # Frustum corners at chosen depth
    corners_3d = C[:, None] + depth * rays_world

    # Optical axis
    z_axis = R.T @ np.array([0, 0, 1])
    ax.quiver(
        C[0], C[1], C[2],
        z_axis[0], z_axis[1], z_axis[2],
        length=0.8 * depth,
        color=color
    )

    # Frustum edges
    for i in range(4):
        X = corners_3d[:, i]
        ax.plot(
            [C[0], X[0]],
            [C[1], X[1]],
            [C[2], X[2]],
            color=color
        )

    # Image plane rectangle
    order = [0, 1, 2, 3, 0]
    ax.plot(
        corners_3d[0, order],
        corners_3d[1, order],
        corners_3d[2, order],
        color=color
    )


def rq(P):
    """
    RQ decomposition of a matrix.

    Decomposes a matrix P into an upper triangular matrix R and an
    orthogonal matrix Q such that P = R @ Q. This is useful for
    decomposing camera projection matrices P = K[R|t] into intrinsic
    matrix K and extrinsic parameters [R|t].

    Parameters:
    - P: ndarray of shape (m, n) to decompose

    Returns:
    - R: upper triangular matrix of shape (m, m)
    - Q: orthogonal matrix (if P is square) or extended matrix (if P has more columns)
    """
    if not isinstance(P, np.ndarray):
        raise TypeError(f"Input must be a numpy array, got {type(P).__name__}")

    if P.ndim != 2:
        raise ValueError(f"Input must be a 2D matrix, got {P.ndim}D array")

    if P.shape[0] > P.shape[1]:
        raise ValueError(
            f"Input must have at least as many columns as rows, got shape {P.shape}"
        )

    e = np.eye(P.shape[0])
    p = e[:, ::-1]
    q, r = np.linalg.qr(p @ P[:, : P.shape[0]].T @ p)
    r = p @ r.T @ p
    q = p @ q.T @ p
    fix = np.diag(np.sign(np.diag(r)))
    r = r @ fix
    q = fix @ q
    if np.shape(P)[1] > np.shape(P)[0]:
        q = np.c_[q, np.linalg.inv(r) @ P[:, np.shape(P)[0] :]]
    return r, q


def fivepoint_solver(x1n, x2n):
    """
    Estimate the essential matrix from 5 point correspondences.

    Implements the 5-point algorithm for essential matrix estimation
    using normalized image coordinates. The algorithm solves the epipolar
    constraint along with the essential matrix constraints (rank 2 and
    det(E) = 0, plus the singular value constraint).

    Parameters:
    - x1n: 3x5 array of normalized homogeneous point coordinates in image 1
    - x2n: 3x5 array of normalized homogeneous point coordinates in image 2

    Returns:
    - list of essential matrices (typically up to 10 solutions)

    Note: Input points should be normalized (e.g., using K^{-1} @ x) before
    calling this function. The algorithm returns all mathematically valid
    solutions; additional constraints (e.g., cheirality) are needed to
    select the correct one.
    """
    # Validate inputs
    if not isinstance(x1n, np.ndarray) or not isinstance(x2n, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")

    if x1n.shape != (3, 5):
        raise ValueError(
            f"x1n must be a 3x5 array of homogeneous coordinates, got shape {x1n.shape}"
        )

    if x2n.shape != (3, 5):
        raise ValueError(
            f"x2n must be a 3x5 array of homogeneous coordinates, got shape {x2n.shape}"
        )

    if np.any(np.isnan(x1n)) or np.any(np.isnan(x2n)):
        raise ValueError("Input points contain NaN values")

    if np.any(np.isinf(x1n)) or np.any(np.isinf(x2n)):
        raise ValueError("Input points contain Inf values")

    # Constructing the M matrix
    M = np.zeros((5, 9))
    for i in range(5):
        xx = np.outer(x1n[:, i], x2n[:, i])
        M[i, :] = xx.flatten()

    # Compute the null space of M
    u, s, vt = np.linalg.svd(M)
    Evec = vt[5:10, :].T
    E = [Evec[:, i].reshape(3, 3).T for i in range(4)]

    # Compute constraint coefficients
    coeffs = np.zeros((9, 64))
    mons = np.zeros((4, 64))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                idx = k + j * 4 + i * 16
                mons[i, idx] += 1
                mons[j, idx] += 1
                mons[k, idx] += 1
                new_coeffs = 2 * E[i] @ E[j].T @ E[k] - np.trace(E[i] @ E[j].T) * E[k]
                coeffs[:, idx] = new_coeffs.flatten()

    # Compute determinant constraint
    det_coeffs = np.zeros(64)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                idx = k + j * 4 + i * 16
                det_coeffs[idx] = (
                    E[i][0, 0] * E[j][1, 1] * E[k][2, 2]
                    + E[i][0, 1] * E[j][1, 2] * E[k][2, 0]
                    + E[i][0, 2] * E[j][1, 0] * E[k][2, 1]
                    - E[i][0, 0] * E[j][1, 2] * E[k][2, 1]
                    - E[i][0, 1] * E[j][1, 0] * E[k][2, 2]
                    - E[i][0, 2] * E[j][1, 1] * E[k][2, 0]
                )

    coeffs = np.vstack([coeffs, det_coeffs])

    # Make monomials unique
    mons, unique_indices, J = np.unique(
        mons.T, axis=0, return_index=True, return_inverse=True
    )
    mons = mons.T
    coeffs_small = np.zeros((10, len(unique_indices)))
    for i in range(coeffs.shape[0]):
        coeffs_small[i, :] = np.bincount(
            J, weights=coeffs[i, :], minlength=len(unique_indices)
        )

    # Set x_1 = 1 and perform RREF
    mons = mons[1:4, :]

    reduced_coeffs = np.linalg.inv(coeffs_small[:, :10]) @ coeffs_small

    # Construct action matrix multiplication with x_4
    Mx4 = np.zeros((10, 10))
    mon_basis = mons[:, 10:]
    for i in range(mon_basis.shape[1]):
        x4mon = mon_basis[:, i] + np.array([0, 0, 1])
        if np.sum(x4mon) >= 3:
            row = np.where(np.all(mons[:, :10] == x4mon[:, None], axis=0))[0]
            Mx4[:, i] = -reduced_coeffs[row, 10:].flatten()
        else:
            ind = np.where(np.all(mon_basis == x4mon[:, None], axis=0))[0]
            Mx4[ind, i] = 1

    # Extract real solutions

    eigvals, eigvecs = np.linalg.eig(Mx4.T)
    eigvecs /= eigvecs[-1, :]

    Esol = []
    for i in range(10):
        if np.isreal(eigvals[i]):
            x2, x3, x4 = eigvecs[-2, i], eigvecs[-3, i], eigvecs[-4, i]

            Esol.append(
                E[0] + E[1] * np.real(x2) + E[2] * np.real(x3) + E[3] * np.real(x4)
            )

    return Esol


def compute_reprojection_rms_error(P, x, X):
    """
    Compute the Root-Mean-Squared (RMS) reprojection error for a multi-view reconstruction.

    Parameters:
    - P: list of camera projection matrices (each 3x4)
    - x: list of 2xN arrays of image point coordinates (missing points marked as NaN)
    - X: 3xN or 4xN array of 3D point coordinates (Euclidean or homogeneous)

    Returns:
    - error: total squared reprojection error across all visible points

    Note: Points that are not visible in a particular view should have NaN
    coordinates in the corresponding x array.
    """
    # Validate inputs
    if not isinstance(P, list):
        raise TypeError(f"P must be a list of camera matrices, got {type(P).__name__}")

    if not isinstance(x, list):
        raise TypeError(f"x must be a list of point arrays, got {type(x).__name__}")

    if len(P) != len(x):
        raise ValueError(
            f"Number of cameras ({len(P)}) must match number of point arrays ({len(x)})"
        )

    if not isinstance(X, np.ndarray):
        raise TypeError(f"X must be a numpy array, got {type(X).__name__}")

    if X.ndim != 2 or X.shape[0] not in [3, 4]:
        raise ValueError(f"X must have shape (3, N) or (4, N), got shape {X.shape}")

    # Validate each camera matrix and point array
    for i, (Pi, xi) in enumerate(zip(P, x)):
        if not isinstance(Pi, np.ndarray):
            raise TypeError(
                f"Camera {i} must be a numpy array, got {type(Pi).__name__}"
            )
        if Pi.shape != (3, 4):
            raise ValueError(f"Camera {i} must be 3x4, got shape {Pi.shape}")

        if not isinstance(xi, np.ndarray):
            raise TypeError(f"Point array {i} must be a numpy array, got {type(xi).__name__}")

        if xi.shape[0] != 2:
            raise ValueError(
                f"Point array {i} must have shape (2, N), got shape {xi.shape}"
            )

        if xi.shape[1] != X.shape[1]:
            raise ValueError(
                f"Point array {i} has {xi.shape[1]} points but X has {X.shape[1]} points"
            )

    if X.shape[0] == 3:
        X = np.vstack([X, np.ones(X.shape[1])])

    error = 0
    num_points = X.shape[1]
    num_res = 0
    for i in range(len(P)):
        vis = ~np.isnan(x[i][0])

        PX = P[i] @ X
        for k in range(num_points):
            if ~vis[k]:
                continue

            proj = PX[:2, k] / PX[2, k]
            error += np.sum((proj - x[i][:, k]) ** 2)
            num_res += 1

    if num_res == 0:
        raise ValueError(
            "No visible points found. All points are marked as NaN. "
            "Cannot compute reprojection error."
        )

    return np.sqrt(error / num_res)


def linearize_reprojection_error(P, x, X):
    """
    Compute the reprojection error and its Jacobian for bundle adjustment.

    This function computes the reprojection error and the Jacobian matrix
    needed for iterative optimization (e.g., Levenberg-Marquardt). The
    parameterization uses a 6-parameter representation for each camera
    (3 for translation, 3 for rotation via exponential map) and 3 parameters
    per 3D point.

    Parameters:
    - P: list of camera projection matrices (each 3x4)
    - x: list of 2xN arrays of image point coordinates (missing points marked as NaN)
    - X: 3xN or 4xN array of 3D point coordinates (Euclidean or homogeneous)

    Returns:
    - r: residual vector of length (2 * num_visible_points)
    - J: Jacobian matrix of shape (2 * num_visible_points, num_params)

    Note: The coordinate system is fixed by dropping the first 7 parameters
    (all parameters of P[0] and the first translation parameter of P[1]).
    """
    # Validate inputs (similar to compute_reprojection_rms_error)
    if not isinstance(P, list):
        raise TypeError(f"P must be a list of camera matrices, got {type(P).__name__}")

    if not isinstance(x, list):
        raise TypeError(f"x must be a list of point arrays, got {type(x).__name__}")

    if len(P) != len(x):
        raise ValueError(
            f"Number of cameras ({len(P)}) must match number of point arrays ({len(x)})"
        )

    if not isinstance(X, np.ndarray):
        raise TypeError(f"X must be a numpy array, got {type(X).__name__}")

    if X.ndim != 2 or X.shape[0] not in [3, 4]:
        raise ValueError(f"X must have shape (3, N) or (4, N), got shape {X.shape}")

    # Validate each camera matrix and point array
    for i, (Pi, xi) in enumerate(zip(P, x)):
        if not isinstance(Pi, np.ndarray):
            raise TypeError(
                f"Camera {i} must be a numpy array, got {type(Pi).__name__}"
            )
        if Pi.shape != (3, 4):
            raise ValueError(f"Camera {i} must be 3x4, got shape {Pi.shape}")

        if not isinstance(xi, np.ndarray):
            raise TypeError(f"Point array {i} must be a numpy array, got {type(xi).__name__}")

        if xi.shape[0] != 2:
            raise ValueError(
                f"Point array {i} must have shape (2, N), got shape {xi.shape}"
            )

        if xi.shape[1] != X.shape[1]:
            raise ValueError(
                f"Point array {i} has {xi.shape[1]} points but X has {X.shape[1]} points"
            )

    if X.shape[0] == 3:
        X = np.vstack([X, np.ones(X.shape[1])])

    num_params = 6 * len(P) + 3 * X.shape[1]
    num_points = X.shape[1]

    num_vis = [np.sum(~np.isnan(xc[0])) for xc in x]
    num_vis_total = np.sum(num_vis)

    if num_vis_total == 0:
        raise ValueError(
            "No visible points found. All points are marked as NaN. "
            "Cannot compute Jacobian."
        )

    J = np.zeros((2 * num_vis_total, num_params))
    r = np.zeros(2 * num_vis_total)

    # Index where the 3D points start
    idx_pts = len(P) * 6

    dR1 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    dR2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    dR3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

    idx = 0
    for i in range(len(P)):
        vis = ~np.isnan(x[i][0])

        PX = P[i] @ X
        for k in range(num_points):
            if ~vis[k]:
                continue

            proj = PX[:2, k] / PX[2, k]
            r[2 * idx : 2 * (idx + 1)] = proj - x[i][:, k]

            Jproj = np.array(
                [
                    [1 / PX[2, k], 0, -PX[0, k] / PX[2, k] ** 2],
                    [0, 1 / PX[2, k], -PX[1, k] / PX[2, k] ** 2],
                ]
            )

            # R * exp([w]_x) * X
            # Differentiate w.r.t. w1, w2, w3
            dPX_dw = P[i][:, :3] @ np.c_[dR1 @ X[:3, k], dR2 @ X[:3, k], dR3 @ X[:3, k]]

            # Jacobian w.r.t. translation and rotation parameters
            J[2 * idx : 2 * (idx + 1), 6 * i : 6 * i + 3] = Jproj
            J[2 * idx : 2 * (idx + 1), 6 * i + 3 : 6 * i + 6] = Jproj @ dPX_dw

            # Jacobian w.r.t. 3D point
            J[2 * idx : 2 * (idx + 1), idx_pts + 3 * k : idx_pts + 3 * (k + 1)] = (
                Jproj @ P[i][:, :3]
            )

            idx += 1

    # Fix coordinate system (drop P1 and first translation parameter of P2)
    J = J[:, 7:]

    return r, J


def update_solution(P, X, delta):
    """
    Update camera matrices and 3D points using a parameter update vector.

    This function applies the parameter updates computed by bundle adjustment
    to the current estimates. Camera rotations are updated using the exponential
    map parameterization.

    Parameters:
    - P: list of camera projection matrices (each 3x4)
    - X: 3xN or 4xN array of 3D point coordinates
    - delta: update vector (length should be 6*(num_cameras-1) + 3*num_points - 1)

    Returns:
    - P_new: list of updated camera projection matrices
    - X_new: updated 3xN array of 3D point coordinates

    Note: The first 7 parameters are fixed (prepended as zeros) to anchor the
    coordinate system, following the convention in linearize_reprojection_error.
    """
    # Validate inputs
    if not isinstance(P, list):
        raise TypeError(f"P must be a list of camera matrices, got {type(P).__name__}")

    if not isinstance(X, np.ndarray):
        raise TypeError(f"X must be a numpy array, got {type(X).__name__}")

    if not isinstance(delta, np.ndarray):
        raise TypeError(f"delta must be a numpy array, got {type(delta).__name__}")

    if X.ndim != 2 or X.shape[0] not in [3, 4]:
        raise ValueError(f"X must have shape (3, N) or (4, N), got shape {X.shape}")

    for i, Pi in enumerate(P):
        if not isinstance(Pi, np.ndarray):
            raise TypeError(
                f"Camera {i} must be a numpy array, got {type(Pi).__name__}"
            )
        if Pi.shape != (3, 4):
            raise ValueError(f"Camera {i} must be 3x4, got shape {Pi.shape}")

    num_cameras = len(P)
    num_points = X.shape[1]

    expected_delta_size = 6 * num_cameras + 3 * num_points - 7
    if delta.size != expected_delta_size:
        raise ValueError(
            f"delta has wrong size. Expected {expected_delta_size} "
            f"(6*{num_cameras} cameras + 3*{num_points} points - 7 fixed), "
            f"got {delta.size}"
        )

    P = [P[i].copy() for i in range(len(P))]
    X = X.copy()

    # Convert to Euclidean if homogeneous
    if X.shape[0] == 4:
        X = X[:3, :]

    delta = delta.flatten()
    delta = np.r_[np.zeros(7), delta]

    for i in range(num_cameras):
        dw = delta[6 * i + 3 : 6 * i + 6]
        dR = scipy.linalg.expm(
            np.array([[0, -dw[2], dw[1]], [dw[2], 0, -dw[0]], [-dw[1], dw[0], 0]])
        )
        P[i][:, :3] = P[i][:, :3] @ dR
        P[i][:, 3] += delta[6 * i : 6 * i + 3]

    X[:3, :] += delta[6 * num_cameras :].reshape(num_points, 3).T

    return P, X
