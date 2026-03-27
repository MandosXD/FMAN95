import numpy as np
import utils

def select_initial_pair(model):
    # Select the initial image pair used to bootstrap the reconstruction.

    # Naive implementation: just return the first two images.
    return (0, 1)


def select_next_image(model):
    candidates = [k for k, cam in enumerate(model.cameras) if cam is None]
    reg_images = [k for k, cam in enumerate(model.cameras) if cam is not None]

    if len(candidates) == 0:
        return None

    # Select the next image to register.
    # Currently we only return the first one in the list, but perhaps something
    # smarter could be done?

    return candidates[0]


def to_homogeneous(X):
    return np.vstack((X, np.ones((1, X.shape[1]))))

def depth(P, X):
    return (np.sign(np.linalg.det(P[:, :3])) *
            (P @ X)[2] /
            (np.linalg.norm(P[2, :3]) * X[3])) # Formula from the lecture notes

def pflat(X):
    X = np.asarray(X)
    return X / X[-1] # Divide by the last row

def ransac_essential_matrix(x1, x2, K):
    # Input:
    # x1, x2: 2xN corresponding points in image 1 and 2
    # K: 3x3 intrinsic calibration matrix

    # TODO perform RANSAC to estimate the essential matrix

    x1_h = to_homogeneous(x1)
    x2_h = to_homogeneous(x2)
    
    # Normalize points
    K_inv = np.linalg.inv(K)
    x1n = K_inv @ x1_h
    x2n = K_inv @ x2_h


    threshold = 5.0
    best_inliers = None
    max_inliers = 0

    for _ in range(k):
        # Randomly choose 5 correspondences
        indices = np.random.choice(x1n.shape[1], size=5, replace=False)
        x1n_sample = x1n[:, indices]
        x2n_sample = x2n[:, indices]

        # Fit an essential matrix to the 5 correspondences
        E_list = utils.fivepoint_solver(x1n_sample, x2n_sample)

        # Test all candidate solutions
        for E in E_list:
                # Computing the unnormalized fundamental matrix F
                F = K_inv.T @ E @ K_inv

                # Epipolar lines
                l2 = F @ x1_h              # lines in image 2
                l1 = F.T @ x2_h            # lines in image 1

                # Normalize each line so first two components form unit normal
                l2 = l2 / np.sqrt(l2[0]**2 + l2[1]**2)
                l1 = l1 / np.sqrt(l1[0]**2 + l1[1]**2)

                # Compute point to line distances
                d2 = np.abs(np.sum(l2 * x2_h, axis=0))
                d1 = np.abs(np.sum(l1 * x1_h, axis=0))

                # Inlier condition distance must be small in BOTH images
                inliers = (d1 < threshold) & (d2 < threshold)
                num_inliers = np.sum(inliers)

                if num_inliers > max_inliers:
                    max_inliers = num_inliers
                    best_inliers = inliers
                    best_E = E
    
    x1n_inlier = x1n[:, best_inliers]
    x2n_inlier = x2n[:, best_inliers]

    # Note that this returns the V matrix already transposed!
    u, s, vt = np.linalg.svd(best_E)
    if np.linalg.det(u @ vt) < 0:
        vt = -vt
    E = u @ np.diag([1, 1, 0]) @ vt
    # Note: Re-computing svd on E may still give U and V that does not fulfill
    # det(U*V') = 1 since the svd is not unique.
    # So don't recompute the svd after this step.

    # Define w matrix
    w = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    # Construct the camera matricies from E
    P2a = np.hstack((u @ w @ vt, u[:, 2:3]))
    P2b = np.hstack((u @ w @ vt, -u[:, 2:3]))
    P2c = np.hstack((u @ w.T @ vt, u[:, 2:3]))
    P2d = np.hstack((u @ w.T @ vt, -u[:, 2:3]))
    # First camera
    P1 = np.hstack((np.eye(3), np.zeros((3,1))))

    candidates = [
        ("P2a", P2a),
        ("P2b", P2b),
        ("P2c", P2c),
        ("P2d", P2d),
    ]

    best_count = -1
    best_P2 = None
    best_X = None
    best_name = None

    for name, P2 in candidates:
        X = np.zeros((4, x1n_inlier.shape[1]))

        for i in range(x1n_inlier.shape[1]):
            X[:, i] = triangulate_pair(P1, P2, x1n_inlier[:, i], x2n_inlier[:, i])
        
        # Check that the points are infront of both cameras
        Z1 = depth(P1, X)
        Z2 = depth(P2, X)

        count = np.sum((Z1 > 0) & (Z2 > 0))

        print(f"{name}: {count} points in front")

        if count > best_count:
            best_count = count
            best_P2 = P2
            best_X = pflat(X)
            best_name = name

    print(f"Selected camera is {best_name} with {best_count} points in front")

    # Convert cameras back to pixel coordinates
    P1_pixel = K @ P1
    P2_pixel = K @ best_P2

    # Output: two 3x4 camera projection matrices
    # that include K, i.e., P = K [R|t]
    return P1_pixel, P2_pixel


def ransac_camera_pose(x, X, K):
    # Input:
    # x: 2xN image points
    # X: 3xN corresponding 3D points
    # K: 3x3 intrinsic calibration matrix

    # TODO perform RANSAC to estimate the camera pose

    # Placeholder
    P = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

    # Output: 3x4 camera projection matrix (including K)
    return P


def triangulate_pair(P1, P2, x1, x2):
    # Input:
    # P1, P2: 3x4 camera projection matrices
    # x1, x2: 2xN corresponding image points in the two images

    # TODO triangulate the 3D points from the two views

    A = np.vstack([
        x1[0]*P1[2] - P1[0],
        x1[1]*P1[2] - P1[1],
        x2[0]*P2[2] - P2[0],
        x2[1]*P2[2] - P2[1]
    ])

    X = np.linalg.svd(A)[2][-1]

    # Output: 3xN array of 3D points
    return X


def compute_reprojection_errors(P, X, x):
    # Input:
    # P: 3x4 camera projection matrix
    # X: 3xN array of 3D points
    # x: 2xN array of corresponding image points

    err = 1e5 * np.ones(x.shape[1])  # Placeholder

    # Output: N array of reprojection errors
    return err


def bundle_adjustment(P, X, x):
    # Input:
    # P: List of 3x4 camera projection matrix
    # X: 3xN array of 3D points
    # x: List of 2xN arrays of corresponding image points
    #    Missing points are indicated with NaNs.
    # Same interface as in the previous exercises.

    # Output: optimized P and X
    return P, X