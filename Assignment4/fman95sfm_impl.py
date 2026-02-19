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

def ransac_essential_matrix(x1, x2, K):
    # Input:
    # x1, x2: 2xN corresponding points in image 1 and 2
    # K: 3x3 intrinsic calibration matrix

    # TODO perform RANSAC to estimate the essential matrix

    # Placeholders
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

    # Output: two 3x4 camera projection matrices
    # that include K, i.e., P = K [R|t]
    return P1, P2


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

    # Placeholder
    X = np.zeros((3, x1.shape[1]))

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