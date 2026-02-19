import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt

# Import functions that you are supposed to implement!
from fman95sfm_impl import (
    select_next_image,
    ransac_essential_matrix,
    triangulate_pair,
    compute_reprojection_errors,
    select_initial_pair,
    ransac_camera_pose,
    bundle_adjustment,
)

import utils

# Pixel threshold used for registration and initial triangulation
tol_px = 5.0
# Pixel threshold used for filtering 3D points after BA
tol_filter = 2.5 

def main():
    # Load the house dataset
    model = init_model()
    load_default_data(model)

    # and run Structure-from-Motion!
    run_sfm(model)


def run_sfm(model):
    # Select initial image pair to boostrap the reconstruction
    init_pair = select_initial_pair(model)

    # Estimate the relative pose using RANSAC
    two_view_init(model, init_pair)
    # Triangulate initial 3D points
    retriangulate_pair(model, init_pair)

    # Run bundle adjustment
    run_bundle_adjustment(model)

    # Try to triangulate new points with refined cameras
    retriangulate_pair(model, init_pair)

    # Print model summary and visualize
    print_model_summary(model)
    viz_model(model)

    # Now we loop through all images and try to register them one by one
    while True:
        # Select next image to register
        next_im = select_next_image(model)

        if next_im is None:
            print("No more images to register.")
            break

        # Try to register the new image
        success = register_new_image(model, next_im)

        if success:
            # Check which current tracks can be extended to the new image
            extend_tracks(model, next_im)

        # Refine the full model using bundle adjustment
        run_bundle_adjustment(model)

        # Filter bad 3D points after bundle adjustment
        filter_tracks(model)

        # Try to triangulate new points with the new image
        for im_id in registered_images(model):
            if im_id == next_im:
                continue
            retriangulate_pair(model, (im_id, next_im))


        # Print model summary and visualize
        print_model_summary(model)
        viz_model(model)

    # Run final bundle adjustment and filtering
    run_bundle_adjustment(model)
    filter_tracks(model)
    run_bundle_adjustment(model)

    # Final model visualization, blocking so that we can inspect it
    viz_model(model, block=True)


def init_model():
    # Sets up the data structures for the model
    model = SimpleNamespace()
    model.image_names = []              # List with image file names
    model.K = []                        # Camera intrinsics matrix
    model.cameras = []                  # List with M camera matrices (3x4 numpy arrays)
                                        # (including the K matrix, i.e. P = K [R|t])
    model.points2d = []                 # List with 2xNp arrays for 2D points.
    model.points2d_to_3d_id = []        # 3D point index for each 2D point. List with 1xNp arrays
                                        # (-1 if no 3D point is associated)
    model.points3d = np.zeros((3, 0))   # 3D points. 3xN numpy array
    model.tracks = []                   # List with N elements with point tracks
                                        # (each list of tuples with (im_id, kp_id))
    model.matches = []                  # MxM array with matches
                                        # (each element is a 2xNp array with indices)
    return model


def load_default_data(model):
    # Reads the house dataset
    data = np.load("data/A4_ex8_data.npz", allow_pickle=True)
    model.cameras = [None for _ in range(12)]
    model.K = data["K"]
    model.matches = data["matches"]
    model.points2d = data["x"]
    model.points2d_to_3d_id = [
        np.full(points.shape[1], -1) for points in model.points2d
    ]
    model.image_names = [f"data/house/house{idx:02d}.jpg" for idx in range(12)]


def register_new_image(model, new_im_id):
    # First, collect 2D-3D correspondences by looking at matches
    # between the new image and all registered images

    # We use a set here to avoid adding duplicates
    corrs = set()

    for im_id in registered_images(model):
        # Get matches between new image and registered image
        ind1 = model.matches[im_id, new_im_id][0, :]
        ind2 = model.matches[im_id, new_im_id][1, :]

        # Find cases where we have a match to a triangulated point
        candidates = np.array(model.points2d_to_3d_id[im_id][ind1] > -1)
        if not np.any(candidates):
            continue

        ind1 = ind1[candidates]
        ind2 = ind2[candidates]
        # Add tuples with index to 2D keypoint in the new image,
        # and index to the matching 3D point
        for ind2d, ind3d in zip(ind2, model.points2d_to_3d_id[im_id][ind1]):
            corrs.add((ind2d, ind3d))

    # Fetch 2D-3D correspondences
    pts2d = np.zeros((2, len(corrs)))
    pts3d = np.zeros((3, len(corrs)))
    for k, (ind2d, ind3d) in enumerate(corrs):
        pts2d[:, k] = model.points2d[new_im_id][:, ind2d]
        pts3d[:, k] = model.points3d[:, ind3d]

    # Need at least 6 correspondences to compute pose with DLT
    if len(corrs) < 6:
        print(
            f"Not enough 2D-3D correspondences to register image {new_im_id} ({len(corrs)} found)."
        )
        return False

    # Estimate camera pose using student-implemented RANSAC PnP
    P = ransac_camera_pose(pts2d, pts3d, model.K)

    if P is None:
        return False

    # Update model with new camera
    model.cameras[new_im_id] = P
    return True


def run_bundle_adjustment(model):
    # Collects correspondences and 3D points, reshuffles data such that it is
    # consistent with the format for the previous hand-in exercise.
    n = model.points3d.shape[1]
    m = len(model.cameras)
    x = [np.nan * np.ones((2, n)) for _ in range(m)]

    # Keep track of active cameras (i.e. ones with at least one observation)
    active_cameras = np.zeros(m)

    # Collect measurements
    for point3d_id, track in enumerate(model.tracks):
        for im_id, kp_id in track:
            x[im_id][:, point3d_id] = model.points2d[im_id][:, kp_id]
            active_cameras[im_id] = True

    # Filter out inactive cameras
    active_cameras = np.where(active_cameras)[0]
    P_active = [model.cameras[i] for i in active_cameras]
    x_active = [x[i] for i in active_cameras]

    print(
        f"Running bundle adjustment with {len(P_active)} cameras and {model.points3d.shape[1]} points."
    )

    # Call the student-implemented bundle adjustment function
    P, X = bundle_adjustment(P_active, model.points3d, x_active)

    # Update model with refined cameras and points
    for idx, cam_id in enumerate(active_cameras):
        model.cameras[cam_id] = P[idx]
    model.points3d = X

    pass


def initialize_new_tracks(model, image_pair, kp_ind1, kp_ind2, points3d):
    # Adds new tracks to the model given triangulated 3D points from an image pair
    (im1, im2) = image_pair

    # Loop through all potential new tracks
    for kp_i, kp_j, X in zip(kp_ind1, kp_ind2, points3d.T):
        # Skip whenever keypoint is already part of a track
        if (
            model.points2d_to_3d_id[im1][kp_i] != -1
            or model.points2d_to_3d_id[im2][kp_j] != -1
        ):
            continue

        # Add new 3D point and track
        model.points3d = np.hstack([model.points3d, X[:, None]])
        model.tracks.append([(im1, kp_i), (im2, kp_j)])  # two-view track
        point3d_id = model.points3d.shape[1] - 1
        model.points2d_to_3d_id[im1][kp_i] = point3d_id
        model.points2d_to_3d_id[im2][kp_j] = point3d_id


def extend_tracks(model, new_im_id):
    # Tries to extend current tracks into a new cameras
    # (i.e. find already triangulated points that are visible in the new image)
    for im_id in registered_images(model):
        if im_id == new_im_id:
            continue

        # Find matches to registered images
        ind1 = model.matches[im_id, new_im_id][0, :]
        ind2 = model.matches[im_id, new_im_id][1, :]

        # Find cases where we have a 3D point in the old image, and none in the new
        candidates = np.array(model.points2d_to_3d_id[im_id][ind1] > -1) & np.array(
            model.points2d_to_3d_id[new_im_id][ind2] == -1
        )

        if not np.any(candidates):
            continue

        ind1 = ind1[candidates]
        ind2 = ind2[candidates]

        # Get 3D points
        points3d_ids = model.points2d_to_3d_id[im_id][ind1]
        X = model.points3d[:, points3d_ids]

        # Get 2D points from new image
        x = model.points2d[new_im_id][:, ind2]

        # Compute reprojection error in new image
        err = compute_reprojection_errors(model.cameras[new_im_id], X, x)

        # Find inliers
        inl = err < tol_px

        if not np.any(inl):
            continue

        ind1 = ind1[inl]
        ind2 = ind2[inl]

        # Extend tracks to new image
        for kp_j, point3d_id in zip(ind2, model.points2d_to_3d_id[im_id][ind1]):
            model.tracks[point3d_id].append((new_im_id, kp_j))
            model.points2d_to_3d_id[new_im_id][kp_j] = point3d_id

    print(
        f"Extended {np.sum(model.points2d_to_3d_id[new_im_id] > -1)} tracks into image {new_im_id}, total 3D points: {model.points3d.shape[1]}"
    )

def filter_tracks(model):
    # Go through each image and filter out 2D-3D associations that are inconsistent
    
    filtered_observations = 0
    for im_id in registered_images(model):
        X_ids = model.points2d_to_3d_id[im_id]
        active = np.where(X_ids > -1)[0]
        X = model.points3d[:, X_ids[active]]
        x = model.points2d[im_id][:, active]
        err = compute_reprojection_errors(model.cameras[im_id], X, x)

        # We filter points that have high reprojection error or are behind the camera
        behind = model.cameras[im_id][2,:] @ utils.phom(X) < 0
        outlier = (err >= tol_filter) | behind
        to_remove = active[outlier]
        filtered_observations += len(to_remove)

        # Go through each observation to remove from the corresponding track
        for kp_id in to_remove:
            point3d_id = model.points2d_to_3d_id[im_id][kp_id]
            # Remove from track
            track = model.tracks[point3d_id]
            track = [t for t in track if t[0] != im_id]
            model.tracks[point3d_id] = track
            # Remove 2D-3D association
            model.points2d_to_3d_id[im_id][kp_id] = -1

    # Now some tracks might be empty, remove corresponding 3D points
    filtered_tracks = 0
    for point3d_id, track in enumerate(model.tracks):
        if len(track) < 2:
            for (im_id, kp_id) in track:
                model.points2d_to_3d_id[im_id][kp_id] = -1
            filtered_tracks += 1
            model.tracks[point3d_id] = []
            model.points3d[:, point3d_id] = np.nan

    print(
        f"Filtered out {filtered_observations} 2D-3D observations and {filtered_tracks} 3D points."
    )


def two_view_init(model, pair):
    # Initialize the model from an image pair
    (im1, im2) = pair

    # Get matched 2D points
    ind1 = model.matches[pair][0, :]
    ind2 = model.matches[pair][1, :]
    x1 = model.points2d[im1][:, ind1]
    x2 = model.points2d[im2][:, ind2]

    # Call student-implemented RANSAC essential matrix estimation
    P1, P2 = ransac_essential_matrix(x1, x2, model.K)

    # Write cameras into the model
    model.cameras[im1] = P1
    model.cameras[im2] = P2


def retriangulate_pair(model, pair):
    # Looks at matches between an image pair and tries to triangulate new 3D points
    # Ignores any keypoints which are already part of a track
    (im1, im2) = pair
    ind1 = model.matches[pair][0, :]
    ind2 = model.matches[pair][1, :]

    # Look for matches where we have no 3D point yet
    candidate = np.array(model.points2d_to_3d_id[im1][ind1] == -1) & np.array(
        model.points2d_to_3d_id[im2][ind2] == -1
    )
    if not np.any(candidate):
        return

    ind1 = ind1[candidate]
    ind2 = ind2[candidate]

    # Get 2D points and cameras
    x1 = model.points2d[im1][:, ind1]
    x2 = model.points2d[im2][:, ind2]
    P1 = model.cameras[im1]
    P2 = model.cameras[im2]

    # Call student-implemented triangulation code
    X = triangulate_pair(P1, P2, x1, x2)

    # Compute reprojection errors and filter outliers
    err1 = compute_reprojection_errors(P1, X, x1)
    err2 = compute_reprojection_errors(P2, X, x2)
    inl = np.maximum(err1, err2) < tol_px

    # Keep only inliers
    ind1 = ind1[inl]
    ind2 = ind2[inl]
    X = X[:, inl]

    # Initialize new tracks in the model
    initialize_new_tracks(model, pair, ind1, ind2, X)
    print(f"Triangulated {X.shape[1]} new 3D points from images {im1} and {im2}")


def registered_images(model):
    # Returns a list of image indices for registered images
    return [k for k, cam in enumerate(model.cameras) if cam is not None]


def viz_model(model, block=False):
    # Visualizes the current 3D model using matplotlib
    # If block=True, the visualization will block execution until closed.
    if block:
        plt.ioff()
    else:
        plt.ion()

    # Try to use the same window if one exists
    fig = plt.gcf()
    if not plt.get_fignums():
        # Otherwise, create new figure
        fig = plt.figure()
    fig.clf()

    # 3D scatter plot of points (the options here make rendering faster)
    ax = plt.axes(projection="3d")
    ax.scatter3D(
        model.points3d[0, :],
        model.points3d[1, :],
        model.points3d[2, :],
        c="b",
        s=1,
        marker=".",
        depthshade=False,
        antialiased=False,
    )

    # Plot cameras
    cam_centers = []
    for cam in model.cameras:
        if cam is None:
            continue
        utils.plotcam_frustum(cam)
        cam_center = -np.linalg.inv(cam[:, :3]) @ cam[:, 3]
        cam_centers.append(cam_center)

    # Adjust axes. We use robust=True to avoid problems when there are few points
    # which are very far from the rest. must_include makes sure that all camera
    # centers are visible in the plot still
    utils.set_axes_equal(ax, robust=True, must_include=np.array(cam_centers))
    plt.show(block=block)
    if not block:
        # This ensures the window shows up if we are not blocking
        plt.pause(0.001)


def print_model_summary(model):
    # Print a summary of the current model
    n_cams = len(registered_images(model))
    n_points = model.points3d.shape[1]
    track_lengths = [len(track) for track in model.tracks]

    # Compute the reprojection errors across all registered images
    errs = []
    for im_id in registered_images(model):
        p3d_ids = model.points2d_to_3d_id[im_id]
        vis = p3d_ids > -1
        p2d = model.points2d[im_id][:, vis]
        p3d = model.points3d[:, p3d_ids[vis]]
        err = compute_reprojection_errors(model.cameras[im_id], p3d, p2d)
        errs += err.tolist()
    if len(errs) == 0:
        mean_err = np.nan
        max_err = np.nan
    else:
        mean_err = np.mean(errs)
        max_err = np.max(errs)

    mean_track_length = np.mean(track_lengths) if len(track_lengths) > 0 else 0

    print(
        f"---> Model summary: {n_cams} cameras, {n_points} 3D points ({mean_track_length:.2f} track len), reprojection error: mean={mean_err:.2f}, max={max_err:.2f} pixels"
    )


if __name__ == "__main__":
    main()
