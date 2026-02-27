import utils
import numpy as np
import matplotlib.pyplot as plt

def to_homogeneous(X):
    return np.vstack((X, np.ones((1, X.shape[1]))))

def pflat(X):
    X = np.asarray(X)
    return X / X[-1] # Divide by the last row

def DLT_homography(x1_h, x2_h):
    N = x1_h.shape[1]
    M = []

    for i in range(N):
        x = x1_h[:, i]     # first image homogenous coordinates
        xp = x2_h[:, i]   # second image homogenous coordinates

        # Setup the rows of the matrix M
        row1 = np.hstack([-x, np.zeros(3),  x*xp[0]])
        row2 = np.hstack([np.zeros(3), -x, x*xp[1]])

        M.append(row1)
        M.append(row2)

    M = np.array(M)

    # Solve the DLT system using SVD
    _, _, Vt = np.linalg.svd(M)
    sol = Vt[-1]

    # Return the estimated homography matrix
    H = sol.reshape(3,3)
    H /= H[2,2]
    return H

def ransac_homography(x1_h, x2_h, k=100, threshold=5.0):
    best_inliers = None
    max_inliers = 0

    for _ in range(k):
        # Randomly choose 4 correspondences
        indices = np.random.choice(x1_h.shape[1], size=4, replace=False)
        x1_h_sample = x1_h[:, indices]
        x2_h_sample = x2_h[:, indices]

        # Fit a Homography to the 4 correspondences
        H_est = DLT_homography(x1_h_sample, x2_h_sample)

        # Compute the inliers
        x2_pred = pflat(H_est @ x1_h)
        errors = np.linalg.norm(x2_pred[:2] - x2_h[:2], axis=0)
        inliers = errors < threshold
        num_inliers = np.sum(inliers)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers = inliers

    # Compute the best homography based on the inlier set
    x1_h_inliers = x1_h[:, best_inliers]
    x2_h_inliers = x2_h[:, best_inliers]
    H_final = DLT_homography(x1_h_inliers, x2_h_inliers)
    
    # Return the final model parameters and inliers
    return H_final, best_inliers

if __name__ == "__main__":
    # Loading the .npz file
    data = np.load('data/A4_ex3_data.npz')
    print(list(data.keys())) # Print names of variables in data

    x1 = data['x1']            # 2xN image points
    x2 = data['x2']

    x1_h = to_homogeneous(x1)
    x2_h = to_homogeneous(x2)

    im1 = plt.imread('data/A4_ex3_im1.jpg')
    im2 = plt.imread('data/A4_ex3_im2.jpg')

    # Draw both images together
    plt.figure()
    plt.imshow(np.hstack((im1, im2)), cmap='gray')

    # Select 10 random points
    index = np.random.permutation(x1.shape[1])[:10]

    # Shift x-coordinates for x2 with the image width
    w = im1.shape[1]
    plt.plot(
        np.vstack([x1[0, index], x2[0, index] + w]),
        np.vstack([x1[1, index], x2[1, index]]),
        '-'
    )

    plt.axis('off')
    plt.show()
    


    H, inliers = ransac_homography(x1_h, x2_h)

    print(H)
    print(np.sum(inliers))

    # Computes the avergae number of inliers over N runs
    if False:
        N = 100  # number of runs
        total_inliers = 0

        for _ in range(N):
            _, inliers = ransac_homography(x1_h, x2_h)
            
            num_inliers = np.sum(inliers)
            print(num_inliers)
            
            total_inliers += num_inliers

        average_inliers = total_inliers / N
        print(f"\nAverage number of inliers over {N} runs: {average_inliers}")


    # Warp im1 to im2's coordinate system and get the bounds
    im1_warp, bounds = utils.homography_warp_image(im1, H, bounds=None)
    # Warp im2 using the same bounds (identity homography keeps it unchanged)
    im2_warp, _ = utils.homography_warp_image(im2, np.eye(3), bounds=bounds)
    # Average where both images overlap, otherwise use whichever image has content
    im1_mask = np.any(im1_warp > 0, axis=-1)
    im2_mask = np.any(im2_warp > 0, axis=-1)
    blend = (im1_warp.astype(float) + im2_warp.astype(float)) / 2
    overlay = im2_warp.copy()
    overlay[im1_mask & ~im2_mask] = im1_warp[im1_mask & ~im2_mask]
    overlay[im1_mask & im2_mask] = blend[im1_mask & im2_mask].astype(overlay.dtype)

    plt.figure(figsize=(10,8))
    plt.imshow(overlay)
    plt.title("Image overlay Result")
    plt.axis("off")
    plt.show()