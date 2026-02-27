import utils
import numpy as np
import matplotlib.pyplot as plt

def to_homogeneous(X):
    return np.vstack((X, np.ones((1, X.shape[1]))))

def pflat(X):
    X = np.asarray(X)
    return X / X[-1] # Divide by the last row

def depth(P, X):
    """
    Compute depth of 3D points X w.r.t. camera P.

    """
    # Left 3x3 block
    A = P[:, :3]

    # sign(det(A))
    sign_det = np.sign(np.linalg.det(A))

    # Norm of third row of A
    A3_norm = np.linalg.norm(A[2, :])

    # Project points
    PX = P @ X

    # lambda = third row of PX
    lam = PX[2, :]

    # rho = fourth coordinate of X
    rho = X[3, :]

    return sign_det * lam / (A3_norm * rho)

# From A2_ex12
def triangulate_point(P1, P2, x1, x2):
    A = np.zeros((4,4))

    # Set up the system of equations from the two cameras and two corresponding points
    A[0] = x1[0] * P1[2] - P1[0]
    A[1] = x1[1] * P1[2] - P1[1]
    A[2] = x2[0] * P2[2] - P2[0]
    A[3] = x2[1] * P2[2] - P2[1]

    # Solve the equation using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X

def ransac_essential(x1n, x2n, x1_h, x2_h, K_inv, k=100, threshold=5.0):
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

                # Compute point-to-line distances
                d2 = np.abs(np.sum(l2 * x2_h, axis=0))
                d1 = np.abs(np.sum(l1 * x1_h, axis=0))

                # Inlier condition distance must be small in BOTH images
                inliers = (d1 < threshold) & (d2 < threshold)
                num_inliers = np.sum(inliers)

                if num_inliers > max_inliers:
                    max_inliers = num_inliers
                    best_inliers = inliers
                    best_E = E
    
    # Return the final model parameters and inliers
    return best_E, best_inliers

if __name__ == "__main__":
    # Loading the .npz file
    data = np.load('data/A4_ex5_data.npz')
    print(list(data.keys())) # Print names of variables in data

    x1 = data['x1']            # 2xN image points
    x2 = data['x2']
    K = data['K']

    x1_h = to_homogeneous(x1)
    x2_h = to_homogeneous(x2)
    
    # Normalize points
    K_inv = np.linalg.inv(K)
    x1n = K_inv @ x1_h
    x2n = K_inv @ x2_h

    im1 = plt.imread('data/A4_ex5_im1.jpg')
    im2 = plt.imread('data/A4_ex5_im2.jpg')

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

    E, inliers = ransac_essential(x1n, x2n, x1_h, x2_h, K_inv)

    print(E)
    print(np.sum(inliers))

    # Computes the avergae number of inliers over N runs
    if False:
        N = 100  # number of runs
        total_inliers = 0

        for _ in range(N):
            _, inliers = ransac_essential(x1n, x2n, x1_h, x2_h, K_inv)
            
            num_inliers = np.sum(inliers)
            print(num_inliers)
            
            total_inliers += num_inliers

        average_inliers = total_inliers / N
        print(f"\nAverage number of inliers over {N} runs: {average_inliers}")

    x1n_inlier = x1n[:, inliers]
    x2n_inlier = x2n[:, inliers]

    # Note that this returns the V matrix already transposed!
    u, s, vt = np.linalg.svd(E)
    if np.linalg.det(u @ vt) < 0:
        vt = -vt
    E = u @ np.diag([1, 1, 0]) @ vt
    # Note: Re-computing svd on E may still give U and V that does not fulfill
    # det(U*V') = 1 since the svd is not unique.
    # So don't recompute the svd after this step.

    # Epipolar constraints, should be roughly zero
    epi = np.sum(x2n_inlier * (E @ x1n_inlier), axis=0)

    print("Mean epipolar constraint (normalized):", np.mean(np.abs(epi)))

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
            X[:, i] = triangulate_point(P1, P2, x1n_inlier[:, i], x2n_inlier[:, i])
        
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

    # 3D plot of inlier points and cameras
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plot 3D points
    ax.scatter(best_X[0], best_X[1], best_X[2], s=2)

    # Plot cameras
    utils.plotcam(P1, ax)
    utils.plotcam(best_P2, ax)

    plt.show()

    # Reproject points
    x1proj = pflat(P1_pixel @ best_X)
    x2proj = pflat(P2_pixel @ best_X)

    x1_inlier = x1[:, inliers]
    x2_inlier = x2[:, inliers]

    # Compute Euclidean reprojection error
    err1 = np.sqrt(np.sum((x1_inlier - x1proj[:2, :])**2, axis=0))
    err2 = np.sqrt(np.sum((x2_inlier - x2proj[:2, :])**2, axis=0))

    print("Average reprojection error image 1:", np.mean(err1))
    print("Average reprojection error image 2:", np.mean(err2))

    # Plot histograms
    plt.figure()
    plt.hist(err1, bins=100)
    plt.title("Reprojection Error Histogram - Image 1")
    plt.xlabel("Reprojection error (pixels)")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure()
    plt.hist(err2, bins=100)
    plt.title("Reprojection Error Histogram - Image 2")
    plt.xlabel("Reprojection error (pixels)")
    plt.ylabel("Frequency")
    plt.show()

    # Plot the reprojected points in the image (not required)
    if False:
        plt.figure()
        plt.imshow(im1)
        plt.plot(x1_inlier[0,:], x1_inlier[1,:], '*', label = "image points")
        plt.plot(x1proj[0,:], x1proj[1,:], 'ro',  markerfacecolor='none', label = "projected points")
        plt.title("Reprojection image 1")
        plt.axis('off')
        plt.legend()
        plt.show()

        plt.figure()
        plt.imshow(im2)
        plt.plot(x2_inlier[0,:], x2_inlier[1,:], '*', label = "image points")
        plt.plot(x2proj[0,:], x2proj[1,:], 'ro',  markerfacecolor='none', label = "projected points")
        plt.title("Reprojection image 2")
        plt.axis('off')
        plt.legend()
        plt.show()

