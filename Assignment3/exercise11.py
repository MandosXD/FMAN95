import utils
import numpy as np
import matplotlib.pyplot as plt

def pflat(X):
    X = np.asarray(X)
    return X / X[-1] # Divide by the last row

def normalize_points(x):
    m = np.mean(x[0:2,:], axis=1)
    s = np.std(x[0:2,:], axis=1)

    N = np.array([[1/s[0], 0, -m[0]/s[0]],
                  [0, 1/s[1], -m[1]/s[1]],
                  [0, 0, 1]])
    
    return N

def to_homogeneous(X):
    return np.vstack((X, np.ones((1, X.shape[1]))))

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

if __name__ == "__main__":
    # Loading the .npz file
    data = np.load('data/A3_ex5_data.npz')
    print(list(data.keys())) # Print names of variables in data

    x1 = data['x1']            # 2xN image points
    x2 = data['x2']

    data9 = np.load('data/A3_ex9_data.npz')
    K = data9['K']

    im1 = plt.imread('data/A3_ex5_kronan1.jpg')
    im2 = plt.imread('data/A3_ex5_kronan2.jpg')

    # Convert to homogeneous
    x1_h = to_homogeneous(x1)
    x2_h = to_homogeneous(x2)
    
    # Normalize points
    K_inv = np.linalg.inv(K)
    x1n = K_inv @ x1_h
    x2n = K_inv @ x2_h

    # Build M matrix for epipolar constraints
    M = np.zeros((x1n.shape[1], 9))
    for i in range(x1n.shape[1]):
        xx = np.outer(x2n[:, i], x1n[:, i]) # 3×3 outer product
        M[i, :] = xx.ravel()

    # Solve Mv = 0 using SVD
    _, S, Vt = np.linalg.svd(M)
    v = Vt[-1]

    print("Smallest singular value:", S[-1])
    print("||Mv||:", np.linalg.norm(M @ v))

    # Reshape solution vector v to a 3x3 matrix
    E = v.reshape(3, 3)

    # Note that this returns the V matrix already transposed!
    u, s, vt = np.linalg.svd(E)
    if np.linalg.det(u @ vt) < 0:
        vt = -vt
    E = u @ np.diag([1, 1, 0]) @ vt
    # Note: Re-computing svd on E may still give U and V that does not fulfill
    # det(U*V') = 1 since the svd is not unique.
    # So don't recompute the svd after this step.

    # Epipolar constraints, should be roughly zero
    epi = np.sum(x2n * (E @ x1n), axis=0)

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
        X = np.zeros((4, x1n.shape[1]))

        for i in range(x1n.shape[1]):
            X[:, i] = triangulate_point(P1, P2, x1n[:, i], x2n[:, i])
        X = pflat(X)
        
        # Check that the points are infront of both cameras
        # Camera 1: Z > 0
        Z1 = X[2, :]

        # Camera 2: Z > 0 after projection
        X_cam2 = P2 @ X
        Z2 = X_cam2[2, :]

        count = np.sum((Z1 > 0) & (Z2 > 0))

        print(f"{name}: {count} points in front")

        if count > best_count:
            best_count = count
            best_P2 = P2
            best_X = X
            best_name = name

    print(f"Selected camera is {best_name} with {best_count} points in front")

    # Convert cameras back to pixel coordinates
    P1_pixel = K @ P1
    P2_pixel = K @ best_P2

    # Reproject points
    x1proj = pflat(P1_pixel @ best_X)
    x2proj = pflat(P2_pixel @ best_X)

    # Plot image 1
    plt.figure()
    plt.imshow(im1)
    plt.plot(x1[0,:], x1[1,:], '*', label = "image points")
    plt.plot(x1proj[0,:], x1proj[1,:], 'ro',  markerfacecolor='none', label = "projected points")
    plt.title("Reprojection image 1")
    plt.axis('off')
    plt.legend()
    plt.show()

    plt.figure()
    plt.imshow(im2)
    plt.plot(x2[0,:], x2[1,:], '*', label = "image points")
    plt.plot(x2proj[0,:], x2proj[1,:], 'ro',  markerfacecolor='none', label = "projected points")
    plt.title("Reprojection image 2")
    plt.axis('off')
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plot 3D points
    ax.scatter(best_X[0], best_X[1], best_X[2], s=2)

    # Plot cameras
    utils.plotcam(P1, ax)
    utils.plotcam(best_P2, ax)

    plt.show()
