import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space

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

    im1 = plt.imread('data/A3_ex5_kronan1.jpg')
    im2 = plt.imread('data/A3_ex5_kronan2.jpg')

    # Convert to homogeneous
    x1_h = to_homogeneous(x1)
    x2_h = to_homogeneous(x2)
    
    # Normalize points
    N1 = normalize_points(x1_h)
    N2 = normalize_points(x2_h)
    x1n = N1 @ x1_h
    x2n = N2 @ x2_h

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
    Fn = v.reshape(3, 3)

    # Enforce det(F) = 0 (rank-2 constraint)
    U_f, S_f, Vt_f = np.linalg.svd(Fn)
    S_f[2] = 0
    Fn = U_f @ np.diag(S_f) @ Vt_f

    # Epipolar constraints, should be roughly zero
    epi = np.sum(x2n * (Fn @ x1n), axis=0)

    print("Mean epipolar constraint (normalized):", np.mean(np.abs(epi)))
    print("Determinant is:", np.linalg.det(Fn))

    # Unnormalized fundamental matrix
    F = N2.T @ Fn @ N1

    # Compute second epipole as the left nullspace of F
    e2 = null_space(F.T)[:, 0]

    # Constructs the cross-product matrix
    e2x = np.array([
                [0, -e2[2], e2[1]],
                [e2[2], 0, -e2[0]],
                [-e2[1], e2[0], 0 ]
                ])

    # First camera
    P1 = np.hstack((np.eye(3), np.zeros((3,1))))

    # Second camera: P2 = [[e2]_x F | e2]
    P2 = np.hstack((e2x @ F, e2.reshape(3,1)))

    # Triangulate all points
    X = np.zeros((4, x1_h.shape[1]))
    for i in range(x1_h.shape[1]):
        X[:, i] = triangulate_point(P1, P2, x1_h[:, i], x2_h[:, i])

    X = pflat(X)

    x1proj = pflat(P1 @ X)
    x2proj = pflat(P2 @ X)

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
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[0,:], X[1,:], X[2,:],
               marker='.', s=1)

    plt.title("Triangulated points")
    plt.show()

