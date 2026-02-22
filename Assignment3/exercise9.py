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

    print("Calibration matrix is \n", K)
    
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

    # Enforce 2 equal singular values with third one being zero
    U_e, S_e, Vt_e = np.linalg.svd(E)
    if np.linalg.det(U_e @ Vt_e) < 0: #Ensure proper rotation
        Vt_e = -Vt_e
    E = U_e @ np.diag([1,1,0]) @ Vt_e # Arbitrary scale

    # Epipolar constraints, should be roughly zero
    epi = np.sum(x2n * (E @ x1n), axis=0)

    print("Mean epipolar constraint (Essential matrix):", np.mean(np.abs(epi)))

    # Computing the unnormalized fundamental matrix F
    F = K_inv.T @ E @ K_inv

    # Epipolar lines
    l = F @ x1_h

    # Normalization factor (1×N)
    norms = np.sqrt(l[0, :]**2 + l[1, :]**2)
    # Normalize each line so first two components form unit normal
    l = l / norms

    # Index vector with 20 random indices
    ind = np.random.choice(x2.shape[1], size=20, replace=False)

    # Plot image + points + epipolar lines
    plt.figure()
    plt.imshow(im2)
    plt.scatter(x2[0, ind], x2[1, ind], c='b')

    # Plot only the indexed lines
    l_plot = l[:, ind]

    for i in range(l_plot.shape[1]):
        utils.draw_hom_line(l_plot[:, i])

    plt.title("Epipolar lines and points (Essential matrix)")
    plt.show()

    # Compute distances and plot histogram
    distances = np.abs(np.sum(l * x2_h, axis=0))
    plt.hist(distances, bins=100)

    plt.title("Epipolar distance histogram (Essential matrix)")
    plt.show()

    print("Mean epipolar distance:", np.mean(distances))

