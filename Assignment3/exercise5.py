import utils
import numpy as np
import matplotlib.pyplot as plt

def normalize_points(x):
    m = np.mean(x[0:2,:], axis=1)
    s = np.std(x[0:2,:], axis=1)

    N = np.array([[1/s[0], 0, -m[0]/s[0]],
                  [0, 1/s[1], -m[1]/s[1]],
                  [0, 0, 1]])
    
    return N

def to_homogeneous(X):
    return np.vstack((X, np.ones((1, X.shape[1]))))

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
    
    N1,N2 = np.eye(3), np.eye(3)
    
    norm = True

    if norm:
        # Normalize
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

    plt.title("Epipolar lines and points")
    plt.show()

    # Compute distances and plot histogram
    distances = np.abs(np.sum(l * x2_h, axis=0))
    plt.hist(distances, bins=100)

    if norm:
        plt.title("Epipolar distance histogram (normalized case)")
    else:
        plt.title("Epipolar distance histogram (unnormalized case)")

    plt.show()

    print("Mean epipolar distance:", np.mean(distances))
