import utils
import numpy as np
import matplotlib.pyplot as plt

# Helper functions
def pflat(X):
    """
    Normalize homogeneous coordinates by dividing by the last row.

    Parameters:
    - X: ndarray of shape (d, N) or (d,) representing Homogeneous coordinates
    Returns:
    - ndarray of the same shape as X representing Normalized coordinates
    """
    X = np.asarray(X)
    return X / X[-1] # Divide by the last row

def to_homogeneous(X):
    return np.vstack((X, np.ones((1, X.shape[1]))))

# Loading the .npz file
data = np.load('data/A2_ex10_data.npz')
print(list(data.keys())) # Print names of variables in data

Xcube = data['Xcube']          # 3xN 3D model
x1 = data['x1cube']            # 2xN image points
x2 = data['x2cube']
cube_edges = data['cube_edges'] # 126x2

im1 = plt.imread('data/A2_ex10_cube1.jpg')
im2 = plt.imread('data/A2_ex10_cube2.jpg')

# Convert to homogeneous
Xcube_h = to_homogeneous(Xcube)
x1_h = to_homogeneous(x1)
x2_h = to_homogeneous(x2)

# Part a)

def normalize_points(x):
    m = np.mean(x[0:2,:], axis=1)
    s = np.std(x[0:2,:], axis=1)

    N = np.array([[1/s[0], 0, -m[0]/s[0]],
                  [0, 1/s[1], -m[1]/s[1]],
                  [0, 0, 1]])

    x_norm = N @ x
    return x_norm, N

x1n, N1 = normalize_points(x1_h)
x2n, N2 = normalize_points(x2_h)

# Plot normalized points
plt.figure()
plt.scatter(x1n[0,:], x1n[1,:])
plt.title("Normalized image 1 points")
plt.axis('equal')
plt.show()

plt.figure()
plt.scatter(x2n[0,:], x2n[1,:])
plt.title("Normalized image 2 points")
plt.axis('equal')
plt.show()


# part b)

def dlt_resection(X, x):
    n = X.shape[1]
    M = []

    for i in range(n):
        Xi = X[:, i]
        xi = x[0,i]
        yi = x[1,i]

        row1 = np.hstack([np.zeros(4), -Xi, yi*Xi])
        row2 = np.hstack([Xi, np.zeros(4), -xi*Xi])

        M.append(row1)
        M.append(row2)

    M = np.array(M)

    U, S, Vt = np.linalg.svd(M)
    sol = Vt[-1]

    print("Smallest singular value:", S[-1])
    print("||Mv||:", np.linalg.norm(M @ sol))

    P = sol.reshape(3,4)
    return P

P1n = dlt_resection(Xcube_h, x1n)
P2n = dlt_resection(Xcube_h, x2n)

# Part c)
def fix_sign(P):
    A = P[:, :3]
    if np.linalg.det(A) < 0:
        P = -P
    return P

P1n = fix_sign(P1n)
P2n = fix_sign(P2n)

# Part d)
P1 = np.linalg.inv(N1) @ P1n
P2 = np.linalg.inv(N2) @ P2n

x1proj = pflat(P1 @ Xcube_h)
x2proj = pflat(P2 @ Xcube_h)

plt.figure()
plt.imshow(im1)
plt.plot(x1[0,:], x1[1,:], '*')
plt.plot(x1proj[0,:], x1proj[1,:], 'ro')
plt.title("Reprojection image 1")
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(im2)
plt.plot(x2[0,:], x2[1,:], '*')
plt.plot(x2proj[0,:], x2proj[1,:], 'ro')
plt.title("Reprojection image 2")
plt.axis('off')
plt.show()

# 3D visualization
fig = plt.figure()
ax = plt.subplot(projection='3d')

[ax.plot(*Xcube[:, (s,e)], 'b-') for s,e in cube_edges]

utils.plotcam(P1, ax)
utils.plotcam(P2, ax)

utils.set_axes_equal(ax)
plt.title("Cube and estimated cameras")
plt.show()

# Part e)
def compute_intrinsics(P):
    K, _ = utils.rq(P)
    K /= K[2,2]
    return K

K1 = compute_intrinsics(P1)
K2 = compute_intrinsics(P2)

print("Intrinsic matrix camera 1:\n", np.round(K1,3))
print("Intrinsic matrix camera 2:\n", np.round(K2,3))
