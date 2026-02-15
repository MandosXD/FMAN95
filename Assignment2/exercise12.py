import utils
import numpy as np
import matplotlib.pyplot as plt
from exercise10 import to_homogeneous, normalize_points, dlt_resection, fix_sign, pflat

# Loading the .npz file
data = np.load('data/A2_ex10_data.npz')
print(list(data.keys())) # Print names of variables in data

Xcube = data['Xcube']          # 3xN 3D model
x1 = data['x1cube']            # 2xN image points
x2 = data['x2cube']
cube_edges = data['cube_edges'] # 126x2

im1 = plt.imread('data/A2_ex10_cube1.jpg')
im2 = plt.imread('data/A2_ex10_cube2.jpg')

# Load SIFT matches from exercise 11
data11 = np.load('data/A2_ex11_data.npz')

x1_sift = data11['x1']   # 2xN
x2_sift = data11['x2']

print(np.shape(x1_sift))
# Convert to homogeneous
x1_sift_h = to_homogeneous(x1_sift)
x2_sift_h = to_homogeneous(x2_sift)

# Abridged version of ex10
Xcube_h = to_homogeneous(Xcube)
x1_h = to_homogeneous(x1)
x2_h = to_homogeneous(x2)

x1n, N1 = normalize_points(x1_h)
x2n, N2 = normalize_points(x2_h)

P1n = dlt_resection(Xcube_h, x1n)
P2n = dlt_resection(Xcube_h, x2n)

P1n = fix_sign(P1n)
P2n = fix_sign(P2n)

# Transform to the original coordinate system
P1 = np.linalg.inv(N1) @ P1n
P2 = np.linalg.inv(N2) @ P2n

# Part a)
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

# Triangulate all points
X_est = []

for i in range(x1_sift_h.shape[1]):
    X = triangulate_point(P1, P2,
                          x1_sift_h[:,i],
                          x2_sift_h[:,i])
    X_est.append(X)

X_est = np.array(X_est).T  # 4xN
X_est = pflat(X_est)

# Part b)
x1proj = pflat(P1 @ X_est)
x2proj = pflat(P2 @ X_est)

# Computes the reprojection error in the first image
res1 = np.sqrt(np.sum((x1proj[0:2,:] - x1_sift)**2, axis=0))
res2 = np.sqrt(np.sum((x2proj[0:2,:] - x2_sift)**2, axis=0))

# Finds the points with reprojection error less than 3 pixels in both images
inliers = np.maximum(res1, res2) < 3.0

print("Total matches:", X_est.shape[1])
print("Inliers:", np.sum(inliers))

# Removes points that are not good enough.
X_inliers = X_est[:, inliers]
x1_inliers = x1_sift[:, inliers]
x2_inliers = x2_sift[:, inliers]

x1proj_in = x1proj[:, inliers]
x2proj_in = x2proj[:, inliers]

# Part c)
# Plot image 1
plt.figure()
plt.imshow(im1)
plt.plot(x1_inliers[0,:], x1_inliers[1,:], '*',  label = "image points")
plt.plot(x1proj_in[0,:], x1proj_in[1,:], 'ro',  markerfacecolor='none', label = "projected points")
plt.title("Inlier reprojections - Image 1")
plt.axis('off')
plt.legend()
plt.show()

# Plot image 2
plt.figure()
plt.imshow(im2)
plt.plot(x2_inliers[0,:], x2_inliers[1,:], '*', label = "image points")
plt.plot(x2proj_in[0,:], x2proj_in[1,:], 'ro',  markerfacecolor='none', label = "projected points")
plt.title("Inlier reprojections - Image 2")
plt.axis('off')
plt.show()


# Part d)
plt.figure()
ax = plt.subplot(projection='3d')
[ax.plot(*Xcube[:, (s,e)], 'b-') for s,e in cube_edges]

# Plot cameras
utils.plotcam(P1, ax)
utils.plotcam(P2, ax)

# Plot triangulated points
ax.scatter(X_inliers[0,:], X_inliers[1,:], X_inliers[2,:], c='r', s=1)
utils.set_axes_equal(ax)
plt.title("Triangulated inlier points + cube + cameras")
plt.show()