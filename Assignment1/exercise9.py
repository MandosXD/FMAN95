import utils
import numpy as np
import matplotlib.pyplot as plt
from exercise2 import pflat

# Read the images
im1 = plt.imread('data/A1_ex9_image1.jpg')
im2 = plt.imread('data/A1_ex9_image2.jpg')

# Draw the raw images
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(im1, cmap='gray')
axes[0].set_title('Image 1')
axes[0].axis('off')

axes[1].imshow(im2, cmap='gray')
axes[1].set_title('Image 2')
axes[1].axis('off')

plt.tight_layout()

# Loading the .npz file
data = np.load('data/A1_ex9_data.npz')
print(list(data.keys())) # Print names of variables in data

K = data['K']
R1 = data['R1']
t1 = data['t1']
R2 = data['R2']
t2 = data['t2']
U = data['U']

# Camera matrices
P1 = K @ np.hstack((R1, t1.reshape(3, 1)))
P2 = K @ np.hstack((R2, t2.reshape(3, 1)))

# Camera centers
c1 = -R1.T @ t1
c2 = -R2.T @ t2

# Principal axes (extract the third row of R)
v1 = R1[2]
v2 = R2[2]

# Normalize to unit length
v1 = v1 / np.linalg.norm(v1)
v2 = v2 / np.linalg.norm(v2)

# Print results
print("Camera center 1:", c1)
print("Camera center 2:", c2)
print("Principal axis 1:", v1)
print("Principal axis 2:", v2)

U_flat = pflat(U)

# Plot 3D points
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(U_flat[0, :], U_flat[1, :], U_flat[2, :], marker='.')

# Plot camera centers
ax.scatter(*c1, color='r', label='Camera 1')
ax.scatter(*c2, color='b', label='Camera 2')

# Plot principal axes as vectors
ax.quiver(*c1, *v1, length=1, color='r')
ax.quiver(*c2, *v2, length=1, color='b')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Points, Camera Centers, and Principal Axes')
utils.set_axes_equal(ax)
ax.legend()

# Project points
x1 = pflat(P1 @ U)
x2 = pflat(P2 @ U)

# Plot 2D projections
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Image 1
axes[0].imshow(im1, cmap='gray')
axes[0].plot(x1[0, :], x1[1, :], '.', markersize=2)
axes[0].set_title('Image 1 with projected points')
axes[0].axis('off')

# Image 2
axes[1].imshow(im2, cmap='gray')
axes[1].plot(x2[0, :], x2[1, :], '.', markersize=2)
axes[1].set_title('Image 2 with projected points')
axes[1].axis('off')

plt.tight_layout()

plt.show()