import utils
import numpy as np
import matplotlib.pyplot as plt

# Helper function
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


# Loading the .npz file
data = np.load('data/A2_ex2_data.npz')
print(list(data.keys())) # Print names of variables in data

X = data['X']  # 4x9471 homogeneous 3D points
x = data['x']  # list of 3x9471 image points per camera
P = data['P']  # list of camera matrices
imfiles = data['imfiles']  # list of image filenames


# Part a)
# Plots a small '.' at all the 3D points.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[0, :], X[1, :], X[2, :], marker='.', s=1)

for Pi in P:
    utils.plotcam(Pi, ax)

utils.set_axes_equal(ax) # Equal aspect ratio in 3D
plt.show()

# Part c)

j = 0  # choose camera 0

plt.figure()
#Reads the imagefile with name in imfiles[i]
im = plt.imread("data/" + imfiles[j]);
plt.imshow(im)
# Determines which of the points are visible in image i
visible = np.isfinite(x[j][0,:])
# Plots a '*' at each point coordinate
plt.plot(x[j][0,visible], x[j][1,visible],'*', label = "visible points");
# Compute the projection and normalize it
xproj = pflat(P[j] @ X)
# Plots a red 'o' at each visible point in xproj
plt.plot(xproj[0,visible], xproj[1,visible],'ro', markerfacecolor='none', label = "projected points");

plt.title("Original reprojection")
plt.axis('off')
plt.legend()
plt.show()


# Part d)

T1 = np.array([[1,0,0,0],
               [0,4,0,0],
               [0,0,1,0],
               [1/10,1/10,0,1]])

T2 = np.array([[1,0,0,0],
               [0,1,0,0],
               [0,0,1,0],
               [1/16,1/16,0,1]])

for idx, T in enumerate([T1, T2]):

    X_new = T @ X
    P_new = [Pi @ np.linalg.inv(T) for Pi in P]

    Xeu_new = pflat(X_new)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(Xeu_new[0,:], Xeu_new[1,:], Xeu_new[2,:],
               marker='.', s=1)

    for Pi in P_new:
        utils.plotcam(Pi, ax)

    utils.set_axes_equal(ax)
    plt.title(f"Projective Reconstruction T{idx+1}")
    plt.show()
    
    # Part f)

    xproj_new = pflat(P_new[j] @ X_new)

    plt.figure()
    plt.imshow(im)
    plt.plot(x[j][0,visible], x[j][1,visible], '*', label = "visible points")
    plt.plot(xproj_new[0,visible], xproj_new[1,visible], 'ro',  markerfacecolor='none', label = "projected points")
    plt.title(f"Reprojection after T{idx+1}")
    plt.axis('off')
    plt.legend()
    plt.show()
