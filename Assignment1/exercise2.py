import utils
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    # Loading the .npz file
    data = np.load('data/A1_ex2_data.npz')
    print(list(data.keys())) # Print names of variables in data

    x2D = data['x2D']
    x3D = data['x3D']

    # Apply pflat
    x2D_flat = pflat(x2D)
    x3D_flat = pflat(x3D)


    # Plotting in 2D:
    fig = plt.figure()
    plt.plot(x2D_flat[0, :], x2D_flat[1, :], '.')
    plt.axis('equal') # Equal aspect ratio in 2D plots

    # Plotting in 3D:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x3D_flat[0, :], x3D_flat[1, :], x3D_flat[2, :], marker='.')
    utils.set_axes_equal(ax) # Utility function for equal aspect ratio in 3D

    plt.show()