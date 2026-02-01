import numpy as np
import matplotlib.pyplot as plt
from exercise2 import pflat

# Loading the .npz file
data = np.load('data/A1_ex7_data.npz')
print(list(data.keys())) # Print names of variables in data

startpoints = data['startpoints']
endpoints = data['endpoints']

# Convert to homogeneous coordinates (3 x N)
ones = np.ones((1, startpoints.shape[1]))
start_h = np.vstack((startpoints, ones))
end_h = np.vstack((endpoints, ones))

# Define projective mappings
H1 = np.array([
    [np.sqrt(3), -1, 1],
    [1,  np.sqrt(3), 1],
    [0,  0,  2]
])

H2 = np.array([
    [1, -1, 1],
    [1,  1, 0],
    [0,  0, 1]
])

H3 = np.array([
    [1, 1, 0],
    [0, 2, 0],
    [0, 0, 1]
])

H4 = np.array([
    [np.sqrt(3), -1, 1],
    [1,  np.sqrt(3), 1],
    [1/4, 1/2, 2]
])

Hs = [H1, H2, H3, H4]
titles = ['H1', 'H2', 'H3', 'H4']

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for ax, H, title in zip(axes, Hs, titles):
    # Transform points
    start_t = pflat(H @ start_h)
    end_t = pflat(H @ end_h)

    ax.plot(
        np.vstack((start_t[0, :], end_t[0, :])),
        np.vstack((start_t[1, :], end_t[1, :])),
        'b-'
    )
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

plt.tight_layout()
plt.show()