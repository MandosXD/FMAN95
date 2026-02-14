import utils
import numpy as np
import matplotlib.pyplot as plt

data = np.load('data/A2_ex11_data.npz')
x1, x2 = data['x1'], data['x2']
im1 = plt.imread('data/A2_ex10_cube1.jpg')
im2 = plt.imread('data/A2_ex10_cube2.jpg')

# Select 10 random points
index = np.random.permutation(x1.shape[1])[:10]

# Draw both images together
plt.figure()
plt.imshow(np.hstack((im1, im2)), cmap='gray')

# Shift x-coordinates for x2 with the image width
w = im1.shape[1]
plt.plot(
    np.vstack([x1[0, index], x2[0, index] + w]),
    np.vstack([x1[1, index], x2[1, index]]),
    '-'
)

plt.axis('off')
plt.show()