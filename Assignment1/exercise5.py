import utils
import numpy as np
import matplotlib.pyplot as plt
from exercise2 import pflat

# Read an image
im = plt.imread('data/A1_ex5_image.jpg')

# Draw an image
plt.imshow(im, cmap='gray') # cmap='gray' since it is a grayscale image
plt.axis('image') # similar to axis equal for images

# Loading the .npz file
data = np.load('data/A1_ex5_data.npz')
print(list(data.keys())) # Print names of variables in data

p1 = data['p1']
p2 = data['p2']
p3 = data['p3']

# Compute lines (cross product of point pairs)
l1 = np.cross(p1[:, 0], p1[:, 1])
l2 = np.cross(p2[:, 0], p2[:, 1])
l3 = np.cross(p3[:, 0], p3[:, 1])

# Plot points
plt.plot(p1[0, :], p1[1, :], 'ro')
plt.plot(p2[0, :], p2[1, :], 'go')
plt.plot(p3[0, :], p3[1, :], 'bo')

# Intersection of line 2 and line 3
x_inter = np.cross(l2, l3)
x_inter = pflat(x_inter) # Normalize coordinates

# Plot the intersection on the same figure
plt.plot(x_inter[0], x_inter[1], 'y*', markersize=12, label=f'Intersection of $l_2$ and $l_3$')

# Compute the distance between the intersection and l1
a, b, c = l1
x, y = x_inter[0], x_inter[1]
distance = np.abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)

# Relative distance (fraction of image diagonal)
height, width = im.shape[:2]  # image size
diag = np.sqrt(width**2 + height**2)
relative_distance = distance / diag

print("Distance from intersection to first line:", distance)
print(f"Distance relative to image diagonal: {relative_distance:.4f}")

# Plot a 2D line given in homogeneous coordinate l = (a,b,c)
utils.draw_hom_line(l1)
utils.draw_hom_line(l2)
utils.draw_hom_line(l3)

plt.legend()
plt.show()