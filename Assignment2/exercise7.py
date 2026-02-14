import utils
import numpy as np

# Loading the .npz file
data = np.load('data/A2_ex2_data.npz')
print(list(data.keys())) # Print names of variables in data

P = data['P']  # list of camera matrices


# The transformation matricies from ex2
T1 = np.array([[1,0,0,0],
               [0,4,0,0],
               [0,0,1,0],
               [1/10,1/10,0,1]])

T2 = np.array([[1,0,0,0],
               [0,1,0,0],
               [0,0,1,0],
               [1/16,1/16,0,1]])

for i, T in enumerate([T1, T2]):

    P_new = P[1] @ np.linalg.inv(T)

    K, _ = utils.rq(P_new)

    K /= K[2,2]

    print(f"\nFor transformation T{i+1} the intrinsic matrix K is:")
    print(np.round(K, 3))