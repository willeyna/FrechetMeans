import numpy as np
from itertools import product

'''
Bank of finite groups G <= O(n) of order k represented by functions which take X (pxn) to a new matrix
[G_1x, ..., G_kx] (px(kn))
Should probably be methods of a groups class or something cleaner tbh
'''

# cyclic translation
def Greps_Cyclic(x):
    p = len(x)
    Xreps = []
    # cyclic translation group generator
    C = np.eye(p)[:, np.append(np.arange(1,p),0)]
    for j in range(p):
        Cj = np.linalg.matrix_power(C, j)
        Xreps.append(np.dot(Cj, x))
    return np.hstack(Xreps)

# inverting sign
def Greps_pmId(x):
    Xreps = [x, -x]
    return np.hstack(Xreps)

# reflect along hyperplane orthogonal to y
def Greps_Reflect(x, y = None):
    p = len(x)
    # if not specified reflect about x_n axis
    if y is None:
        y = np.zeros(p)
        y[-1] = 1
    # householder reflector
    R = np.eye(p) - 2*np.outer(y,y)/np.linalg.norm(y)**2
    Xreps = [x, np.dot(R,x)]
    return np.hstack(Xreps)

# # cylic group of rotations of order n (about the first two coordinates)
# def Greps_Rotate(x, n = 3):
#     Xreps = []
#     p = len(x)
#     theta = (2*np.pi)/n
#     R = np.eye(p)
#     # Define the 2x2 rotation matrix for the given theta
#     rotation_submatrix = np.array([[np.cos(theta), -np.sin(theta)],
#                                    [np.sin(theta),  np.cos(theta)]])
#     R[0:2, 0:2] = rotation_submatrix

#     for j in range(n):
#         Rj = np.linalg.matrix_power(R, j)
#         Xreps.append(np.dot(Rj, x))
#     return np.hstack(Xreps)

# creates matrix where block diagonal is 2x2 rotation matrices R_i of order n[i]
def Greps_Rotate(x, n):
    Xreps = []
    p = len(x)
    half_p = p // 2
    R = np.eye(p)

    if len(n) > half_p:
        print("Warning: more rotations than p//2 inputted")

    # Construct block-diagonal rotation matrices for each block
    for i in range(half_p):
        if n[i] > 1:
            theta = (2 * np.pi) / n[i]
            rotation_submatrix = np.array([[np.cos(theta), -np.sin(theta)],
                                           [np.sin(theta),  np.cos(theta)]])
            R[2*i:2*i+2, 2*i:2*i+2] = rotation_submatrix

    # Apply rotations with different powers for each block and accumulate results
    max_n = max([n[i] for i in range(half_p)])
    for j in range(max_n):
        Rj = np.linalg.matrix_power(R, j)
        Xreps.append(np.dot(Rj, x))

    return np.hstack(Xreps)

# group w/ reflections about xy, xz planes and a pi-rotation about (1,1,1) ONLY FOR R3
def Greps_2R1R(x):
    Xreps = []
    R = (1/3) * np.array([[-1, 2, 2],
                          [2, -1, 2],
                          [2, 2, -1]])
    XY = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, -1]])
    XZ = np.array([[1, 0, 0],
                   [0, -1, 0],
                   [0, 0, 1]])
    # manually get all 8 group reps
    Xreps.append(np.dot(np.eye(3), x))
    Xreps.append(np.dot(R, x))
    Xreps.append(np.dot(XY, x))
    Xreps.append(np.dot(XZ, x))
    Xreps.append(np.dot(R@XY, x))
    Xreps.append(np.dot(R@XZ, x))
    Xreps.append(np.dot(XY@XZ, x))
    Xreps.append(np.dot(R@XY@XZ, x))
    return np.hstack(Xreps)

# two reflections, ONLY FOR R2
def Greps_2refl2d(x):
    X1 = np.array([[-1,0], [0,1]])
    X2 = np.array([[1,0], [0,-1]])
    X3 = np.array([[-1,0], [0,-1]])

    Xreps = [x, X1@x, X2@x, X3@x]
    return np.hstack(Xreps)

# rotation in each 2x2 subspace, ONLY FOR R3
def Greps_4rots(x):
    X1 = np.array([[-1,0, 0], [0, -1, 0], [0, 0, 1]])
    X2 = np.array([[1,0, 0], [0,-1, 0], [0, 0, -1]])
    X3 = np.array([[-1,0, 0], [0,1, 0], [0,0, -1]])

    Xreps = [x, X1@x, X2@x, X3@x]
    return np.hstack(Xreps)

'''
Combinatorial brute force computation of Karcher means given a dataset X (n by p) and Greps describing
    the orbit of a point x under G
'''
def Frechet(X, G):
    p, n = X.shape
    GX = G(X)
    k = GX.shape[1]//n

    # Split GX into a list of k blocks, each of shape (p, n)
    GX_blocks = np.hsplit(GX, k)

    # combinations of g elements
    combinations = product(range(k), repeat=n)
    means =  []
    distances = []
    frames = []

    # For each combination (g_1, ..., g_n)
    for combo in combinations:
        frame = np.zeros((p, n))
        for i, g in enumerate(combo):
            # pull submatrix corresponding to combo-indexed g-tuple on X
            # I think this can be improved by creating n! G-tensors to act on X rather than building section col by col
            frame[:, i] = GX_blocks[g][:, i]

        u = np.mean(frame, axis=1)
        means.append(u)
        dist = np.sum((frame.T-u)**2)
        distances.append(dist)
        frames.append(frame)

    j = np.argmin(distances)
    frechet = means[j]
    optimal_framing = frames[j]

    return frechet, optimal_framing

'''
Function to iteratively find the Frechet mean of a set of n data points acted on by a group G
Follows the "max-max" algorithm in
"Inconsistency of Template Estimation by Minimizing of the Variance/Pre-Variance in the Quotient Space"
Always converges in finitely many steps if G acts transitively on R^n
'''
def IterativeFrechet(X, G, u = None):
    p, n = X.shape
    GX = G(X)
    k = GX.shape[1]//n

    # initialize frechet mean guess
    if u is None:
        u = np.random.random(p)

    # split X into g orbits
    GX_blocks = np.hsplit(GX, k)

    niter = 0

    while True:
        frame = np.zeros([p,n])
        # (kxn) array where D[i,:] are the distances between each g_iX and u
        D = np.array([np.sum((GX_block.T - u)**2, axis=1) for GX_block in GX_blocks])  # Shape (k, n)
        # find optimal orbit point for each x_i
        closest_indices = np.argmin(D, axis=0)
        # create (pxn) array of X_i under optimal framing wrt u
        frame = np.array([GX_blocks[closest_indices[i]][:, i] for i in range(n)])
        u_new = np.mean(frame, axis = 0)

        if np.all(u == u_new):
            break

        niter += 1
        u = u_new

    return u, frame.T, niter

'''
Function to iteratively find the Frechet mean of a set of n data points acted on by a group G
alpha = 1 is identical to IterativeFrechet
Performs direct gradient descent on the data with fixed learning rate
May not converge even for transitive group actions for general alpha
'''
def FrechetGD(X, G, u, alpha = 1, niter = 100):
    p, n = X.shape
    GX = G(X)
    k = GX.shape[1]//n
    # split X into g orbits
    GX_blocks = np.hsplit(GX, k)

    for i in range(niter):
        frame = np.zeros([p,n])
        # (kxn) array where D[i,:] are the distances between each g_iX and u
        D = np.array([np.sum((GX_block.T - u)**2, axis=1) for GX_block in GX_blocks])  # Shape (k, n)
        # find optimal orbit point for each x_i
        closest_indices = np.argmin(D, axis=0)
        # create (pxn) array of X_i under optimal framing wrt u
        frame = np.array([GX_blocks[closest_indices[i]][:, i] for i in range(n)])
        iteration_mean = np.mean(frame, axis = 0)

        gradF = u - iteration_mean
        u = u - alpha*gradF

    return u, frame.T

def FrechetFunctional(x, X, G):
    p, n = X.shape
    GX = G(X)
    k = GX.shape[1]//n
    # split X into g orbits
    GX_blocks = np.hsplit(GX, k)

    frame = np.zeros([p,n])
    # (kxn) array where D[i,:] are the distances^2 between each g_iX and u
    D = np.array([np.sum((GX_block.T - x)**2, axis=1) for GX_block in GX_blocks])  # Shape (k, n)
    # find optimal orbit point for each x_i
    closest_indices = np.argmin(D, axis=0)
    # create (pxn) array of X_i under optimal framing wrt u
    frame = np.array([GX_blocks[closest_indices[i]][:, i] for i in range(n)])
    # sum of squared distances
    J = np.sum((frame - x)**2)

    return J

# '''
# Function to iteratively find the Frechet mean of a set of n data points acted on by a group G
# alpha = 1 is identical to IterativeFrechet
# Performs direct gradient descent on the data with fixed learning rate
# May not converge even for transitive group actions for general alpha
# '''
# def FrechetSGD(X, G, u, alpha = 1, TOL = 1e-6):
#     p, n = X.shape
#     GX = G(X)
#     k = GX.shape[1]//n
#     # init
#     u_new = np.random.random(p)
#     # split X into g orbits
#     GX_blocks = np.hsplit(GX, k)

#     while np.linalg.norm(u - u_new) > TOL:
#         frame = np.zeros([p,n])
#         u = u_new
#         # (kxn) array where D[i,:] are the distances between each g_iX and u
#         D = np.array([np.sum((GX_block.T - u)**2, axis=1) for GX_block in GX_blocks])  # Shape (k, n)
#         # find optimal orbit point for each x_i
#         closest_indices = np.argmin(D, axis=0)
#         # create (pxn) array of X_i under optimal framing wrt u
#         frame = np.array([GX_blocks[closest_indices[i]][:, i] for i in range(n)])
#         iteration_mean = np.mean(frame, axis = 0)

#         gradF = u - frame
#         u_new = u - alpha*gradF

#     return u, frame.T

# N points in R^p uniform in B(y,R)
def generate_points_within_ball(N, p, R, y):
    # Ensure y is an array and reshape it for broadcasting
    y = np.array(y).reshape(-1, 1)  # Reshape y to (p, 1)

    # Generate random directions
    directions = np.random.randn(p, N)
    directions /= np.linalg.norm(directions, axis=0)

    # Generate random distances from the center within [0, R]
    distances = np.random.rand(N) ** (1 / p) * R

    # Scale directions by distances and add the center point y
    points = y + directions * distances
    return points
