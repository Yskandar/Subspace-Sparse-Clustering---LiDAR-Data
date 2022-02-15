# Importing required libraries
from cvxpy.constraints.constraint import Constraint
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cvxpy as cp
from cvxpy.atoms.elementwise.power import power
from scipy import linalg
import pandas as pd
from sklearn.cluster import KMeans

###
# This file regroups all the necessary functions to perform subspace sparse clustering on 3D point clouds obtained from LiDAR data.
# The theory behind this code originates from the following papers: 
# "Robust Subspace Clustering" By Mahdi Soltanolkotabi 1 , Ehsan Elhamifar and Emmanuel J. Candès.
# "Sparse Subspace Clustering: Algorithm, Theory, and Applications" By Ehsan Elhamifar, and René Vidal.
###


#####################################################################################################################################################
# MAIN FUNCTION : APPLYING SUBSPACE CLUSTERING 
#####################################################################################################################################################

def retrieving_planes_and_points(points, sigma = 0.05, K = 10):
    """
    Computes the plane equations that match the different subspaces of the given convex hull.
    Uses the LASSO with data-driven regularization as a robustsparse regression strategy.
    Returns the equations of those planes, and the intersection points of convex hull with an horizontal slicing plane.
    
    Parameters
    ----------
    points : array_like
        Given point cloud, of shape (nb_points, dimension).
    sigma : float or None, optional
        Estimation of the noise level. See "Robust Subspace Clustering", section 2.4
    K : int or None, optional
        Number of subspaces

    See Also
    --------
    find_sparse_sol_robust : Implements two successive LASSO algorithms to perform subspace clustering
    run_k_means : a simple k-means clustering algorithm
    separate_points : separates points depending on the cluster they belong to
    optimal_plane_constructions : computes the plane that fits a cluster the best
    all_plane_intersection : computes all the intersection points off the given lines
    check_point_using_origin : given a convex hull and a point cloud, only retains points inside the convex hull
    
    """
    
    data = np.transpose(points)  # shape = (Dimension, nb_points)
    N = np.shape(data)[1]  # Number of points
    D = np.shape(data)[0]  # Dimension

    # We now apply this function to each datapoint to construct the global sparse representation matrix C
    C = np.concatenate((np.zeros((1,1)),find_sparse_sol_robust(data,0,sigma)),axis=0)  # filling the first row of the matrix
    for i in range(1,N):
        ci = find_sparse_sol_robust(data,i,sigma)
        zero_element = np.zeros((1,1))  # making sure Diag(C) = 0, as cii must not be a combination of itself
        cif = np.concatenate((ci[0:i,:],zero_element,ci[i:N,:]),axis=0)
        C = np.concatenate((C,cif),axis=1)
    
    # We must now normalize the columns of C
    for i in range(N):
        column = C[:,i]
        inf_norm = max(np.abs(column))  # computing the infinite norm
        C[:,i] = column/inf_norm


    # Calculating the weights of the similarity graph
    W = np.abs(C) + np.transpose(np.abs(C))


    # Check sparsity by counting the number of zeros
    cz = 0
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i,j] < 1e-5 and W[i,j] > -1e-5:
                cz += 1
    print(cz)
    print("percentage of zeros : {}%".format(100*cz/(N**2)))

    # Computing the degree matrix
    D = np.zeros((N,N))
    sum_list=[]
    for i in range(N):
        csum = np.sum(W[:,i],axis=0)
        sum_list.append(csum)
    D = np.diag(sum_list)
    print(D)
    
    # Laplacian Matrix &  Symmetric Normalized Laplacian Matrix
    L = D - W
    LN =  np.diag(np.divide(1, np.sqrt(np.sum(D, axis=0) + np.finfo(float).eps)))@ L @  np.diag(np.divide(1, np.sqrt(np.sum(D, axis=0) + np.finfo(float).eps)))
    print(LN.shape)



    # Computing the eigenvalues/eigenvectors of the Laplacian Matrix
    eigenvals, eigenvcts = linalg.eig(LN)
    eigenvals = np.real(eigenvals)
    eigenvcts = np.real(eigenvcts)
    eig = eigenvals.reshape((N,1))

    # sorting the eigenvalues
    eigenvals_sorted_indices = np.argsort(eigenvals)
    eigenvals_sorted = eigenvals[eigenvals_sorted_indices]

    # Getting the K smallest eigenvalues
    indices = []
    for i in range(0,K):
        ind = []
        print(eigenvals_sorted_indices[i])
        ind.append(eigenvals_sorted_indices[i])
        indices.append(np.asarray(ind))
    indices = np.asarray(indices)
    zero_eigenvals_index = np.array(indices)
    eigenvals[zero_eigenvals_index]

    # Getting the eigenvectors associated to the smallest eigenvalues
    proj_df = pd.DataFrame(eigenvcts[:, zero_eigenvals_index.squeeze()])
    proj_df.columns = ['v_' + str(c) for c in proj_df.columns]
    proj_df.head()

    # Performing K-means clustering on the eigenvectors
    cluster = run_k_means(proj_df, n_clusters=K) +1

    # Separating points depending on the cluster they belong to
    dict_points = separate_points(data, cluster, K)

    # Computing the planes of the 3D convex hull from the dict_points
    planes = optimal_plane_construction(dict_points)

    # Slicing the obtained planes with the horizontal plane : 
    # Computing the intersection points between the convex hull and the slicing plane
    slicing_plane = [0, 0, 1, 0]
    intersection_points = all_plane_intersection(planes, slicing_plane)
    int_points = np.transpose(np.reshape(intersection_points, (-1, 3)))

    # Keeping only the intersection points that belong inside the convex hull
    remaining_points = check_point_using_origin(int_points, planes)


    return remaining_points, planes



#####################################################################################################################################################
# TOOL FUNCTIONS
#####################################################################################################################################################


def find_sparse_sol_robust(Y,i,sigma):
    """
    Returns a sparse representation the given datapoint
    Uses the LASSO with data-driven regularization to express the datapoint solely in function of points in the same subspace
    
    Parameters
    ----------
    Y : array_like
        Complete dataset
    i : int
        Index of the datapoint currently processed
    sigma : float
        Standard deviation of the noise applied to the data    
    """
    
    N = np.shape(Y)[1]  # Number of points
    D = np.shape(Y)[0]  # Dimension
    
    # Making sure we do not express the data point as a linear combination of itself
    if i == 0:
        Ybari = Y[:,1:N]  # removing data point 0
    if i == N-1:
        Ybari = Y[:,0:N-1]  # removing data point N
    if i!=0 and i!=N-1:
        Ybari = np.concatenate((Y[:,0:i],Y[:,i+1:N]),axis=1)  # removing data point i
    yi = Y[:,i].reshape(D,1)  # reshaping in case it's not already done
    
    
    # First step : solve a hard constrained version of the LASSO, to estimate the dimension of the subspace
    Tau = 2*sigma
    ci = cp.Variable(shape=(N-1,1))
    constraint = [cp.norm(yi-Ybari@ci,2)<=Tau]
    obj = cp.Minimize(cp.norm(ci,1))
    prob = cp.Problem(obj, constraint)
    prob.solve(verbose=True)

    lbda = 0.25/np.linalg.norm(ci.value, ord = 1)  # This result is chosen empirically in the paper "Robust subspace Clustering"

    # Second step : solve the optimization problem to minimize false discoveries
    cj = cp.Variable(shape=(N-1,1))
    constraint = [cp.sum(cj)==1]
    obj = cp.Minimize(0.5 * power(cp.norm(yi-Ybari@cj,2),2) + lbda*cp.norm(cj,1))
    prob = cp.Problem(obj, constraint)
    prob.solve(verbose=True)
    
    return cj.value

def plot_points_3d(int_points):
    """
    Plots the points resulting from the intersection of the convex hull and the slicing plane.
    
    Parameters
    ----------
    int_points : array_like
        Array regrouping the intersection_points
    """
    fig = plt.figure(np.random.randint(1, 500))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(int_points[0,:], int_points[1,:], int_points[2,:])
    plt.show()



def run_k_means(df, n_clusters):
    """
    Returns an array contining the index of the cluster each point belongs to
    
    Parameters
    ----------
    df : dataframe
        Dataset of points
    n_clusters : int
        Number of clusters
    """
    k_means = KMeans(random_state=25, n_clusters=n_clusters)
    k_means.fit(df)
    cluster = k_means.predict(df)
    return cluster

def separate_points(data, cluster, K):
    """
    Returns a dictionary containing points regrouped by clusters.
    
    Parameters
    ----------
    data : array_like
        Dataset of points
    cluster : array_like
        Array containing the index of the cluster each point belongs to
    K : int
        Number of clusters
    """

    N = np.shape(data)[1]  # get the number of points in the dataset
    dict_points = dict()  # build the dictionary of points regrouped by clusters
    
    for j in range(1, K+1):
        dict_points[str(j)] = np.zeros((3,1))  # for now, we fill each cluster matrix with a zero vector
    
    for i in range(N):  # for each point of the data
        point = data[:,i]
        point = np.reshape(point,(3,1))
        num_cluster = cluster[i]  # we get the cluster it belongs to
        new_matrix = np.hstack((dict_points[str(num_cluster)],point))
        dict_points[str(num_cluster)] = new_matrix  # and add the point to the corresponding cluster matrix
    
    for j in range(1, K+1):  # we then remove the first zero vector of each cluster matrix
        dict_points[str(j)] = dict_points[str(j)][:,1:]
    
    return dict_points

def plot_points(dict_points):
    """
    Plots the points from the same cluster in the same color.
    
    Parameters
    ----------
    dict_points : dict
        Dictionary regrouping points by cluster
    """
    fig = plt.figure()
    
    ax = fig.add_subplot(projection='3d')
    colors = ["pink", "green", "yellow", "red", "blue", "black", "brown", "orange", "purple", "grey"]
    for key in dict_points.keys():
        points_matrix = dict_points[key]
        ax.scatter(points_matrix[0,:], points_matrix[1,:], points_matrix[2,:], c = colors[int(key)-1])

    plt.show()




def z_construction(points):
    """
    Returns a plane equation matching the given cluster.
    The  regression is operated along the z axis.
    
    Parameters
    ----------
    points : array_like
        Dataset of points from a common cluster
    """
    points = np.transpose(points)
    nb_points = np.shape(points)[0] # number of points
    D = np.shape(points)[1]  # dimension

    # Reshaping the data
    One = np.ones(shape = (nb_points,1))
    A = np.hstack((points[:,:2], One))
    B = points[:,-1]
    B = np.reshape(B, (nb_points, 1))

    # Fitting the plane to the points using optimization
    X = cp.Variable(shape=(3,1))
    obj = cp.Minimize(power(cp.norm(A@X-B,2),2))
    prob = cp.Problem(obj)
    prob.solve(verbose=True)

    z_plane = X.value
    a, b, c = float(z_plane[0]), float(z_plane[1]), float(z_plane[2])
    point  = np.array([0.0, 0.0, c])
    normal = np.array(cross([1,0,a], [0,1,b]))
    d = -point.dot(normal)

    plane = [normal[0], normal[1], normal[2], d]  # ax + by + cz + d = 0

    return (plane, np.linalg.norm(A@X.value - B, 2), X.value)  # (plane equation, regression error)


def y_construction(points):
    """
    Returns a plane equation matching the given cluster.
    The  regression is operated along the y axis.
    
    Parameters
    ----------
    points : array_like
        Dataset of points from a common cluster
    """

    points = np.transpose(points)
    nb_points = np.shape(points)[0]  # number of points
    D = np.shape(points)[1]  # dimension

    # Reshaping the data
    One = np.ones(shape = (nb_points,1))
    a = np.reshape(points[:,0], (nb_points, 1))
    b = np.reshape(points[:,2], (nb_points, 1))
    A = np.hstack((a, b, One))
    B = points[:,1]
    B = np.reshape(B, (nb_points, 1))

    # Fitting the plane to the points using optimization
    X = cp.Variable(shape=(3,1))
    obj = cp.Minimize(power(cp.norm(A@X-B,2),2))
    prob = cp.Problem(obj)
    prob.solve(verbose=True)

    y_plane = X.value
    a, b, c = float(y_plane[0]), float(y_plane[1]), float(y_plane[2])
    point  = np.array([0.0, c, 0.0])
    normal = np.array(cross([1,a,0], [0,b,1]))
    d = -point.dot(normal)

    plane = [normal[0], normal[1], normal[2], d]  # ax + by + cz + d = 0

    return (plane, np.linalg.norm(A@X.value - B, 2), X.value)  # (plane equation, regression error)


def x_construction(points):
    """
    Returns a plane equation matching the given cluster.
    The  regression is operated along the x axis.
    
    Parameters
    ----------
    points : array_like
        Dataset of points from a common cluster
    """
    points = np.transpose(points)
    nb_points = np.shape(points)[0]  # number of points
    D = np.shape(points)[1]  # dimension

    # Reshaping the data
    One = np.ones(shape = (nb_points,1))
    A = np.hstack((points[:,1:], One))
    B = points[:,0]
    B = np.reshape(B, (nb_points, 1))

    # Fitting the plane to the points using optimization
    X = cp.Variable(shape=(3,1))
    obj = cp.Minimize(power(cp.norm(A@X-B,2),2))
    prob = cp.Problem(obj)
    prob.solve(verbose=True)

    x_plane = X.value
    a, b, c = float(x_plane[0]), float(x_plane[1]), float(x_plane[2])
    point  = np.array([c, 0.0, 0.0])
    normal = np.array(cross([a,1,0], [b,0,1]))
    d = -point.dot(normal)
    plane = [normal[0], normal[1], normal[2], d]

    return (plane, np.linalg.norm(A@X.value - B, 2), X.value)  # (plane equation, regression error)


def optimal_plane_construction(dict_points):
    """
    Returns the optimal plane equations matching the different given clusters.
    The  regression is operated along the x, y or z axis depending on the lowest regression error.
    
    Parameters
    ----------
    dict_points : dict
        Dictionary regrouping points by cluster
    """
    list_planes = []
    for ind_cluster in dict_points.keys():
        points = dict_points[str(ind_cluster)]
        x_plane = x_construction(points)
        y_plane = y_construction(points)
        z_plane = z_construction(points)
        ls = [x_plane, y_plane, z_plane]
        ls_1 = [x[1] for x in ls]  # list containing the loss associated to using each technique
        best_fit_index = np.argmin(ls_1)  # retrieve the plane with the lowest loss
        best_fit = ls[best_fit_index][0]  # get the plane equation
        list_planes.append(best_fit)
    return list_planes

def plane_intersection(plane_1, plane_2, plane_3):
    """
    Returns the intersection points of the three given planes
    
    Parameters
    ----------
    plane_1 : list
        list regrouping the parameters of the first plane equation
    plane_2 : list
        list regrouping the parameters of the second plane equation
    plane_3 : list
        list regrouping the parameters of the third plane equation
    """
    a1, b1, c1, d1 = plane_1
    a2, b2, c2, d2 = plane_2
    a3, b3, c3, d3 = plane_3
    A = np.array([[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]])
    d = np.transpose(np.array([[-d1, -d2, -d3]]))
    if np.linalg.matrix_rank(A) != 3:
        return "Two or more planes are parallel"
    else:
        X = np.linalg.inv(A)
        return np.linalg.inv(A) @ d

def all_plane_intersection(list_planes, slicing_plane):
    """
    Returns the intersection points of the convex hull with the slicing plane.
    
    Parameters
    ----------
    list_planes : list
        list regrouping the parameters of all the planes matching the subspaces
    slicing_plane : list
        list regrouping the parameters of the slicing plane
    """
    intersection_points = []
    for i in range(len(list_planes)):
        for j in range(i+1, len(list_planes)):
            plane1 = list_planes[i]
            plane2 = list_planes[j]
            if type(plane_intersection(plane1, plane2, slicing_plane)) != str:
                intersection_points.append(plane_intersection(plane1, plane2, slicing_plane))
    
    return intersection_points

def plot_points_intersec(int_points):
    """
    Plots the intersection points.
    
    Parameters
    ----------
    int_points : array_like
        Array regrouping the intersection_points
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(int_points[0,:], int_points[1,:], int_points[2,:])
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim( -1,1)
    plt.show()

def check_point_using_origin(points, planes):
    """
    Checks if points are inside the convex hull formed by the planes.
    For each plane, we keep only the points on the good side of the plane corresponding to the inside of the convex hull.
    
    Parameters
    ----------
    points : array_like
        array regrouping the intersection_points
    planes : list
        list regrouping the parameters of all the planes matching the subspaces
    """
    # Reshaping data
    list_points = [point for point in np.transpose(points)]
    planes = np.array(planes)
    origin = np.array([[0.0, 0.0, 0.0]])

    # Going through every line equation and keeping only the points inside of the convex hull
    for plane in planes:
        plane_abc = plane[0:3]
        plane_abc = np.round(plane_abc, 6)
        constant = plane[3:]
        constant = np.round(constant, 6)
        list_inter = [point for point in list_points if (np.round(float(plane_abc @ np.transpose(point)), 6) == -float(constant)) 
                    or ((np.round(float(plane_abc @ np.transpose(point)), 6) <= -float(constant))==(plane_abc @ np.transpose(origin) <= -float(constant)))]
        list_points = list_inter
    
    return np.transpose(np.array(list_points))



#####################################################################################################################################################
# OUTDATED FUNCTIONS
#####################################################################################################################################################


def find_sparse_sol(Y,i,N):
    """
    param Y : complete dataset
    param i : data point which is currently analyzed
    param N : number of points
    """

    N = np.shape(Y)[1]  # Number of points
    D = np.shape(Y)[0]  # Dimension
    # Making sure we do not express the data point as a linear combination of itself
    if i == 0:
        Ybari = Y[:,1:N]  # removing data point 0
    if i == N-1:
        Ybari = Y[:,0:N-1]  # removing data point N
    if i!=0 and i!=N-1:
        Ybari = np.concatenate((Y[:,0:i],Y[:,i+1:N]),axis=1)  # removing data point i
    yi = Y[:,i].reshape(D,1)  # reshaping in case it's not already done
    
    # this ci will contain the solution of the l1 optimisation problem:  
    # min (||yi - Ybari*ci||F)^2 + lambda*||ci||1   st. sum(ci) = 1
    
    ci = cp.Variable(shape=(N-1,1))
    constraint = [cp.sum(ci)==1]
    obj = cp.Minimize(power(cp.norm(yi-Ybari@ci,2),2) + 100000*cp.norm(ci,1)) #lambda = 100
    prob = cp.Problem(obj, constraint)
    prob.solve(verbose=True)
    return ci.value

def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]

def apply_rotation(point_cloud, theta = np.pi/2):
    
    points = np.transpose(point_cloud)
    
    rotation_angle = theta
    cos = np.cos(rotation_angle)
    sin = np.sin(rotation_angle)

    rot_matrix = np.zeros((3,3))
    rot_matrix[0,0],rot_matrix[1,1], rot_matrix[2,2] = cos, cos, 1
    rot_matrix[0,1],rot_matrix[1,0] = -sin, sin

    rotated_point_cloud = rot_matrix @ points

    return np.transpose(rotated_point_cloud)