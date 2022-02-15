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
# This file regroups all the necessary functions to perform subspace sparse clustering on 2D point clouds obtained from LiDAR data.
# The theory behind this code originates from the following papers: 
# "Robust Subspace Clustering" By Mahdi Soltanolkotabi 1 , Ehsan Elhamifar and Emmanuel J. Candès.
# "Sparse Subspace Clustering: Algorithm, Theory, and Applications" By Ehsan Elhamifar, and René Vidal.
###


#####################################################################################################################################################
# MAIN FUNCTION : APPLYING SUBSPACE CLUSTERING 
#####################################################################################################################################################


def retrieving_lines_and_points(points, sigma = 0.05, K = 10):
    """
    Computes the line equations that match the different subspaces of the given convex hull.
    Uses the LASSO with data-driven regularization as a robustsparse regression strategy.
    Returns the equations of those lines, and the intersection points of convex hull.
    
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
    optimal_line_constructions : computes the line that fits a cluster the best
    all_line_intersection : computes all the intersection points off the given lines
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


    # Let us compute the estimated number of clusters (equal to the index of the largest difference between two successive eigenvalues)
    eigenvals_sorted_reversed = list(eigenvals_sorted)
    eigenvals_sorted_reversed.reverse()
    eigenvals_sorted_shifted = eigenvals_sorted_reversed[1:]
    delta = np.array(eigenvals_sorted_reversed[:-1]) - np.array(eigenvals_sorted_shifted)
    K_estimate = N - (np.argmax(delta) +1)  # we add one since the index starts at One in the paper, whereas it starts at 0 in Python
    K_estimate = K  # Doesn't seem too accurate, let us keep the given one


    # Getting the K smallest eigenvalues
    indices = []
    for i in range(0,K_estimate):
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
    cluster = run_k_means(proj_df, n_clusters=K_estimate) +1

    # Separating points depending on the cluster they belong to
    dict_points = separate_points(data, cluster, K_estimate)

    # Computing the lines of the 2D convex hull from the dict_points
    lines = optimal_line_construction(dict_points)

    # computing the intersection points between the different lines
    intersection_points = all_line_intersection(lines)
    int_points = np.transpose(np.reshape(intersection_points, (-1, np.shape(data)[0])))

    # Keeping only the intersection points that belong inside the convex hull
    remaining_points = check_point_using_origin(int_points, lines)


    return remaining_points, lines



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


def run_k_means(df, n_clusters):
    """
    Returns an array containing the index of the cluster each point belongs to
    
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
    D = np.shape(data)[0]  # Dimension of the points
    dict_points = dict()  # build the dictionary of points regrouped by clusters
    
    for j in range(1, K+1):
        dict_points[str(j)] = np.zeros((D,1))  # for now, we fill each cluster matrix with a zero vector
    
    for i in range(N):  # for each point of the data
        point = data[:,i]
        point = np.reshape(point,(D,1))
        num_cluster = cluster[i]  # we get the cluster it belongs to
        new_matrix = np.hstack((dict_points[str(num_cluster)],point))
        dict_points[str(num_cluster)] = new_matrix  # and add the point to the corresponding cluster matrix
    
    for j in range(1, K+1):  # we then remove the first zero vector of each cluster matrix
        dict_points[str(j)] = dict_points[str(j)][:,1:]
    
    return dict_points


def x_construction(points):
    """
    Returns a line equation matching the given cluster.
    The  regression is operated along the x axis.
    
    Parameters
    ----------
    points : array_like
        Dataset of points from a common cluster
    """
    points = np.transpose(points)
    N = np.shape(points)[0]  # number of points
    D = np.shape(points)[1]  # dimension

    # Reshaping the data
    One = np.ones(shape = (N,1))
    A = np.hstack((points[:,1:], One))
    B = points[:,0]
    B = np.reshape(B, (N, 1))

    # Fitting the line to the points using optimization
    X = cp.Variable(shape=(D,1))
    obj = cp.Minimize(power(cp.norm(A@X-B,2),2))
    prob = cp.Problem(obj)
    prob.solve(verbose=True)

    x_line = X.value
    a, b = -float(x_line[0]), -float(x_line[1])
    line = (1, a, b)

    return (line, np.linalg.norm(A@X.value - B, 2), X.value)  # (line equation, regression error)

def y_construction(points):
    """
    Returns a line equation matching the given cluster.
    The  regression is operated along the y axis.
    
    Parameters
    ----------
    points : array_like
        Dataset of points from a common cluster
    """
    points = np.transpose(points)
    N = np.shape(points)[0]  # number of points
    D = np.shape(points)[1]  # dimension

    # Reshaping the data
    One = np.ones(shape = (N,1))
    pts = np.reshape(points[:,0], (N,1))
    A = np.hstack((pts, One))
    B = points[:,1]
    B = np.reshape(B, (N, 1))

    # Fitting the line to the points using optimization
    X = cp.Variable(shape=(D,1))
    obj = cp.Minimize(power(cp.norm(A@X-B,2),2))
    prob = cp.Problem(obj)
    prob.solve(verbose=True)

    y_line = X.value
    a, b = -float(y_line[0]), -float(y_line[1])
    line = (a, 1, b)
    return (line, np.linalg.norm(A@X.value - B, 2), X.value)  # (line equation, regression error)


def optimal_line_construction(dict_points):
    """
    Returns the optimal line equations matching the different given clusters.
    The  regression is operated along the x or y axis depending on the lowest regression error.
    
    Parameters
    ----------
    dict_points : dict
        Dictionary regrouping points by cluster
    """
    list_lines = []
    for ind_cluster in dict_points.keys():
        points = dict_points[str(ind_cluster)]
        x_line = x_construction(points)
        y_line = y_construction(points)
        ls = [x_line, y_line]
        ls_1 = [x[1] for x in ls]  # list containing the loss associated to using each technique
        best_fit_index = np.argmin(ls_1)  # retrieve the line with the lowest loss
        best_fit = ls[best_fit_index][0]  # get the line equation
        list_lines.append(best_fit)
    return list_lines


def line_intersection(line1, line2):
    """
    Returns the intersection points of the two given lines
    
    Parameters
    ----------
    line1 : list
        list regrouping the parameters of the first line equation
    line2 : list
        list regrouping the parameters of the second line equation
    """
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    A = np.array([[a1, b1], [a2, b2]])
    B = np.transpose(np.array([[-c1, -c2]]))

    if np.linalg.matrix_rank(A) != 2:
        return "Two lines are parallel"
    else:
        X = np.linalg.inv(A)
        return np.linalg.inv(A) @ B

def all_line_intersection(list_lines):
    """
    Returns the intersection points of the given lines
    
    Parameters
    ----------
    list_lines : list
        list regrouping the parameters of all the lines matching the subspaces
    """
    intersection_points = []
    for i in range(len(list_lines)):
        for j in range(i+1, len(list_lines)):
            line1 = list_lines[i]
            line2 = list_lines[j]

            if type(line_intersection(line1, line2)) != str:
                intersection_points.append(line_intersection(line1, line2))
    return intersection_points



def plot_points(dict_points):
    """
    Plots the points from the same cluster in the same color.
    
    Parameters
    ----------
    dict_points : dict
        Dictionary regrouping points by cluster
    """
    fig = plt.figure()
    
    ax = fig.add_subplot()
    colors = ["pink", "green", "yellow", "red", "blue", "black", "brown", "orange", "purple", "grey"]
    for key in dict_points.keys():
        points_matrix = dict_points[key]
        ax.scatter(points_matrix[0,:], points_matrix[1,:], c = colors[int(key)-1])

    plt.show()



def plot_points_intersec(int_points):
    """
    Plots the intersection points.
    
    Parameters
    ----------
    int_points : array_like
        Array regrouping the intersection_points
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(int_points[0,:], int_points[1,:])
    ax.grid(True)
    plt.show()

def check_point_using_origin(points, lines):
    """
    Checks if points are inside the convex hull formed by the lines.
    For each line, we keep only the points on the good side of the line corresponding to the inside of the convex hull.
    
    Parameters
    ----------
    points : array_like
        array regrouping the intersection_points
    lines : list
        list regrouping the parameters of all the lines matching the subspaces
    """

    # Reshaping data
    list_points = [point for point in np.transpose(points)]
    lines = np.array(lines)
    origin = np.array([[0.0, 0.0]])

    # Going through every line equation and keeping only the points inside of the convex hull
    for line in lines:
        line_ab = line[0:2]
        constant = line[2:]
        constant = np.round(constant, 6)
        list_inter = [point for point in list_points if (np.round(float(line_ab @ np.transpose(point)), 6) == -float(constant)) 
                    or ((np.round(float(line_ab @ np.transpose(point)), 6) <= -float(constant))==(line_ab @ np.transpose(origin) <= -float(constant)))]
        list_points = list_inter
    return np.transpose(np.array(list_points))



def get_point_tuples(points, lines):  # get the points belonging to the same line
    """
    Returns the intersection points belonging to the same line.
    
    Parameters
    ----------
    points : array_like
        array regrouping the intersection_points which are inside the convex hull
    lines : list
        list regrouping the parameters of all the lines matching the subspaces
    """
    list_tuples = []
    for line in lines:
        line_recovered = []
        a, b, c = line
        line_recovered = [np.transpose(point) for point in np.transpose(points) if np.round(float(line[:2]@point), 6) == -np.round(float(line[-1]), 6)]
        if line_recovered != []:
            list_tuples.append(np.round(line_recovered, 6))
    return list_tuples

def get_transformation(line1, line2):
    """
    Computes the rotation and translation that turns line1 into line2.
    
    Parameters
    ----------
    line1 : list
        list regrouping the two interseciont points that define the first line
    line2 : list
        list regrouping the two interseciont points that define the first line
    """

    # Let's convert the points in 2D, since z = 0
    line1 = [point[:2] for point in line1]
    line2 = [point[:2] for point in line2]
    # substract the centroid from the data points to make the origin the new centroid
    centroid1 = sum(line1)/len(line1)
    centroid2 = sum(line2)/len(line2)
    line1 -= centroid1
    line2 -= centroid2

    H = np.transpose(line1)@line2  # compute the covariance matrix
    U, S, V = np.linalg.svd(H)  # SVD(H)
    d = np.sign(np.linalg.det(np.transpose(V)@np.transpose(U)))  
    # if d negative, it's a reflection thus we need to get a rotation
    sign_fix_matrix = np.array([[1, 0], [0, d]])
    R = np.round(np.transpose(V) @ sign_fix_matrix @ np.transpose(U), 6)
    T = centroid2 - R @ centroid1

    return R, T









#####################################################################################################################################################
# OUTDATED FUNCTIONS
#####################################################################################################################################################

def get_eigenvectors(points, sigma, K):

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

    indices = []
    for i in range(0,K):
        ind = []
        print(eigenvals_sorted_indices[i])
        ind.append(eigenvals_sorted_indices[i])
        indices.append(np.asarray(ind))

    indices = np.asarray(indices)
    zero_eigenvals_index = np.array(indices)
    eigenvals[zero_eigenvals_index]

    # Getting the eigenvectors associated to the smallest eigenvallues
    proj_df = pd.DataFrame(eigenvcts[:, zero_eigenvals_index.squeeze()])
    proj_df.columns = ['v_' + str(c) for c in proj_df.columns]
    proj_df.head()

    return proj_df

def apply_rotation(point_cloud, theta = np.pi/2):
    
    points = np.transpose(point_cloud)
    D, N = np.shape(points)
    rotation_angle = theta
    cos = np.cos(rotation_angle)
    sin = np.sin(rotation_angle)

    rot_matrix = np.zeros((D,D))
    rot_matrix[0,0],rot_matrix[1,1] = cos, cos
    rot_matrix[0,1],rot_matrix[1,0] = -sin, sin

    rotated_point_cloud = rot_matrix @ points

    return np.transpose(rotated_point_cloud)