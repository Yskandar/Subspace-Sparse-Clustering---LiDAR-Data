# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab
import math
from Lidar_utils.subspace_clustering_2D import *


###
# This file uses all the previously defined functions to process real 2D LiDAR data.
# The LiDAR used is the RPLiDAR A2.
###


#####################################################################################################################################################
# TOOL FUNCTION
#####################################################################################################################################################

def get_points(file):
    """
    Returns an array containing the points of the snapshot
    
    Parameters
    ----------
    file : str
        Path of the point cloud
    """

    df = pd.read_csv(file)
    dict_ranges = {}
    list_values = []
    id_angle = 0
    for column in df.columns[1:]:
        values = list(df[column])
        values = [value for value in values if str(value) != 'inf']
        if values != []:
            dict_ranges[str(id_angle)] = np.mean(values)
            id_angle += 1


    # Let's now retrieve the coordinates of the points corresponding to these angles
    point_matrix = np.array([[0.0, 0.0]])

    for (angle, range) in dict_ranges.items():
        angle = math.radians(float(angle))
        X = - range * np.cos(angle)  # projecting the measurements in the plane
        Y = range * np.sin(angle)
        point = np.array([[X, Y]])
        point_matrix = np.vstack((point_matrix, point))

    point_matrix = point_matrix[1:]
    list_X = np.transpose(point_matrix[:, 0])
    list_Y = np.transpose(point_matrix[:, 1])

    return (list_X, list_Y), point_matrix



#####################################################################################################################################################
# MAIN CODE : PROCESSING LIDAR DATA
#####################################################################################################################################################

### First Part : Compute the lines matching the convex hull of our snapshots

# Let us pick the repository of the point cloud we seek to process
repository = './data/Precise_Traj'

# Let us pick the two LiDAR snapshots from which we seek to compute the rotation and translation
file = repository + '/point_cloud1.csv'
file_rotated = repository + '/point_cloud11.csv'

# Get the point clouds
datapoints, point_matrix = get_points(file)  
datapoints_rotated, point_matrix_rotated = get_points(file_rotated)


# Let us take a look at the two point clouds
plt.figure()
plt.scatter(datapoints[0], datapoints[1], label = 'recovered datapoints', color = 'blue')
plt.scatter(datapoints_rotated[0], datapoints_rotated[1], label = 'rotated recovered datapoints', color = 'red')
plt.show()

# Use the predefined functions to retrieve the line matching the subspaces of our point clouds
relevant_points, lines = retrieving_lines_and_points(point_matrix, num_clusters = 6, sigma = 0.05, K = 6)
relevant_points_rotated, lines_rotated = retrieving_lines_and_points(point_matrix_rotated, num_clusters = 6, sigma = 0.05, K = 6)

# Let us take a look at the recovered intersection points from the first snapshot
plt.figure()
plt.scatter(datapoints[0], datapoints[1], label = 'recovered datapoints', color = 'blue')
plt.scatter(relevant_points[0], relevant_points[1], label = 'rotated recovered datapoints', color = 'red')
plt.show()

# Let us take a look at the recovered intersection points from the second snapshot
plt.figure()
plt.scatter(datapoints_rotated[0], datapoints_rotated[1], label = 'recovered datapoints', color = 'blue')
plt.scatter(relevant_points_rotated[0], relevant_points_rotated[1], label = 'rotated recovered datapoints', color = 'red')
plt.show()


### Second Part : Compute the rotation between the matching lines

# Get the points belonging to the same lines
lines_recovered = get_point_tuples(relevant_points, lines)  # shape = (Nb_lines, 2, D)
lines_recovered_rotated = get_point_tuples(relevant_points_rotated, lines_rotated)  # shape = (Nb_lines, 2, D)


# Print the lines from the two snapshots and deduce the matching
rot_angles = []
translations = []
print(lines_recovered)
print(lines_recovered_rotated)
line1 = lines_recovered[4]  # Two matching lines is enough to compute the rotation
line1_rotated = lines_recovered_rotated[2]

# Compute the rotation and translation matrices
rotation, translation = get_transformation(line1, line1_rotated)  # rotation and translation matrices
rotation_angle = math.degrees(np.arctan2(rotation[1,0], rotation[0,0]))  # rotation angle
rot_angles.append(rotation_angle)
translations.append(translation)

