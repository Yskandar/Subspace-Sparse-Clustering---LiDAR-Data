# Project Description

The purpose of this project is to provide an estimation of the position of a robot, using the LiDAR measurements performed by the robot.
To do so, I apply robust spectral clustering techniques derived from papers referenced below.
After experimenting the algorithm on simulation data, I conducted several experiments to gather LiDAR data of a convex hull and processed them.
The LiDAR used to perform the experiments is the RPLiDAR A2.
This project is funded by the [Cyber Physical Systems Laboratory, UCLA](https://www.cyphylab.ee.ucla.edu/)


# Pathway

To compute the position of the robot with respect to its origin stance, I first apply robust subspace clustering to obtain clusters of cleaned datapoints belonging to the same subspace. Then, I use regression techniques to compute the best fitting lines to my clusters.
Eventually, I compute the rotation and translation between the lines of the first convex hull and the lines of the second convex hull.
This project relies on the assumption that the environment of the robot is mostly convex.


# References

1. Ehsan Elhamifar and René Vidal: Sparse Subspace Clustering: Algorithm, Theory and Aplications. In IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 35, NO. 11, NOVEMBER 2013.

2. Mahdi Soltanolkotabi 1 , Ehsan Elhamifar and Emmanuel J. Candès: Robust Subspace Clustering. In arXiv:1301.2603 [cs.LG] 23 May 2014