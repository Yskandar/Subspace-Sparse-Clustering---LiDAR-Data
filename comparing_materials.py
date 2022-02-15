# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab


###
# This file regroups all the necessary functions I used to assess the performances and reliability of the LiDAR module.
# I compare the performance of the LiDAR RP2 depending on the reflecting material, the distance of the LiDAR to the reflection material, and the lighting conditions.
# The resulting graphs are stored in /comparing_materials 
###


#####################################################################################################################################################
# TOOL FUNCTIONS
#####################################################################################################################################################

def get_data(distance, material, angle = 0):

    repository = './data/{}/'.format(material)
    file = repository + 'datafile_{}_{}.csv'.format(material, distance)

    df = pd.read_csv(file)
    angle_values = list(df.columns)[1:]
    angle_value = angle_values[angle]
    raw_data = list(df[angle_value])
    data = [point for point in raw_data if str(point) != 'inf']

    return data


def count_failed_measurements(distance, material, angle = 0):

    repository = './data/{}/'.format(material)
    file = repository + 'datafile_{}_{}.csv'.format(material, distance)

    df = pd.read_csv(file)
    angle_values = list(df.columns)[1:]
    angle_value = angle_values[angle]
    raw_data = list(df[angle_value])
    failed_measurements = [point for point in raw_data if str(point) == 'inf']
    return len(failed_measurements)/len(raw_data)


def get_scores(distance, material):
    repository = './data/{}/'.format(material)
    file = repository + 'datafile_{}_{}.csv'.format(material, distance)

    df = pd.read_csv(file)
    angle_values = list(df.columns)[1:]
    angle_value = angle_values[0]

    raw_data = list(df[angle_value])
    data = [point for point in raw_data if str(point) != 'inf']
    time = [i for i in range(len(data))]
    scores = (np.var(data), np.std(data))

    return scores


def plot_measurements(data, material):

    time = [i for i in range(len(data))]
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot()
    ax.plot(time, data, label = material)
    ax.set_xlabel('time')
    ax.set_ylabel('measurement')
    ax.grid()
    ax.legend()
    plt.show()



def comparing_std_materials(distances, materials, id_distance):
    distance = distances[id_distance]
    std_dev = []
    for material in materials:
        data = get_data(distance, material)
        plot_measurements(data, material)
        std_dev.append(get_scores(distance, material)[1])
    

    x = [i for i in range(len(std_dev))]
    height = [value*100 for value in std_dev]
    width = 0.1
    BarName = materials
    #ax = fig.add_subplot()
    plt.figure()

    plt.bar(x, height, width, color=('green', 'red', 'grey', 'brown', 'blue', 'black') )
    plt.scatter([i+width/2.0 for i in x],height,color='k',s=40)

    plt.xlim(-0.2,3.2)
    plt.grid()

    plt.ylabel('Standard Deviation (in centimeters)')
    plt.title('Standard Deviation across different materials, distance = {}'.format(distance))

    pylab.xticks(x, BarName, rotation=40)

    plt.savefig('Standard Deviation across 3 different materials, distance = {}'.format(distance))
    plt.show()






def comparing_std_depth(distances, materials, id_material):
    material = materials[id_material]
    std_dev = []
    for distance in distances:
        data = get_data(distance, material)
        plot_measurements(data, material)
        std_dev.append(get_scores(distance, material)[1])
    

    x = [i for i in range(len(std_dev))]
    height = [value*100 for value in std_dev]
    width = 0.1
    BarName = distances
    #ax = fig.add_subplot()
    plt.figure()

    plt.bar(x, height, width, color=('green', 'red', 'grey', 'brown', 'blue', 'black') )
    plt.scatter([i+width/2.0 for i in x],height,color='k',s=40)

    plt.xlim(-0.2,7.2)
    plt.grid()

    plt.ylabel('Standard Deviation (in centimeters)')
    plt.title('Standard Deviation across different distances, material = {}'.format(material))

    pylab.xticks(x, BarName, rotation=40)

    plt.savefig('Standard Deviation across different distances, material = {}'.format(material))
    plt.show()






def comparing_std_light_conditions(light_conditions, id_material):
    material = materials[id_material]
    std_dev = []
    for light_condition in light_conditions:
        data = get_data(light_condition, material)
        plot_measurements(data, material)
        std_dev.append(get_scores(light_condition, material)[1])
    

    x = [i for i in range(len(std_dev))]
    height = [value*100 for value in std_dev]
    width = 0.1
    BarName = light_conditions
    #ax = fig.add_subplot()
    plt.figure()

    plt.bar(x, height, width, color=('green', 'red', 'grey', 'brown', 'blue', 'black') )
    plt.scatter([i+width/2.0 for i in x],height,color='k',s=40)

    plt.xlim(-0.2,3.2)
    plt.grid()

    plt.ylabel('Standard Deviation (in centimeters)')
    plt.title('Standard Deviation across different light conditions, material = {}'.format(material))

    pylab.xticks(x, BarName, rotation=40)

    plt.savefig('Standard Deviation across different light conditions, material = {}'.format(material))
    plt.show()




def comparing_failed_measurements_materials(distances, materials, id_distance):
    distance = distances[id_distance]
    nb_failed_measurements = []
    for material in materials:
        data = get_data(distance, material)
        nb_failed_measurements.append(count_failed_measurements(distance, material))
    

    x = [i for i in range(len(nb_failed_measurements))]
    height = [value*100 for value in nb_failed_measurements]
    width = 0.1
    BarName = materials
    #ax = fig.add_subplot()
    plt.figure()

    plt.bar(x, height, width, color=('green', 'red', 'grey', 'brown', 'blue', 'black') )
    plt.scatter([i+width/2.0 for i in x],height,color='k',s=40)

    plt.xlim(-0.2,3.2)
    plt.grid()

    plt.ylabel('Failed Measurements (percentage)')
    plt.title('Failed Measurements across different materials, distance = {}'.format(distance))

    pylab.xticks(x, BarName, rotation=40)

    plt.savefig('Failed Measurements across different materials, distance = {}'.format(distance))
    plt.show()





def comparing_failed_measurements_distances(distances, materials, id_material):
    material = materials[id_material]
    nb_failed_measurements = []
    for distance in distances:
        data = get_data(distance, material)
        nb_failed_measurements.append(count_failed_measurements(distance, material))
    

    x = [i for i in range(len(nb_failed_measurements))]
    height = [value*100 for value in nb_failed_measurements]
    width = 0.1
    BarName = distances
    #ax = fig.add_subplot()
    plt.figure()

    plt.bar(x, height, width, color=('green', 'red', 'grey', 'brown', 'blue', 'black') )
    plt.scatter([i+width/2.0 for i in x],height,color='k',s=40)

    plt.xlim(-0.2,7.2)
    plt.grid()

    plt.ylabel('Failed Measurements (percentage)')
    plt.title('Standard Deviation across different materials, material = {}'.format(material))

    pylab.xticks(x, BarName, rotation=40)

    plt.savefig('Failed Measurements across different distances, material = {}'.format(material))
    plt.show()



#####################################################################################################################################################
# MAIN CODE : ASSESSING PERFORMANCES
#####################################################################################################################################################


# For a fixed distance, compare the different materials (STD of the measurements)
distances = ['0m50', '1m', '1m50', '2m', '2m50', '3m',  '3m50']
materials = ['green_wall', 'white_wall', 'grey_metal', 'wood']
id_distance = 5
comparing_std_materials(distances, materials, id_distance)


# For a fixed material, compare the different distances (STD of the measurements)
distances = ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m']
materials = ['green_wall', 'white_wall', 'grey_metal', 'wood']
id_material = 3
comparing_std_depth(distances, materials, id_material)

# For a fixed material, compare the different lighting conditions
light_conditions = ['nolight', 'light_1', 'light_2', 'light_3']
id_material = 1
comparing_std_light_conditions(light_conditions, 1)

# For a fixed distance, compare the different materials (Number of failed measurements)
distances = ['0m50', '1m', '1m50', '2m', '2m50', '3m',  '3m50']
materials = ['green_wall', 'white_wall', 'grey_metal', 'wood']
id_distance = 3
comparing_failed_measurements_materials(distances, materials, 3)

# For a fixed material, compare the different distances (Number of failed measurements)
distances = ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m']
materials = ['green_wall', 'white_wall', 'grey_metal', 'wood']
id_material = 3
comparing_failed_measurements_distances(distances, materials, 3)