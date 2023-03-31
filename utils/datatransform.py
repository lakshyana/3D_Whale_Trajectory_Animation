

"""
This script contains functions to load and transform the PRH data for use in animation visualizations.
"""

import scipy.io    # for reading .mat files
import numpy as np # for converting .mat files to numpy arrays
import io          # for reading bytes object output

def get_prh_data_from_mat(file):
    """
    Load the .mat file into a numpy array.
    :param file: The file object or filepath.
    :return: The numpy array.
    """
    # load the PRH .mat file into a numpy array
    data = scipy.io.loadmat(io.BytesIO(file)) # convert bytes string to file like object

    return data


#### Data Transformation Functions ####
def get_rotation_matrix(head, pitch, roll):
    """
    Get the rotation matrix from the head, pitch, and roll angles.
    :param head: The head angle in radians as a numpy array.
    :param pitch: The pitch angle in radians as a numpy array.
    :param roll: The roll angle in radians as a numpy array.
    :return: The rotation matrix.
    """

    # calculate the sin and cosine values for each angle
    sh = np.sin(head)  # sin of head
    ch = np.cos(head)  # cos of head
    sp = np.sin(pitch) # sin of pitch
    cp = np.cos(pitch) # cos of pitch
    sr = np.sin(roll)  # sin of roll
    cr = np.cos(roll)  # cos of roll

    # formula source: http://msl.cs.uiuc.edu/planning/node102.html

    # # construct the rotation matrix
    rotation_matrix = np.array([[ch*cp,  -ch*sp*sr - sh*cr,  -ch*sp*cr + sh*sr],
                                [sh*cp,  -sh*sp*sr + ch*cr,  -sh*sp*cr - ch*sr],
                                [ sp,     cp*sr,             cp*cr]])

    # rotation_matrix = np.array([[ch*cp,  ch*sp*sr - sh*cr,  ch*sp*cr + sh*sr],
    #                             [sh*cp,  sh*sp*sr + ch*cr,  sh*sp*cr - ch*sr],
    #                             [ -sp,     cp*sr,             cp*cr]])


    #  return matrix after reshape into format N x 3 x 3 matrices for multiple orientations
    #  (from original shape 3 x 3 x N x 1) or a single 3 x 3 matrix for a single orientation
    return np.rollaxis(rotation_matrix.squeeze(), -1, 0) if rotation_matrix.shape[-1] == 1 else rotation_matrix



def get_direction_vector(initial_direction, rotation_matrix):
    """
    Get the direction vector from the initial direction and rotation matrix.
    :param initial_direction: The initial direction vector.
    :param rotation_matrix: The rotation matrix.
    :return: The direction vector.
    """

    # get the direction vector
    direction_vector = np.matmul(initial_direction , rotation_matrix)

    return direction_vector


def get_speed_estimate(direction_vector, depth, sampling_rate, surface_speed= 1.5):
    """
    Get the whale speed estimate using the depth data and direction vector. If the head angle is large enough for a dive,
    the speed is estimated based on the depth data. Otherwise, the speed is assumed to be a static value (default is 1.5 m/s).

    :param direction_vector: The direction vector.
    :param depth: The depth in meters.
    :param sampling_rate: The sampling rate in frames per second (eg. 1 if data recorded every second).
    :return: The speed estimate in m/s.

    Note: Arc tangent of the z component of the direction vector over the magnitude of the x and y components
          This represents the angle between the direction vector and the vertical axis
    """

    ### Get the depth rate for all time steps
    depth_difference = np.abs(np.diff(depth, append=depth[-1]) ) # take the difference between consecutive depth readings

    # indices in the array
    indices = np.arange(0, len(depth_difference))

    # Get the angle of whale's motion with respect to the vertical axis (angle between the direction vector and the vertical axis)
    # to determine if the whale is diving or not

    # Compute the projection of the vector onto the xy plane
    xy_projection = np.sqrt(direction_vector[:, 0] ** 2 + direction_vector[:, 1] ** 2)

    # Compute the angle between the vector and the xy plane
    angle = abs(np.arctan2(direction_vector[:, 2], xy_projection))

    # Compute the speed estimate based on whether the whale is diving or not
    diving_mask =  angle > np.pi / 6  # diving if the angle is greater than 30 degrees
    diving_mask[-1] = False  # note: for the final index, the angle is not defined, so we set it to False

    # & (indices != len(depth_difference) - 1)  # and if the index is not the last index

    # Compute the speed estimate based on whether the whale is diving or not
    speed_estimate = np.where(diving_mask,
                                depth_difference / direction_vector[:, 2], # if diving, speed = depth difference / z component of direction vector
                                surface_speed / sampling_rate)             # if not diving, speed = surface speed / sampling rate


    return speed_estimate.squeeze()


def get_velocity_estimate(speed_estimate, direction_vector):
    """
    Get the velocity estimate from the speed estimate and direction vector.
    :param speed_estimate: The speed estimate.
    :param direction_vector: The direction vector.
    :return: The velocity estimate.
    """

    # get the velocity estimate
    velocity_estimate = speed_estimate[:, np.newaxis] * direction_vector # adding a new axis to speed estimate to make it 2D

    return velocity_estimate


def get_position_estimate(velocity_estimate, depth):
    """
    Get the position estimate from the velocity estimate and depth.
    :param velocity_estimate: The velocity estimate.
    :param depth: The depth in meters.
    :return: The position estimate.
    """

    # get the position estimate
    position_estimate = np.cumsum(velocity_estimate,
                                  axis=0 ) # cumulative sum of velocity estimate



    # add the depth to the position estimate
    position_estimate[:, 2] = -1 * depth

    # add a row of zeros to the beginning of the position estimate
    position_estimate = np.vstack((np.zeros(3), position_estimate))

    return position_estimate


def get_trajectory_estimate(head, pitch, roll, initial_direction, depth, sampling_rate):
    """
    Get the trajectory estimate from the head, pitch, roll, initial direction, depth, and sampling rate.
    :param head:    The head angle in radians.
    :param pitch:   The pitch angle in radians.
    :param roll:    The roll angle in radians.
    :param initial_direction:  The initial direction vector.
    :param depth:  The depth in meters.
    :param sampling_rate:  The sampling rate in frames per second (eg. 1 if data recorded every second).
    :return:  The trajectory estimates (position vector, speed).
    """

    # get the rotation matrix from the head, pitch, and roll angles
    rotation_matrix = get_rotation_matrix(head, pitch, roll)

    # get the direction vector of the whale at each time step
    direction_vector = get_direction_vector(initial_direction, rotation_matrix)

    # get the speed estimate of the whale at each time step
    speed_estimate = get_speed_estimate(direction_vector,
                                              depth,
                                              sampling_rate)

    # Get the velocity estimate of the whale at each time step
    velocity_estimate = get_velocity_estimate(speed_estimate, direction_vector)

    # Get the position estimate of the whale at each time step
    position_estimate = get_position_estimate(velocity_estimate, depth)

    return position_estimate, speed_estimate


