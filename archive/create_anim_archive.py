

"""
This script creates whale movement animations from the PRH files.
The animations are saved in the animations directory.
The PRH files are saved in the data directory.
The animations include the following:
    - A zoomed out view of the whale's movement.
    - A zoomed in view of the whale's movement.
    - A view of the whale's movement in the XY plane.
    - A view of the whale's orientation.
    - A view of the whale's depth.

The animations are created using the following steps:
    1. Load the PRH file.
    2. Get the orientation angles from the PRH file.
    3. Get the position data from the PRH file.
    4. Get the depth data from the PRH file.
    5. Create the zoomed out animation.
    6. Create the zoomed in animation.
    7. Create the XY animation.
    8. Create the orientation animation.
    9. Create the depth animation.

    Note: This script assumes that the user running this script has the AWS credentials set up on their machine to access
    the whaleproject bucket on AWS S3 using the boto3 library.
"""


## import helper functions from utils
import utils.dataload, utils.datatransform, utils.datavisualize

import numpy as np
import time

# TODO: remove later
import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation

if __name__ == '__main__':

    #### 1. LOAD DATA ####

    # params
    bucket_name  = 'whaleproject' # the bucket name
    directory    = 'PRH_all/'     # the directory to check for PRH files
    subdirectory = 'PRH/'        # the subdirectory to check for PRH files

    initial_direction = np.array([1, 0, 0])  # unit vector in the x direction

    # Connect to the AWS S3 client and resource objects.
    s3_client   = utils.dataload.get_s3_client()
    s3_resource = utils.dataload.get_s3_resource()

    # Get the result iterator
    results = utils.dataload.get_s3_data_iterator(s3_client, bucket_name, directory)


    # iterate through all the PRH files in the aws bucket
    for result in results:

        # iterate through all files in subdirectory
        for obj in result['Contents'][:1]:

            file_key = obj['Key'] # get the file key

            # Get the PRH file object
            file_object = utils.dataload.get_s3_file_object(s3_resource, file_key, bucket_name, subdirectory)

            # Load the PRH data from file object
            prh_data = utils.datatransform.get_prh_data_from_mat(file_object)

            # Get the orientation angles, depth, sampling rate values from the data
            head  = prh_data['head']          # head angle
            pitch = prh_data['pitch']         # pitch angle
            roll  = prh_data['roll']          # roll angle
            depth = prh_data['p'].squeeze()   # depth data of shape (n, 1) to (n,)
            sampling_rate = prh_data['fs'].squeeze()    # sampling rate



            ##### 2. COMPUTE WHALE TRAJECTORY ESTIMATES #####

            # Compute the rotation matrix from the head, pitch, and roll angles
            rotation_matrix = utils.datatransform.get_rotation_matrix(head, pitch, roll)

            ### Compute the direction vector of the whale at each time step
            direction_vector = utils.datatransform.get_direction_vector(initial_direction, rotation_matrix)

            ### Compute the speed estimate of the whale at each time step
            speed_estimate = utils.datatransform.get_speed_estimate(direction_vector,
                                                      depth,
                                                      sampling_rate)

            ### Compute the velocity estimate of the whale at each time step
            velocity_estimate = utils.datatransform.get_velocity_estimate(speed_estimate, direction_vector)

            ### Compute the position estimate of the whale at each time step
            position_estimate = utils.datatransform.get_position_estimate(velocity_estimate, depth)


            ###### 3. CREATE ANIMATIONS

            ### Get Zoomed out animation

            # get start time
            start_time = time.time()

            filename = file_key.split("/")[-1].split(".")[0] + "_zoomed_out"

            ### create the zoomed out animation

            # utils.datavisualize.create_zoomed_out_animation(position_estimate,
            #                                   sampling_rate,
            #                                   filename,
            #                                   step_size=50,
            #                                   figsize=(10, 10),
            #                                   title='Whale Trajectory',
            #                                   colormap='cividis'
            #                                   )

            # utils.datavisualize.create_trajectory_animation(position_estimate,
            #                                                 sampling_rate,
            #                                                 filename,
            #                                                 step_size=100,
            #                                                 figsize=(10, 10),
            #                                                 title='Whale Trajectory',
            #                                                 colormap='cividis')


            utils.datavisualize.create_zoomed_out_animation(position_estimate,
                                                            sampling_rate,
                                                            filename,
                                                            step_size=50,
                                                            figsize=(6, 6),
                                                            title='Whale Trajectory',
                                                            colormap='autumn'
                                                            )


            # get end time
            end_time = time.time()

            # get time elapsed to create the animation
            time_elapsed = end_time - start_time

            print(f"Time elapsed to create the zoomed out animation: {time_elapsed:.2f} seconds")









            # ####### for comparison with old method
            #
            # # create empty lists to store the whale trajectory
            # x = [0]
            # y = [0]
            # z = [0]
            # speed_list = []
            # whales = []
            #
            # if 'head' in prh_data.keys():  # check if the whale data contains head, pitch, and roll angles
            #
            #     for i in range(len(prh_data['head'])):  # len(prh_data['head']) is the number of time steps in the data
            #
            #         # create a rotation matrix based on the head, pitch, and roll angles
            #         rotation_matrix_1 = utils.datatransform.make_rotation_matrix(prh_data['head'][i][0], prh_data['pitch'][i][0],
            #                                                prh_data['roll'][i][0])
            #
            #
            #         # # print("angles")
            #         # # print(prh_data['head'][i][0], prh_data['pitch'][i][0], prh_data['roll'][i][0])
            #         # print("old rotation matrix")
            #         # print(rotation_matrix_1)
            #         # print(rotation_matrix_1.shape)
            #
            #         #
            #         # print("new rotation matrix")
            #         # print(rotation_matrix[i])
            #
            #         # print(rotation_matrix.reshape(3,3,-1)[:,:,i,])
            #
            #         # print("new rotation matrix with roll axis")
            #         # print(np.rollaxis(rotation_matrix.squeeze(), 2, 0)[i])
            #
            #
            #         rot_mat_single = utils.get_rotation_matrix(prh_data['head'][i][0],
            #                                          prh_data['pitch'][i][0],
            #                                          prh_data['roll'][i][0])
            #
            #
            #         # print("single rotation matrix")
            #         # print(rot_mat_single)
            #
            #         # use the rotation matrix to obtain the 3D orientation of the whale
            #         no_speed_whale = np.array([1, 0, 0])  # unit vector in the x direction
            #
            #         rotated_whale_vector = np.matmul(no_speed_whale, rotation_matrix_1)
            #
            #
            #         # print("Rotated Whale: ", rotated_whale_vector)
            #         #
            #         # print("orientation at i: ", direction_vector[i])
            #
            #
            #
            #
            #
            #         if i != len(prh_data['head']) - 1 and \
            #             abs(math.atan(rotated_whale_vector[2] / (math.sqrt(rotated_whale_vector[0] ** 2
            #                                    + rotated_whale_vector[1] ** 2)))) > math.pi / 6:
            #             #
            #             # angle = abs(math.atan(rotated_whale_vector[2]
            #             #                       / (math.sqrt(rotated_whale_vector[0] ** 2
            #             #                                    + rotated_whale_vector[1] ** 2))))
            #             # print("angle: ", angle)
            #
            #             speed_multiplier = abs(prh_data['p'][i][0] - prh_data['p'][i + 1][0]) / rotated_whale_vector[2]
            #
            #             # print("Speed Multiplier dive: ", speed_multiplier)
            #
            #         else:
            #             # Assumes the whale moves at 1.5 m/s at the surface if it is not diving. This value can be changed if incorrect.
            #             speed_multiplier = 1.5 / prh_data['fs'][0][0]
            #
            #             # print("Speed Multiplier surface: ", speed_multiplier)
            #
            #         speed_list.append(speed_multiplier)
            #
            #         whale = speed_multiplier * rotated_whale_vector
            #
            #         x.append(x[-1] + whale[0])
            #         y.append(y[-1] + whale[1])
            #         z.append(-1 * prh_data['p'][i][0])
            #
            #         whales.append(whale)
            #
            #
            #     print("Outputs: ")
            #
            #     print("PRH File: ")
            #     print(file_key)
            #     print(file_key.split("/")[-1])
            #     print("speed estimate: ", len(speed_estimate), len(speed_list))
            #
            #     print("first values: ", speed_list[:5], speed_estimate[:5])
            #
            #     print("last values: ", speed_list[-5:], speed_estimate[-5:])
            #
            #     print("whales: ", len(whales), len(velocity_estimate))
            #     print("first values: ", whales[:5], velocity_estimate[:5])
            #     print(whales[:5] == velocity_estimate[:5])
            #
            #     print("last values: ", whales[-5:],  velocity_estimate[-5:])
            #     print(whales[-5:] == velocity_estimate[-5:])
            #
            #     print("position estimate: ")
            #     print("Shape: ", position_estimate.shape, np.array(x).shape)
            #     print("first values: ", position_estimate[:5])
            #     print("x: ", x[:5])
            #     print("y: ", y[:5])
            #     print("z: ", z[:5])
            #     print("last values: ", position_estimate[-5:])
            #     print("x: ", x[-5:])
            #     print("y: ", y[-5:])
            #     print("z: ", z[-5:])

