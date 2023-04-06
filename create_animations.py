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


## TODO remove old rotation matrix later
import math
import numpy as np
import subprocess
import os

def make_rotation_matrix(head, pitch, roll):
    '''
    Create a rotation matrix based on head, pitch, and roll angles that can be used to obtain the orientation of the whale in 3 dimensions.
    '''
    return np.array([
        [math.cos(head)*math.cos(pitch), -1*math.cos(head)*math.sin(pitch)*math.sin(roll) - math.sin(head)*math.cos(roll), -1*math.cos(head)*math.sin(pitch)*math.cos(roll) + math.sin(head)*math.sin(roll)],
        [math.sin(head)*math.cos(pitch), -1*math.sin(head)*math.sin(pitch)*math.sin(roll) + math.cos(head)*math.cos(roll), -1*math.sin(head)*math.sin(pitch)*math.cos(roll) - math.cos(head)*math.sin(roll)],
        [math.sin(pitch), math.cos(pitch)*math.sin(roll), math.cos(pitch)*math.cos(roll)]])




if __name__ == '__main__':

    ####### 1. LOAD DATA FROM AWS S3 Buckets #######

    # PARAMS: s3 bucket name, directory, subdirectory
    bucket_name  = 'whaleproject' # the bucket name
    directory    = 'PRH_all/'     # the directory to check for PRH files
    subdirectory = 'PRH/'        # the subdirectory to check for PRH files
    whale_object_path = 'assets/whale.obj'

    ##### Other params
    initial_direction = np.array([1, 0, 0])  # unit vector in the x direction

    # Connect to the AWS S3 client and resource objects.
    s3_client   = utils.dataload.get_s3_client()
    s3_resource = utils.dataload.get_s3_resource()


    # Get the result iterator to iterate through all the pages of results
    results = utils.dataload.get_s3_data_iterator(s3_client, bucket_name, directory)


    # Iterate through all the pages of results from the S3 bucket
    for result in results:

        # Iterate through all files in the current page of results
        for obj in result['Contents'][5:6]:

            # Get the file key and file name
            file_key = obj['Key']
            filename = file_key.split("/")[-1]

            # Get the PRH file object
            file_object = utils.dataload.get_s3_file_object(s3_resource, file_key, bucket_name, subdirectory)

            # Load the PRH data from Mat file
            prh_data = utils.datatransform.get_prh_data_from_mat(file_object)

            # Get the orientation angles, depth, sampling rate values from the data
            head  = prh_data['head']          # head angle
            pitch = prh_data['pitch']         # pitch angle
            roll  = prh_data['roll']          # roll angle
            depths = prh_data['p'].squeeze()   # depth data of shape (n, 1) to (n,)
            sampling_rate = prh_data['fs'].squeeze()    # sampling rate


            ##### 2. COMPUTE WHALE TRAJECTORY ESTIMATES FOR EACH FILE #####
            # Compute the rotation matrix from the head, pitch, and roll angles
            rotation_matrices = utils.datatransform.get_rotation_matrix(head, pitch, roll)

            ### Compute the direction vector of the whale at each time step
            direction_vectors = utils.datatransform.get_direction_vector(initial_direction, rotation_matrices)

            ### Compute the speed estimate of the whale at each time step
            speed_estimate = utils.datatransform.get_speed_estimate(direction_vectors,
                                                                    depths,
                                                                    sampling_rate)

            ### Compute the velocity estimate of the whale at each time step
            velocity_estimate = utils.datatransform.get_velocity_estimate(speed_estimate, direction_vectors)

            ### Compute the position estimate of the whale at each time step
            position_estimate, z_component = utils.datatransform.get_position_estimate(velocity_estimate, depths)



            ###### 3. CREATE ANIMATIONS ########

            # get start time for full animation
            full_animation_start_time = time.time()

            ### Get Zoomed out animation of the whale's trajectory

            # get start time
            start_time = time.time()

            print(f"Creating zoomed out animation for {filename}...")

            #### Create the zoomed out animation
            utils.datavisualize.create_zoomed_out_animation(position_estimate,
                                              sampling_rate,
                                              filename,
                                              step_size=50,
                                              figsize=(10, 5),
                                              title='Whale Trajectory',
                                              colormap='autumn',
                                              dpi=25,
                                              )

            # get end time
            end_time = time.time()

            # get time elapsed to create the animation
            time_elapsed = end_time - start_time

            print(f"Time elapsed to create the zoomed out animation: {time_elapsed:.2f} seconds")


            ### Get Zoomed in animation of the whale's orientation
            # get start time
            start_time = time.time()

            print(f"Creating zoomed in animation for {filename}...")

            utils.datavisualize.create_zoomed_in_animation(rotation_matrices,
                                                           direction_vectors,
                                                           sampling_rate,
                                                           whale_object_path,
                                                           filename,
                                                           step_size=50,
                                                           title='Whale Orientation',
                                                           figsize=(5, 5),
                                                           dpi = 25,
                                                              )


            # get end time
            end_time = time.time()

            # get time elapsed to create the animation
            time_elapsed = end_time - start_time

            print(f"Time elapsed to create the zoomed in animation: {time_elapsed:.2f} seconds")


            #### Create depth animation

            # get start time
            start_time = time.time()

            print(f"Creating depth animation for {filename}...")

            utils.datavisualize.create_depth_animation(position_estimate[:,2],
                                                       sampling_rate,
                                                           filename,
                                                           step_size=50,
                                                           title='Depth',
                                                           figsize=(5, 5),
                                                            dpi = 25,
                                                              )

            # utils.datavisualize.create_depth_animation(z_component,
            #                                            sampling_rate,
            #                                            f"{filename}_z_component",
            #                                            step_size=200,
            #                                            title='Depth',
            #                                            figsize=(6, 6),
            #                                            dpi=80,
            #                                            )

            # get end time
            end_time = time.time()

            # get time elapsed to create the animation
            time_elapsed = end_time - start_time

            print(f"Time elapsed to create the depth animation: {time_elapsed:.2f} seconds")


            ### Create orientation animation

            # get start time
            start_time = time.time()

            print(f"Creating orientation animation for {filename}...")

            utils.datavisualize.create_orientation_animation(head,
                                                             pitch,
                                                             roll,
                                                             sampling_rate,
                                                             filename,
                                                             step_size=50,
                                                             title='Orientation Angles',
                                                             figsize=(5, 5),
                                                             dpi = 25,
                                                             )


            # get end time
            end_time = time.time()

            # get time elapsed to create the animation
            time_elapsed = end_time - start_time

            print(f"Time elapsed to create the orientation animation: {time_elapsed:.2f} seconds")

            ### Create xy animation

            # get start time
            start_time = time.time()

            print(f"Creating xy animation for {filename}...")

            utils.datavisualize.create_xy_animation(position_estimate[:,0],
                                                    position_estimate[:,1],
                                                    sampling_rate,
                                                    filename,
                                                    step_size=50,
                                                    title='XY Movement',
                                                    figsize=(5, 5),
                                                    dpi = 25,
                                                    )


            # get end time
            end_time = time.time()

            # get time elapsed to create the animation
            time_elapsed = end_time - start_time

            print(f"Time elapsed to create the xy animation: {time_elapsed:.2f} seconds")


            ### Merge all of the videos to make one big visualization

            # top section with zoomed in and orientation animations
            merge_command = f'ffmpeg -i {filename}_zoomed_in.mp4 -i {filename}_orientation.mp4 -filter_complex "[0:v][1:v]hstack" -c:v libx264 {filename}_top.mp4'
            res = subprocess.call(merge_command, shell=True)
            print("top.mp4 created.")

            # bottom section with depth, and xy animations
            merge_command = f'ffmpeg -i {filename}_xy.mp4 -i {filename}_depth.mp4 -filter_complex "[0:v][1:v]hstack" -c:v libx264 {filename}_bottom.mp4'
            res = subprocess.call(merge_command, shell=True)
            print("bottom.mp4 created.")

            # merge the zoomed out animation with the top section
            merge_command = f'ffmpeg -i {filename}_zoomed_out.mp4 -i {filename}_top.mp4 -filter_complex "[0:v][1:v]vstack" -c:v libx264 {filename}_merged_top.mp4'
            res = subprocess.call(merge_command, shell=True)
            print("mergedtop.mp4 created.")

            # merge the bottom section with the merged top section
            merge_command = f'ffmpeg -i {filename}_merged_top.mp4 -i {filename}_bottom.mp4 -filter_complex "[0:v][1:v]vstack" -c:v libx264 {filename}_merged.mp4'
            res = subprocess.call(merge_command, shell=True)
            print("merged.mp4 created.")

            # get end time
            end_time = time.time()

            # get time elapsed to create the animation
            time_elapsed = end_time - full_animation_start_time

            print(f"Time elapsed to create full animation: {time_elapsed:.2f} seconds")


            # # delete the intermediate files
            # os.remove(f"{filename}_top.mp4")
            # os.remove(f"{filename}_bottom.mp4")
            # os.remove(f"{filename}_merged_top.mp4")
            # os.remove(f"{filename}_zoomed_in.mp4")
            # os.remove(f"{filename}_orientation.mp4")
            # os.remove(f"{filename}_xy.mp4")
            # os.remove(f"{filename}_depth.mp4")
            # os.remove(f"{filename}_zoomed_out.mp4")








            # ### Compare with old method for rotation matrix computation
            # if 'head' in prh_data.keys():  # check if the whale data contains head, pitch, and roll angles
            #
            #     for i in range(len(prh_data['head'][:100])):  # len(prh_data['head']) is the number of time steps in the data
            #
            #         # create a rotation matrix based on the head, pitch, and roll angles
            #         rot_mat = make_rotation_matrix(prh_data['head'][i][0], prh_data['pitch'][i][0],
            #                                                prh_data['roll'][i][0])
            #
            #         # use the rotation matrix to obtain the 3D orientation of the whale
            #         no_speed_whale = np.array([1, 0, 0])  # unit vector in the x direction
            #
            #         rotated_whale_vector = np.matmul(no_speed_whale, rot_mat)
            #






