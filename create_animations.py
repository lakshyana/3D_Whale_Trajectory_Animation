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
        for obj in result['Contents'][:1]:

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
            depth = prh_data['p'].squeeze()   # depth data of shape (n, 1) to (n,)
            sampling_rate = prh_data['fs'].squeeze()    # sampling rate



            ##### 2. COMPUTE WHALE TRAJECTORY ESTIMATES FOR EACH FILE #####
            # Compute the rotation matrix from the head, pitch, and roll angles
            rotation_matrices = utils.datatransform.get_rotation_matrix(head, pitch, roll)

            ### Compute the direction vector of the whale at each time step
            direction_vector = utils.datatransform.get_direction_vector(initial_direction, rotation_matrices)

            ### Compute the speed estimate of the whale at each time step
            speed_estimate = utils.datatransform.get_speed_estimate(direction_vector,
                                                      depth,
                                                      sampling_rate)

            ### Compute the velocity estimate of the whale at each time step
            velocity_estimate = utils.datatransform.get_velocity_estimate(speed_estimate, direction_vector)

            ### Compute the position estimate of the whale at each time step
            position_estimate = utils.datatransform.get_position_estimate(velocity_estimate, depth)


            ###### 3. CREATE ANIMATIONS ########

            ### Get Zoomed out animation of the whale's trajectory

            # # get start time
            # start_time = time.time()
            #
            # print(f"Creating zoomed out animation for {filename}...")
            #
            # #### Create the zoomed out animation
            # utils.datavisualize.create_zoomed_out_animation(position_estimate,
            #                                   sampling_rate,
            #                                   filename,
            #                                   step_size=200,
            #                                   figsize=(8, 8),
            #                                   title='Whale Trajectory',
            #                                   colormap='autumn',
            #                                   )
            #
            #
            # # get end time
            # end_time = time.time()
            #
            # # get time elapsed to create the animation
            # time_elapsed = end_time - start_time
            #
            # print(f"Time elapsed to create the zoomed out animation: {time_elapsed:.2f} seconds")
            #

            ### Get Zoomed in animation of the whale's orientation

            # get start time
            start_time = time.time()

            print(f"Creating zoomed in animation for {filename}...")

            utils.datavisualize.create_zoomed_in_animation(rotation_matrices,
                                                           sampling_rate,
                                                           whale_object_path,
                                                           filename,
                                                           step_size=200,
                                                           title='Whale Orientation',
                                                           figsize=(8, 8),

                                                              )

            # get end time
            end_time = time.time()

            # get time elapsed to create the animation
            time_elapsed = end_time - start_time

            print(f"Time elapsed to create the zoomed in animation: {time_elapsed:.2f} seconds")






