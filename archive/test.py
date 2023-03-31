# Gets the trajectory estimates for the whale, used to create all of the different movement plots.
import math
import numpy as np
import scipy.io

# # added by LKC
# import pandas as pd

def make_rotation_matrix(head, pitch, roll):
    '''
    Create a rotation matrix based on head, pitch and roll angles that can be used to get the orientation of the whale in 3 dimensions.
    '''
    return np.array([
        [math.cos(head)*math.cos(pitch), -1*math.cos(head)*math.sin(pitch)*math.sin(roll) - math.sin(head)*math.cos(roll), -1*math.cos(head)*math.sin(pitch)*math.cos(roll) + math.sin(head)*math.sin(roll)],
        [math.sin(head)*math.cos(pitch), -1*math.sin(head)*math.sin(pitch)*math.sin(roll) + math.cos(head)*math.cos(roll), -1*math.sin(head)*math.sin(pitch)*math.cos(roll) - math.cos(head)*math.sin(roll)],
        [math.sin(pitch), math.cos(pitch)*math.sin(roll), math.cos(pitch)*math.cos(roll)]])


def create_whale_trajectory(prh_file):
    # Create the whale trajectories from the PRH file, using rotation matrix to get direction angle.
    whale_data = scipy.io.loadmat(prh_file)

    print("Whale PRH Data: ")
    print(pd.DataFrame(whale_data))


    name = prh_file.split('/')[-1]
    frame_rate = whale_data['fs'][0][0]

    x = [0]
    y = [0]
    z = [0]
    speed_list = [0]

    if 'head' in whale_data.keys(): # This just makes sure that the head, pitch and roll angles have been computed. It seems like with the new data you will have to compute these.
        for i in range(len(whale_data['head'])):
            rotation_matrix = make_rotation_matrix(whale_data['head'][i][0], whale_data['pitch'][i][0], whale_data['roll'][i][0])
            no_speed_whale = np.array([1, 0, 0])
            rotated_whale = np.matmul(no_speed_whale, rotation_matrix)
            if i != len(whale_data['head']) - 1 and abs(math.atan(rotated_whale[2]/(math.sqrt(rotated_whale[0]**2 + rotated_whale[1]**2)))) > math.pi/6:
                # This checks that the angle is big enough for a dive, and then computes the speed from change in depth.
                speed_multiplier = abs(whale_data['p'][i][0] - whale_data['p'][i+1][0])/rotated_whale[2]
            else:
                # Otherwise we just assume the whale moves at 1.5m/s at the surface. This was given by a biologist but can be changed if incorrect.
                speed_multiplier = 1.5/frame_rate
            speed_list.append(speed_multiplier)
            whale = speed_multiplier * rotated_whale
            x.append(x[-1] + whale[0])
            y.append(y[-1] + whale[1])
            z.append(-1 * whale_data['p'][i][0])
            speed_list.append(speed_multiplier)
        pitch = whale_data['pitch']
        roll = whale_data['roll']
        head = whale_data['head']
        acc = whale_data['Aw']
    else:
        raise ValueError('Need to implement transformation when not given pitch, roll and head explicitly.')

    return {'name': name, 'x': x, 'y': y, 'z': z, 'fs': frame_rate, 'pitch': pitch, 'roll': roll, 'head': head, 'color': 'black', 'speed': speed_list}




def create_whale_trajectory(prh_file):
    '''
    This function creates the whale trajectory estimates, which are used to create various movement plots.
    The function reads the PRH file of the whale, creates a rotation matrix based on the head, pitch, and roll angles, and uses it to obtain the 3D orientation of the whale.
    The speed of the whale is computed based on the difference in depth readings between consecutive time steps and the time elapsed between them.
    '''
    # load the PRH file
    whale_data = scipy.io.loadmat(prh_file)

    # file name
    name = prh_file.split('/')[-1]
    # frame rate of the data
    frame_rate = whale_data['fs'][0][0]

    # create empty lists to store the whale trajectory
    x = np.zeros(len(whale_data['head'])+1)
    y = np.zeros(len(whale_data['head'])+1)
    z = np.zeros(len(whale_data['head'])+1)
    speed_list = np.zeros(len(whale_data['head'])+1)

    head = whale_data['head'][:,0]
    pitch = whale_data['pitch'][:,0]
    roll = whale_data['roll'][:,0]
    p = whale_data['p'][:,0]

    # create rotation matrix based on head, pitch, and roll angles
    rotation_matrix = make_rotation_matrix(head, pitch, roll)

    # use rotation matrix to obtain 3D orientation of the whale
    no_speed_whale = np.array([1, 0, 0])
    rotated_whale_vectors = np.matmul(no_speed_whale, rotation_matrix)

    # calculate speed multiplier based on depth readings
    diff_p = np.diff(p)
    time_diff = 1/frame_rate
    diving_mask = np.abs(np.arctan(rotated_whale_vectors[:,2] / np.sqrt(rotated_whale_vectors[:,0]**2 + rotated_whale_vectors[:,1]**2)))\
                  > np.pi/6

    diving_speed_multiplier = np.abs(diff_p[diving_mask]) / (rotated_whale_vectors[:-1,:][diving_mask,2] * time_diff)
    surface_speed_multiplier = 1.5 / frame_rate
    speed_multiplier = np.where(diving_mask, diving_speed_multiplier, surface_speed_multiplier)

    # calculate whale movement
    whale_vectors = np.multiply(speed_multiplier.reshape(-1,1), rotated_whale_vectors[:-1,:])

    # calculate whale position
    x[1:], y[1:], z[1:] = np.cumsum(whale_vectors, axis=0).T
    z[1:] *= -1

    # append speed values to list
    speed_list[1:] = speed_multiplier

    return {'name': name,
            'x': x.tolist(),
            'y': y.tolist(),
            'z': z.tolist(),
            'fs': frame_rate,
            'pitch': pitch.tolist(),
            'roll': roll.tolist(),
            'head': head.tolist(),
            'color': 'red',
            'speed': speed_list.tolist()}


# Plot 3D visualization of the entire whale trajectory.
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np




def create_zoomed_out_animation(position_estimate,
                                sampling_rate,
                                filename,
                                step_size = 50,
                                figsize = (6, 6),
                                title = 'Whale Trajectory'):
    '''
    This function creates a zoomed out animation of the whale trajectory.
    :param position_estimate: 3D position estimate of the whale
    :param sampling_rate:  sampling rate of the data (eg 10 for data collected every 0.1 seconds)
    :param filename:  name of the file to save the animation
    :param step_size: step size to use when plotting the trajectory
    :param figsize:  figure
    :param title:    figure title

    :return: None (saves animation as .mp4 file)
    '''

    ### get animation
    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        index = i
        xdata = np.array([x[index]])
        ydata = np.array([y[index]])
        zdata = np.array([z[index]])

        line.set_data(xdata, ydata)
        line.set_3d_properties(zdata)

        return line,

    xdata, ydata = [], []

    # get subset of x, y, z, coordinates of position estimate with step size
    x = position_estimate[::step_size, 0]
    y = position_estimate[::step_size, 1]
    z = position_estimate[::step_size, -1]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111,
                         projection='3d')
    ax.set_title( title, fontsize=20)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Create a 3D scatter plot of the whale's trajectory using the extracted coordinates, with the color of each point
    scatter = ax.scatter(x, y, z,
                         c=z,  # Set the color of each point to the depth of the point.
                         cmap=plt.get_cmap('autumn'),
                         s=2)

    fig.colorbar(scatter, label='Depth', label_size=15 )

    # Create an empty line object that will be used to visualize the whale's current position in the animation.
    line, = ax.plot([], [], [], marker='o', markersize=10, color='black')

    # Create the animation.
    anim = animation.FuncAnimation(fig,
                                   animate,
                                   init_func=init,
                                   fargs=None,
                                   frames=len(x),
                                   # Set the number of frames to the number of positions in the trajectory.
                                   interval=(1 / sampling_rate) * 1000,  # Set the interval between frames to the sampling rate.
                                   blit=True)

    frames = len(x) // step_size  # set the number of frames to the number of positions in the trajectory

    # Save the animation as an mp4 file with lower resolution.
    anim.save(f'whale_trajectory_{filename}.mp4',
              dpi=80,  # video resolution
              bitrate=-1  # video bitrate
              )  # extra_args is a list of arguments to pass to ffmpeg.




def create_zoomed_out_animation(position_estimate,
                                sampling_rate,
                                filename,
                                step_size = 50,
                                title = 'Whale Trajectory',
                                figsize = (6, 6),
                                colormap = 'viridis'):
    '''
    This function creates a zoomed out animation of the whale trajectory.
    :param position_estimate: 3D position estimate of the whale
    :param sampling_rate:  sampling rate of the data (eg 10 for data collected every 0.1 seconds)
    :param filename:  name of the file to save the animation
    :param step_size: step size to use when plotting the trajectory
    :param figsize:  figure size
    :param title:    figure title
    :param colormap: colormap to use for the visualization

    :return: None (saves animation as .mp4 file)
    '''

    ### get animation
    def init():
        line.set_data([], [])
        return line,

    # animation function. Called sequentially
    def animate(i):
        index = i
        xdata = np.array([x[index]])
        ydata = np.array([y[index]])
        zdata = np.array([z[index]])

        line.set_data(xdata, ydata)
        line.set_3d_properties(zdata)

        return line,

    xdata, ydata = [], []

    # get subset of x, y, z, coordinates of position estimate with step size
    x = position_estimate[:, 0]
    y = position_estimate[:, 1]
    z = position_estimate[:, -1]

    # create a 3d figure with specified viewing angle
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection= '3d')
    ax.view_init(azim=45, elev=30)

    ax.set_title(title, fontsize=18)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])



    # Create a 3D scatter plot of the whale's trajectory using the extracted coordinates, with the color of each point
    scatter = ax.scatter(x[::step_size], y[::step_size], z[::step_size],
                         c=z[::step_size],  # Set the color of each point to the depth of the point.
                         cmap=plt.get_cmap(colormap),
                         s=3)

    cb = fig.colorbar(scatter)
    cb.set_label('Depth', fontsize=15)



    # Create an empty line object that will be used to visualize the whale's current position in the animation.
    line, = ax.plot([], [], [], marker='o', markersize=20, color='black')

    frames = len(x) // step_size  # set the number of frames to the number of positions in the trajectory

    print(f'frames: {frames}')

    # Create the animation.
    anim = animation.FuncAnimation(fig,
                                   animate,
                                   init_func=init,
                                   fargs=None,
                                   frames=len(x),
                                   # Set the number of frames to the number of positions in the trajectory.
                                   interval=(1 / sampling_rate) * 1000,  # Set the interval between frames to the sampling rate.
                                   blit=True)



    # Save the animation as an mp4 file with lower resolution.
    anim.save(f'{filename}.mp4',
              dpi=80,  # video resolution
              bitrate=-1  # video bitrate
              )  # extra_args is a list of arguments to pass to ffmpeg.

    print('Animation saved as .mp4 file.')

    # def create_zoomed_out_animation(position_estimate,
    #                                 sampling_rate,
    #                                 filename,
    #                                 step_size = 50,
    #                                 title = 'Whale Trajectory',
    #                                 figsize = (6, 6),
    #                                 colormap = 'viridis'):
    #     '''
    #     This function creates a zoomed out animation of the whale trajectory.
    #     :param position_estimate: 3D position estimate of the whale
    #     :param sampling_rate:  sampling rate of the data (eg 10 for data collected every 0.1 seconds)
    #     :param filename:  name of the file to save the animation
    #     :param step_size: step size to use when plotting the trajectory
    #     :param figsize:  figure size
    #     :param title:    figure title
    #     :param colormap: colormap to use for the visualization
    #
    #     :return: None (saves animation as .mp4 file)
    #     '''
    #
    #     ### get animation
    #     def init():
    #         scatter.set_offsets([])    # set the scatter plot to empty
    #         line.set_data([], [])      # set the line to empty
    #         line.set_3d_properties([]) # set 3d properties to empty
    #         return scatter, line,
    #
    #     # animation function. Called sequentially
    #     def animate(i):
    #         index = i
    #         # xdata = np.array([x[index]])
    #         # ydata = np.array([y[index]])
    #         # zdata = np.array([z[index]])
    #
    #
    #         scatter.set_offsets(np.c_[x[index], y[index]]) # set the scatter plot to the current position
    #
    #
    #         line.set_data(x[:index+1], y[:index+1])
    #         line.set_3d_properties(z[index+1])
    #
    #         return scatter, line,
    #
    #     # get subset of x, y, z, coordinates of position estimate with step size
    #     x = position_estimate[:, 0]
    #     y = position_estimate[:, 1]
    #     z = position_estimate[:, -1]
    #
    #     # create a 3d figure with specified viewing angle
    #     fig = plt.figure(figsize=figsize)
    #     ax = fig.add_subplot(111, projection= '3d')
    #     ax.view_init(azim=-35, elev=25)
    #
    #     ax.set_title(title, fontsize=18)
    #
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])
    #     ax.set_zticklabels([])
    #
    #
    #     # Create a 3D scatter plot of the whale's trajectory using the extracted coordinates, with the color of each point
    #     scatter = ax.scatter(x[0], y[0], z[0],
    #
    #                          # c=z[0],  # Set the color of each point to the depth of the point.
    #                          cmap=plt.get_cmap(colormap),
    #                          s=3)
    #
    #     cb = fig.colorbar(scatter)
    #     cb.set_label('Depth', fontsize=15)
    #
    #
    #
    #     # Create an empty line object that will be used to visualize the whale's current position in the animation.
    #     line, = ax.plot([], [], [],
    #                     linewidth=3,
    #                     # marker='o',
    #                     # markersize=20,
    #                     color='black')
    #
    #     frames = len(x) // step_size  # set the number of frames to the number of positions in the trajectory
    #
    #     print(f'frames: {frames}')
    #
    #     # Create the animation.
    #     anim = animation.FuncAnimation(fig,
    #                                    animate,
    #                                    init_func=init,
    #                                    fargs=None,
    #                                    frames=len(x),
    #                                    # Set the number of frames to the number of positions in the trajectory.
    #                                    interval=(1 / sampling_rate) * 1000,  # Set the interval between frames to the sampling rate.
    #                                    blit=True)
    #
    #
    #
    #     # Save the animation as an mp4 file with lower resolution.
    #     anim.save(f'{filename}.mp4',
    #               dpi=80,  # video resolution
    #               bitrate=-1  # video bitrate
    #               )  # extra_args is a list of arguments to pass to ffmpeg.
    #
    #     print('Animation saved as .mp4 file.')

    # def get_trajectory_animation(position_estimate,
    #                                 sampling_rate,
    #                                 filename,
    #                                 step_size = 50,
    #                                 title = 'Whale Trajectory',
    #                                 figsize = (6, 6),
    #                                 colormap = 'viridis'):
    #     '''
    #     This function creates a zoomed out animation of the whale trajectory.
    #     :param position_estimate: 3D position estimate of the whale
    #     :param sampling_rate:  sampling rate of the data (eg 10 for data collected every 0.1 seconds)
    #     :param filename:  name of the file to save the animation
    #     :param step_size: step size to use when plotting the trajectory
    #     :param figsize:  figure size
    #     :param title:    figure title
    #     :param colormap: colormap to use for the visualization
    #
    #     :return: None (saves animation as .mp4 file)
    #     '''
    #
    #     def init():
    #         line.set_data([], [])
    #         return line,
    #
    #     def update_line(frame_num, data, line):
    #
    #         # NOTE: there is no .set_data() for 3 dim data...
    #         line.set_data(data[: frame_num])
    #         line.set_3d_properties(data[:frame_num, 2])
    #
    #         return line,
    #
    #     # Draw the animation
    #     fig = plt.figure(figsize=figsize)
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.view_init(azim=45, elev=30)
    #
    #     ax.set_title(title, fontsize=18)
    #
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])
    #     ax.set_zticklabels([])
    #
    #
    #     # Create an empty line object that will be used to visualize the whale's current position in the animation.
    #     line, = ax.plot([], [], [], marker='o', markersize=20, color='black')
    #
    #     # line = ax.plot(position_estimate[0, 0:1], position_estimate[1, 0:1], position_estimate[2, 0:1],
    #     #                  # cmap=plt.get_cmap(colormap),
    #     #                  color='black'
    #     #                  )[0]
    #
    #
    #     # plt.rcParams['animation.html'] = 'html5'
    #
    #     # Line animation
    #     line_ani = animation.FuncAnimation(fig,
    #                                        update_line,
    #                                         init_func=init,
    #                                         frames=len(position_estimate),
    #                                         fargs=(position_estimate, line),
    #                                         interval=(1 / sampling_rate) * 1000,
    #                                         blit=True # blit=True means only re-draw the parts that have changed.
    #                                        )
    #
    #
    #     line_ani.save('line_animation_3d_funcanimation.mp4', writer='ffmpeg',
    #                   fps=1000/100 #
    #                   )
    #
    #     # Save the animation as an mp4 file with lower resolution.
    #     # line_ani.save(f'{filename}.mp4',
    #     #                 dpi=80,  # video resolution
    #     #                 bitrate=-1  # video bitrate
    #     #                 )  # extra_args is a list of arguments to pass to ffmpeg.
    #
    #
    #     print('Animation saved as .mp4 file.')







"""
This file contains functions for visualizing the whale trajectory and movement.
The 3d visualizations are created using the matplotlib library and saved as video animations.
"""

### For creating animations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from mpl_toolkits.mplot3d import Axes3D


#### Animation functions ####
def create_zoomed_out_animation(position_estimate,
                                sampling_rate,
                                filename,
                                step_size = 50,
                                title = 'Whale Trajectory',
                                figsize = (6, 6),
                                colormap = 'viridis'):
    '''
    This function creates a zoomed out animation of the whale trajectory.
    :param position_estimate: 3D position estimate of the whale
    :param sampling_rate:  sampling rate of the data (eg 10 for data collected every 0.1 seconds)
    :param filename:  name of the file to save the animation
    :param step_size: step size to use when plotting the trajectory
    :param figsize:  figure size
    :param title:    figure title
    :param colormap: colormap to use for the visualization

    :return: None (saves animation as .mp4 file)
    '''


    # Extract the x, y, and z coordinates from the position estimate.
    x = position_estimate[:, 0]
    y = position_estimate[:, 1]
    z = position_estimate[:, -1]

    # Set up the 3D figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection= '3d')
    ax.view_init(azim=-35, elev=25)

    ax.set_title(title, fontsize=18)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])



    # Create initial empty line object that will be used to visualize the whale's current position in the animation.
    line, = ax.plot([], [], [], linewidth=2, color='b')

    # Set up the animation function.
    def animate(frame):
        # update line data with new coordinates
        line.set_data(x[:frame], y[:frame])
        line.set_3d_properties(z[:frame])
        return line,

    # Set up the animation.
    frames = range(0, len(x), step_size)  # set the number of frames to the number of positions in the trajectory
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=(1 / sampling_rate) * 1000)

    # Set up the colormap and colorbar.
    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(z.min(), z.max())
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm)

    # Save the animation as an mp4 file with lower resolution.
    anim.save(f'{filename}.mp4', dpi=80, bitrate=-1)




    # # Create a 3D scatter plot of the whale's trajectory using the extracted coordinates, with the color of each point
    # scatter = ax.scatter(x, y, z,
    #                      c=z,  # Set the color of each point to the depth of the point.
    #                      cmap=plt.get_cmap(colormap),
    #                      s=4)
    #
    # cb = fig.colorbar(scatter)
    # cb.set_label('Depth', fontsize=15)
    #
    #
    # # Create an empty line object that will be used to visualize the whale's current position in the animation.
    # line, = ax.plot([], [], [], marker='o', markersize=20, color='black')
    #
    # frames = len(x) // step_size  # set the number of frames to the number of positions in the trajectory
    #
    # print(f'frames: {frames}')
    #
    # # Create the animation.
    # anim = animation.FuncAnimation(fig,
    #                                animate,
    #                                init_func=init,
    #                                fargs=None,
    #                                frames= len(x),  # set the number of frames to the number of positions in the trajectory.
    #
    #                                # Set the number of frames to the number of positions in the trajectory.
    #                                interval=(1 / sampling_rate) * 1000,
    #                                # set the interval between frames to the sampling rate of the data
    #                                blit=True)
    #
    #
    #
    # # Save the animation as an mp4 file with lower resolution.
    # anim.save(f'{filename}.mp4',
    #           dpi=80,  # video resolution
    #           bitrate=-1  # video bitrate
    #           )  # extra_args is a list of arguments to pass to ffmpeg.

    print('Animation saved as .mp4 file.')



# ### get animation
#     def init():
#         line.set_data([], [])
#         line.set_3d_properties([])
#         return line,
#
#     # animation function. Called sequentially
#     def animate(i):
#         index = i
#         xdata = np.array([x[index]])
#         ydata = np.array([y[index]])
#         zdata = np.array([z[index]])
#
#         line.set_data(xdata, ydata)
#         line.set_3d_properties(zdata)
#
#         return line,






