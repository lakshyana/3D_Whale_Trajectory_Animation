

"""
This file contains functions for visualizing the whale trajectory and movement.
The 3d visualizations are created using the matplotlib library and saved as video animations.
"""

### For creating animations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
from IPython.display import HTML


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
    ax.view_init(azim=-35, elev=25)

    ax.set_title(title, fontsize=18)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


    # Create a 3D scatter plot of the whale's trajectory using the extracted coordinates, with the color of each point
    scatter = ax.scatter(x[::step_size], y[::step_size], z[::step_size],
                         c=z[::step_size],  # Set the color of each point to the depth of the point.
                         cmap=plt.get_cmap(colormap),
                         # alpha=0.1,
                         s=5)

    cb = fig.colorbar(scatter)
    cb.set_label('Depth', fontsize=15)

    # Create an empty line object that will be used to visualize the whale's current position in the animation.
    line, = ax.plot([], [], [], marker='o', markersize=20, color='black')

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
    anim.save(f'{filename}_zoomed_out.mp4',
              dpi=80,  # video resolution
              bitrate=-1  # video bitrate
              )  # extra_args is a list of arguments to pass to ffmpeg.

    print('Animation saved as .mp4 file.')



def create_zoomed_in_animation(position_estimate,
                               sampling_rate,
                                filename,
                                step_size = 50,
                                title = 'Whale Trajectory',
                                figsize = (6, 6),
                                colormap = 'viridis'):
    '''
    This function creates a zoomed in animation of the whale trajectory.
    :param position_estimate:
    :param sampling_rate:
    :param filename:
    :param step_size:
    :param title:
    :param figsize:
    :param colormap:
    :return:
    '''





