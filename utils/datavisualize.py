

"""
This file contains functions for visualizing the whale trajectory and movement.
The 3d visualizations are created using the matplotlib library and saved as video animations.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
from IPython.display import HTML

import utils.datatransform

def read_obj(filename):
    """
    Read the  3D object fileinto a numpy array.
    :param filename:  The filename.
    :return:  The numpy array containing the vertices and triangles.
    """
    triangles = []
    vertices = []
    with open(filename) as file:
        for line in file:
            components = line.strip(' \n').split(' ')
            if components[0] == "f": # face data
                indices = list(map(lambda c: int(c.split('/')[0]) - 1, components[1:]))
                for i in range(0, len(indices) - 2):
                    triangles.append(indices[i: i+3])
            elif components[0] == "v": # vertex data
                vertex = list(map(lambda c: float(c), components[2:]))
                vertices.append(vertex)
    return np.array(vertices), np.array(triangles)

#### Animation functions ####
def create_zoomed_out_animation(position_estimate,
                                sampling_rate,
                                filename,
                                step_size = 100,
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
    def update_point(i):
        index = i
        xdata = np.array([x[index]])
        ydata = np.array([y[index]])
        zdata = np.array([z[index]])

        line.set_data(xdata, ydata)
        line.set_3d_properties(zdata)

        return line,

    # Extract the x, y, and z coordinates from the position estimate.
    x = position_estimate[:, 0]
    y = position_estimate[:, 1]
    z = position_estimate[:, -1]


    # Calculate the range of the x, y, and z coordinates.
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    z_range = np.max(z) - np.min(z)

    # Set the figure size based on the range of the coordinates.
    figsize = (x_range // 10, y_range // 10, z_range // 10)

    # create a 3d figure with specified viewing angle
    fig = plt.figure(figsize=(6,6)) # set figure size
    ax = fig.add_subplot(111, projection= '3d') # set 3d projection
    ax.view_init(azim=-55, elev=20) # set viewing angle
    ax.set_title(title, fontsize=18) # set title
    # Remove the tick labels from the axes.
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.grid(alpha=0.2) # set grid transparency
    # # Set the limits of the axes.
    # ax.set_xlim3d([np.min(x), np.max(x)])
    # ax.set_ylim3d([np.min(y), np.max(y)])
    # ax.set_zlim3d([np.min(z), np.max(z)])
    # # Set size of the figure
    # fig.set_size_inches(4,4, 4)


    # Create a 3D scatter plot of the whale's trajectory using the extracted coordinates, with the color of each point
    scatter = ax.scatter(x[::step_size], y[::step_size], z[::step_size],
                         c = z[::step_size],  # Set the color of each point to the depth of the point.
                         cmap=plt.get_cmap(colormap),
                         # alpha=0.1,
                         s=10)

    cb = fig.colorbar(scatter)
    cb.set_label('Depth', fontsize=15)


    # Create an empty line object that will be used to visualize the whale's current position in the animation.
    line, = ax.plot([], [], [], marker='o', markersize=15, color='black')

    # Create the animation.
    anim = animation.FuncAnimation(fig,
                                   update_point,
                                   init_func=init,
                                   fargs=None,
                                   frames=len(x),
                                   # Set the number of frames to the number of positions in the trajectory.
                                   interval=(1 / sampling_rate) * 1000,  # Set the interval between frames to the sampling rate.
                                   blit=True)

    # Save the animation as an mp4 file with lower resolution.
    anim.save(f'{filename}_zoomed_out.mp4',
              dpi=80,  # video resolution
              # set a bitrate to avoid a large file size
              bitrate=-1  # video bitrate of -1
              )  # extra_args is a list of arguments to pass to ffmpeg.

    print('Animation saved as .mp4 file.')



def create_zoomed_in_animation(rotation_matrices,
                               pitch,
                               roll,
                               head,
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
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]
        plot = ax.plot_trisurf(x, y, triangles, z, shade=True, color='gray')  # Create a plot_trisurf object
        return plot,


    # animation function. Called sequentially
    def animate(i):  # Update the plot for each frame of the animation
        index = i  # Get the index of the current frame
        ax.clear()  # Clear the plot and set various properties
        ax.set_title('Whale Orientation')
        ax.set_xlim([-350, 350])
        ax.set_ylim([-350, 350])
        ax.set_zlim([-350, 350])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # Get the current rotation matrix
        rotation_matrix = rotation_matrices[index]

        new_vertices = np.matmul(vertices, rotation_matrix)  # Apply the rotation matrix to the vertices of the 3D model
        x = new_vertices[:, 0]
        y = new_vertices[:, 1]
        z = new_vertices[:, 2]
        plot = ax.plot_trisurf(x, y, triangles, z, shade=True,
                               color='gray')  # Create a plot_trisurf object with the new vertices

        # Add in lines coming out of the whale to show orientation.
        x_line_points_top = np.array([400, 0, 0])
        y_line_points_top = np.array([0, 400, 0])
        z_line_points_top = np.array([0, 0, 400])

        rotated_x_line_points_top = np.matmul(x_line_points_top,
                                              rotation_matrix)  # Apply the rotation matrix to the points of the lines
        rotated_y_line_points_top = np.matmul(y_line_points_top, rotation_matrix)
        rotated_z_line_points_top = np.matmul(z_line_points_top, rotation_matrix)

        x_line_points_bottom = np.array([-400, 0, 0])
        y_line_points_bottom = np.array([0, -400, 0])
        z_line_points_bottom = np.array([0, 0, -400])

        rotated_x_line_points_bottom = np.matmul(x_line_points_bottom,
                                                 rotation_matrix)  # Rotate the end points of the up/down lines using the rotation matrix
        rotated_y_line_points_bottom = np.matmul(y_line_points_bottom, rotation_matrix)
        rotated_z_line_points_bottom = np.matmul(z_line_points_bottom, rotation_matrix)

        # Create plot objects for the up/down, left/right, and front/back lines
        x_line, = ax.plot([rotated_x_line_points_bottom[0], rotated_x_line_points_top[0]],
                          [rotated_x_line_points_bottom[1], rotated_x_line_points_top[1]],
                          [rotated_x_line_points_bottom[2], rotated_x_line_points_top[2]], color='purple')
        y_line, = ax.plot([rotated_y_line_points_bottom[0], rotated_y_line_points_top[0]],
                          [rotated_y_line_points_bottom[1], rotated_y_line_points_top[1]],
                          [rotated_y_line_points_bottom[2], rotated_y_line_points_top[2]], color='green')
        z_line, = ax.plot([rotated_z_line_points_bottom[0], rotated_z_line_points_top[0]],
                          [rotated_z_line_points_bottom[1], rotated_z_line_points_top[1]],
                          [rotated_z_line_points_bottom[2], rotated_z_line_points_top[2]], color='red')
        # Add markers at the endpoints of the up/down lines to indicate the top and bottom of the whale
        top_marker, = ax.plot([rotated_z_line_points_top[0], rotated_z_line_points_top[0] + 1],
                              [rotated_z_line_points_top[1], rotated_z_line_points_top[1] + 1],
                              [rotated_z_line_points_top[2], rotated_z_line_points_top[2] + 1], color='black',
                              markersize=100)
        bottom_marker, = ax.plot([rotated_z_line_points_bottom[0], rotated_z_line_points_bottom[0] + 1],
                                 [rotated_z_line_points_bottom[1], rotated_z_line_points_bottom[1] + 1],
                                 [rotated_z_line_points_bottom[2], rotated_z_line_points_bottom[2] + 1], color='orange',
                                 markersize=100)

        # Add a circle in the XY plane of whale frame to show orientation.
        circle_x = [i * 10 for i in range(-35, 36)]
        circle_x = circle_x + circle_x[::-1]
        circle_y = [np.sqrt(122500 - i ** 2) for i in circle_x[:71]] + [-np.sqrt(122500 - i ** 2) for i in
                                                                          circle_x[71:]]
        circle_z = [0 for i in circle_x]
        circle_combined = np.array([circle_x, circle_y, circle_z]).transpose()
        rotated_circle = np.matmul(circle_combined, rotation_matrix)
        circle, = ax.plot(rotated_circle[:, 0], rotated_circle[:, 1], rotated_circle[:, 2], color='black')

        return plot, x_line, y_line, z_line, top_marker, bottom_marker, circle

    # Read in the whale object and make corrections for orientation.
    vertices, triangles = read_obj('whale.obj')

    # create a 3d figure with specified viewing angle
    fig = plt.figure(figsize=figsize) # set figure size
    ax = fig.add_subplot(111, projection= '3d') # set 3d projection
    ax.view_init(azim=-35, elev=25) # set viewing angle
    ax.set_title(title, fontsize=18) # set title
    # Remove the tick labels from the axes.
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.grid(alpha=0.4) # set grid transparency


    # Create a 3D scatter plot of the whale's trajectory using the extracted coordinates, with the color of each point
    scatter = ax.scatter(x[::step_size],
                         y[::step_size],
                         z[::step_size],
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







