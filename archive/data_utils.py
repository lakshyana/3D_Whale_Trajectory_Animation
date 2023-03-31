

"""
This script contains utility functions for fetching and loading data from AWS S3, getting data transformations
for the whale movement animations.

Note: This script assumes that the user running this script has the AWS credentials set up on their machine to access
the whaleproject bucket on AWS S3 using the boto3 library.
"""



















# just for testing old rotation matrix function

import math
import numpy as np
def make_rotation_matrix(head, pitch, roll):
    '''
    Create a rotation matrix based on head, pitch, and roll angles that can be used to obtain the orientation of the whale in 3 dimensions.
    '''
    return np.array([
        [math.cos(head)*math.cos(pitch),
         -1*math.cos(head)*math.sin(pitch)*math.sin(roll) - math.sin(head)*math.cos(roll),
         -1*math.cos(head)*math.sin(pitch)*math.cos(roll) + math.sin(head)*math.sin(roll)],
        [math.sin(head)*math.cos(pitch), -1*math.sin(head)*math.sin(pitch)*math.sin(roll) + math.cos(head)*math.cos(roll), -1*math.sin(head)*math.sin(pitch)*math.cos(roll) - math.cos(head)*math.sin(roll)],
        [math.sin(pitch), math.cos(pitch)*math.sin(roll), math.cos(pitch)*math.cos(roll)]])


#### line plot ####
def create_trajectory_animation(position_estimate,
                                sampling_rate,
                                filename,
                                step_size=50,
                                figsize=(10, 10),
                                title='Whale Trajectory',
                                colormap='cividis'):
    # Animation function
    def update_lines(num, data, line, markers):
        # Update the coordinates of the line
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])

        # Update the position of the markers
        z = data[2, num:]

        # offsets sets the x, y, z coordinates of the markers
        markers._offsets3d = (data[0, num:], data[1, num:], z)

        # set_array sets the color of the markers based on the z coordinate and colormap
        markers.set_array(z)

        # markers._offsets3d = (data[0, num:], data[1, num:], data[2, num:])
        return line, markers

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=-35, elev=25)

    ax.set_title("title", fontsize=18)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Line to plot in 3D
    t = np.linspace(-2 * np.pi, 2 * np.pi, 50)
    x, y, z = np.cos(t), np.sin(t), t / t.max()
    data = np.array([x, y, z])

    # Plot the line
    line = ax.plot(data[0], data[1], data[2])[0]

    # Plot the markers

    # initialize the markers with first index values and colormap
    markers = ax.scatter(data[0, 0],
                         data[1, 0],
                         data[2, 0],
                         marker='o',
                         c=data[2, 0],
                         cmap=plt.get_cmap('cividis'))

    # markers = ax.scatter([ ], [], [], marker='o' )

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.rcParams['animation.html'] = 'jshtml'

    line_ani = animation.FuncAnimation(fig, update_lines, frames=len(t), fargs=(data, line, markers),
                                       interval=100, blit=True, repeat=True)

    line_ani.save('line_animation_3d_funcanimation.mp4', writer='ffmpeg', fps=1000 / 100)


# def get_rotation_matrix2(head, pitch, roll):
#     """
#     Get the rotation matrix from the head, pitch, and roll angles.
#     :param head: The head angle in radians as a numpy array.
#     :param pitch: The pitch angle in radians as a numpy array.
#     :param roll: The roll angle in radians as a numpy array.
#     :return: The rotation matrix.
#     """
#
#     # calculate the sin and cosine values for each angle
#     sh = np.sin(head)  # sin of head
#     ch = np.cos(head)  # cos of head
#     sp = np.sin(pitch) # sin of pitch
#     cp = np.cos(pitch) # cos of pitch
#     sr = np.sin(roll)  # sin of roll
#     cr = np.cos(roll)  # cos of roll
#
#     # formula source: https://soundtags.wp.st-andrews.ac.uk/files/2013/01/animal_orientation_tutorial.pdf
#
#     H = np.array([[ch, -sh, 0],
#                   [sh, ch, 0],
#                   [0, 0, 1]])
#
#     P = np.array([[cp, 0, -sp],
#                   [0, 1, 0],
#                   [sp, 0, cp]])
#
#     R = np.array([[1, 0, 0],
#                   [0, cr, -sr],
#                   [0, sr, cr]])
#
#     rotation_matrix = np.matmul(H, np.matmul(P, R))
#
#     #  return matrix after reshape into format N x 3 x 3 matrices for multiple orientations
#     return np.rollaxis(rotation_matrix.squeeze(), -1, 0) if rotation_matrix.shape[-1] == 1 else rotation_matrix
#     # return  np.rollaxis(rotation_matrix.squeeze(), -1, 0) if rotation_matrix.shape[-1] == 1 else rotation_matrix




############# Zoomed out animation  #############
# filename = file_key.split("/")[-1].split(".")[0] + "_zoomed_out"
# utils.create_zoomed_out_animation(position_estimate,
#                 sampling_rate,
#                 filename,
#                 step_size = 50,
#                 figsize = (6, 6),
#                 title = 'Whale Trajectory')

# ### get animation
# def init():
#     line.set_data([], [])
#     return line,
#
#
# def animate(i):
#     index = i
#     xdata = np.array([x[index]])
#     ydata = np.array([y[index]])
#     zdata = np.array([z[index]])
#
#     line.set_data(xdata, ydata)
#     line.set_3d_properties(zdata)
#
#     return line,
#
#
# xdata, ydata = [], []
#
# # get subset of x, y, z, coordinates of position estimate with step size
# step_size = 50
# x = position_estimate[::step_size, 0]
# y = position_estimate[::step_size, 1]
# z = position_estimate[::step_size, -1]
#
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111,
#                      projection='3d')
# ax.set_title('Whale Trajectory')
#
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_zticklabels([])
#
# # Create a 3D scatter plot of the whale's trajectory using the extracted coordinates, with the color of each point
# scatter = ax.scatter(x, y, z,
#                      c=z, # Set the color of each point to the depth of the point.
#                      cmap=plt.get_cmap('autumn'),
#                      s=5)
#
# fig.colorbar(scatter, label='Depth')
#
# # Create an empty line object that will be used to visualize the whale's current position in the animation.
# line, = ax.plot([], [], [], marker='o', markersize=20, color='black')
#
# # Create the animation.
# anim = animation.FuncAnimation(fig,
#                                animate,
#                                init_func=init,
#                                fargs=None,
#                                frames=len(x),  # Set the number of frames to the number of positions in the trajectory.
#                                interval= (1 / 10) * 1000, # Set the frame rate to 10 frames per second.
#                                blit=True)
#
# frames = len(x) //step_size # set the number of frames to the number of positions in the trajectory
#
# # Save the animation as an mp4 file with lower resolution.
# anim.save(f'whale_trajectory_{file_key.split("/")[-1]}.mp4',
#           dpi = 80,  # video resolution
#           bitrate=-1 # video bitrate
#           ) # extra_args is a list of arguments to pass to ffmpeg.


# ############# Zoomed in animation  #############
# def init():  # Initialize the plot
#     x = vertices[:, 0]
#     y = vertices[:, 1]
#     z = vertices[:, 2]
#     plot = ax.plot_trisurf(x, y, triangles, z, shade=True, color='gray')  # Create a plot_trisurf object
#     return plot,
#
#
# def animate(i):  # Update the plot for each frame of the animation
#     index = i  # Get the index of the current frame
#     ax.clear()  # Clear the plot and set various properties
#     ax.set_title('Whale Orientation')
#     ax.set_xlim([-350, 350])
#     ax.set_ylim([-350, 350])
#     ax.set_zlim([-350, 350])
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_zticklabels([])
#
#
#     new_vertices = np.matmul(vertices,
#                              rotation_matrix)  # Apply the rotation matrix to the vertices of the 3D model
#     x = new_vertices[:, 0]
#     y = new_vertices[:, 1]
#     z = new_vertices[:, 2]
#     plot = ax.plot_trisurf(x, y, triangles, z, shade=True,
#                            color='gray')  # Create a plot_trisurf object with the new vertices
#
#     # Add in lines coming out of the whale to show orientation.
#     x_line_points_top = np.array([400, 0, 0])
#     y_line_points_top = np.array([0, 400, 0])
#     z_line_points_top = np.array([0, 0, 400])
#     rotated_x_line_points_top = np.matmul(x_line_points_top,
#                                           rotation_matrix)  # Apply the rotation matrix to the points of the lines
#     rotated_y_line_points_top = np.matmul(y_line_points_top, rotation_matrix)
#     rotated_z_line_points_top = np.matmul(z_line_points_top, rotation_matrix)
#
#     x_line_points_bottom = np.array([-400, 0, 0])
#     y_line_points_bottom = np.array([0, -400, 0])
#     z_line_points_bottom = np.array([0, 0, -400])
#
#     rotated_x_line_points_bottom = np.matmul(x_line_points_bottom,
#                                              rotation_matrix)  # Rotate the end points of the up/down lines using the rotation matrix
#     rotated_y_line_points_bottom = np.matmul(y_line_points_bottom, rotation_matrix)
#     rotated_z_line_points_bottom = np.matmul(z_line_points_bottom, rotation_matrix)
#
#     # Create plot objects for the up/down, left/right, and front/back lines
#     x_line, = ax.plot([rotated_x_line_points_bottom[0], rotated_x_line_points_top[0]],
#                       [rotated_x_line_points_bottom[1], rotated_x_line_points_top[1]],
#                       [rotated_x_line_points_bottom[2], rotated_x_line_points_top[2]], color='purple')
#     y_line, = ax.plot([rotated_y_line_points_bottom[0], rotated_y_line_points_top[0]],
#                       [rotated_y_line_points_bottom[1], rotated_y_line_points_top[1]],
#                       [rotated_y_line_points_bottom[2], rotated_y_line_points_top[2]], color='green')
#     z_line, = ax.plot([rotated_z_line_points_bottom[0], rotated_z_line_points_top[0]],
#                       [rotated_z_line_points_bottom[1], rotated_z_line_points_top[1]],
#                       [rotated_z_line_points_bottom[2], rotated_z_line_points_top[2]], color='red')
#     # Add markers at the endpoints of the up/down lines to indicate the top and bottom of the whale
#     top_marker, = ax.plot([rotated_z_line_points_top[0], rotated_z_line_points_top[0] + 1],
#                           [rotated_z_line_points_top[1], rotated_z_line_points_top[1] + 1],
#                           [rotated_z_line_points_top[2], rotated_z_line_points_top[2] + 1],
#                           color='black', markersize=100)
#     bottom_marker, = ax.plot([rotated_z_line_points_bottom[0], rotated_z_line_points_bottom[0] + 1],
#                              [rotated_z_line_points_bottom[1], rotated_z_line_points_bottom[1] + 1],
#                              [rotated_z_line_points_bottom[2], rotated_z_line_points_bottom[2] + 1],
#                              color='orange', markersize=100)
#
#     # Add a circle in the XY plane of whale frame to show orientation.
#     circle_x = [i * 10 for i in range(-35, 36)]
#     circle_x = circle_x + circle_x[::-1]
#     circle_y = [math.sqrt(122500 - i ** 2) for i in circle_x[:71]] + [-math.sqrt(122500 - i ** 2) for i
#                                                                       in circle_x[71:]]
#     circle_z = [0 for i in circle_x]
#     circle_combined = np.array([circle_x, circle_y, circle_z]).transpose()
#     rotated_circle = np.matmul(circle_combined, rotation_matrix)
#     circle, = ax.plot(rotated_circle[:, 0], rotated_circle[:, 1], rotated_circle[:, 2], color='black')
#
#     return plot, x_line, y_line, z_line, top_marker, bottom_marker, circle
#
#
# pitch = whale_info['pitch']
# roll = whale_info['roll']
# head = whale_info['head']
#
# pitch = pitch[int(whale_info['fs'] * start_time): int(whale_info['fs'] * end_time)]
# roll = roll[int(whale_info['fs'] * start_time): int(whale_info['fs'] * end_time)]
# head = head[int(whale_info['fs'] * start_time): int(whale_info['fs'] * end_time)]
#
# # Read in the whale object and make corrections for orientation.
# vertices, triangles = read_obj('whale.obj')
# old_vertices = vertices.copy()
# xx = old_vertices[:, 0]
# zz = old_vertices[:, 1]
# yy = old_vertices[:, 2]
# vertices[:, 0] = -xx
# vertices[:, 1] = yy
# vertices[:, 2] = zz
#
# fig = plt.figure(figsize=(4, 4))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title('Whale Orientation')
#
# ax.set_xlim([-350, 350])
# ax.set_ylim([-350, 350])
# ax.set_zlim([-350, 350])
#
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_zticklabels([])
#
# anim = animation.FuncAnimation(fig, animate, init_func=init, fargs=None,
#                                frames=len(pitch), interval=(1 / whale_info['fs']) * 1000, blit=True)
#
# anim.save(
#
# )