#### line plot ####
def create_trajectory_animation(position_estimate,
                                sampling_rate,
                                filename,
                                step_size=300,
                                figsize=(10, 10),
                                title='Whale Trajectory',
                                colormap='cividis'):

    # Animation function
    def update_lines(index):

        # Update the coordinates of the line
        line.set_data(data[0:2, :index])
        line.set_3d_properties(data[2, :index])

        # Update the position of the markers
        z = data[2, index:]


        # offsets sets the x, y, z coordinates of the markers
        markers._offsets3d = (data[0, index:], data[1, index:], z)

        # set_array sets the color of the markers based on the z coordinate and colormap
        markers.set_array(z)


        # markers._offsets3d = (data[0, index:], data[1, index:], data[2, index:])

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

    # data = np.array([x, y, z])
    position_estimate = position_estimate[::step_size, :]
    data = position_estimate.T


    # Plot the line
    line = ax.plot(data[0],
                   data[1],
                   data[2],
                   )[0]

    # Plot the markers

    # initialize the markers with first index values and colormap
    markers = ax.scatter(data[0, 0],
                         data[1, 0],
                         data[2, 0],
                         marker='.',
                         s=2,
                         c=data[2, 0],
                         # cmap=plt.get_cmap('cividis'),
                         alpha=0.2
                         )


    # markers = ax.scatter([ ], [], [], marker='o' )
    #
    # ax.set_xlim(-1.1, 1.1)
    # ax.set_ylim(-1.1, 1.1)
    # ax.set_zlim(-1.1, 1.1)

    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    ax.set_zlabel("Depth")
    plt.rcParams['animation.html'] = 'jshtml'

    line_ani = animation.FuncAnimation(fig, update_lines,
                                       frames=len(position_estimate),
                                       # frames=len(t),
                                       fargs= None,

                                       interval=100, blit=True, repeat=True)

    line_ani.save('line_animation_3d_funcanimation.mp4', writer='ffmpeg', fps=1000 / 100)





# def get_lineplot():
#
#     def update_lines(index, dataLines, lines, markers):
#
#         for line, data, marker in zip(lines, dataLines, markers):
#             # Update the coordinates of the line
#             line.set_data(data[0:2, :index])
#             line.set_3d_properties(data[2, :index])
#             # Update the position of the markers
#             marker._offsets3d = (data[0, index:], data[1, index:], data[2, index:])
#
#         return lines + markers
#
#     # Attaching 3D axis to the figure
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Lines to plot in 3D
#     t = np.linspace(-2 * np.pi, 2 * np.pi, 50)
#     x1, y1, z1 = np.cos(t), np.sin(t), t / t.max()
#     # x2, y2, z2 = t / t.max(), np.cos(t), np.sin(t)
#     data = np.array([x1, y1, z1])
#
#     # Plot the lines
#     line = ax.plot(data[0], data[1], data[2])[0]
#
#     # Plot the markers
#     markers = ax.scatter([], [], [], marker='o') # empty scatter plot
#
#     ax.set_xlim(-1.1, 1.1)
#     ax.set_ylim(-1.1, 1.1)
#     ax.set_zlim(-1.1, 1.1)
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("z")
#     plt.rcParams['animation.html'] = 'jshtml'
#
#     line_ani = animation.FuncAnimation(fig, update_lines, frames=len(t), fargs=(data, line, markers),
#                                        interval=100, blit=True, repeat=True)
#
#     line_ani.save('line_animation_3d_funcanimation.mp4', writer='ffmpeg', fps=1000 / 100)





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
#
#     # Extract the x, y, and z coordinates from the position estimate.
#     x = position_estimate[:, 0]
#     y = position_estimate[:, 1]
#     z = position_estimate[:, -1]
#
#     # Set up the 3D figure
#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_subplot(111, projection= '3d')
#     ax.view_init(azim=-35, elev=25)
#
#     ax.set_title(title, fontsize=18)
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_zticklabels([])
#
#
#
#
#
#     # Create a 3D scatter plot of the whale's trajectory using the extracted coordinates, with the color of each point
#     scatter = ax.scatter(x, y, z,
#                          c=z,  # Set the color of each point to the depth of the point.
#                          cmap=plt.get_cmap(colormap),
#                          s=4)
#
#     cb = fig.colorbar(scatter)
#     cb.set_label('Depth', fontsize=15)
#
#
#
#     # Create an empty line object that will be used to visualize the whale's current position in the animation.
#     line, = ax.plot([], [], [], marker='o', markersize=20, color='black')
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
#                                    frames= len(x),  # set the number of frames to the number of positions in the trajectory.
#
#                                    # Set the number of frames to the number of positions in the trajectory.
#                                    interval=(1 / sampling_rate) * 1000,
#                                    # set the interval between frames to the sampling rate of the data
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













