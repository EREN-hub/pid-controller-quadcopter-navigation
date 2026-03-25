"""
author: Peter Huang
email: hbd730@gmail.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""

from controller import p_data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys

# It keeps the passed points of quadcopter
history = np.zeros((500,3))

# It tracks how many points already passed so far.
count = 0

# Global variable of time record
t_history = []

def plot_quad_3d(waypoints, get_world_frame):
    """
    get_world_frame is a function which return the "next" world frame to be drawn
    """
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    
    # Quadcopter visualization
    ax.plot([], [], [], '-', c='cyan')[0]
    ax.plot([], [], [], '-', c='red')[0]
    ax.plot([], [], [], '-', c='blue', marker='o', markevery=2)[0]
    ax.plot([], [], [], '.', c='red', markersize=4)[0]
    ax.plot([], [], [], '.', c='blue', markersize=2)[0]
    
    # This function sets the limits of 3 dimentional graph
    set_limit((-0.5,0.5), (-0.5,0.5), (-0.5,8))

    # This plots the desired way that quadcopter should follow
    plot_waypoints(waypoints)

    # ---------This part added for this project----------
    # Global 'lines' array initialization
    global lines
    lines = []
    
    # The structure generation of quadcopter
    for _ in range(3):
        line, = ax.plot([], [], [], lw=2)
        lines.append(line)
    
    # The generation of dashed lines at the back of quadcopter
    traj_line, = ax.plot([], [], [], 'b--')
    lines.append(traj_line)
    # ---------------------------------------------------


    # In here, 'anim_callback' function is called with the every 10 ms for 1000 frames
    an = animation.FuncAnimation(fig,
                                 anim_callback,
                                 fargs=(get_world_frame,),
                                 init_func=None,
                                 frames=1000, interval=10, blit=False)

    # This saves the quadcopter simulation result as a gif if it code runned with "python script.py save"
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        print ("saving")
        an.save('sim.gif', dpi=80, writer='imagemagick', fps=60)
    else:
        plt.show()

# Plotting the waypoints of quadcopter
def plot_waypoints(waypoints):
    ax = plt.gca()
    lines = ax.get_lines()
    lines[-2].set_data(waypoints[:,0], waypoints[:,1])
    lines[-2].set_3d_properties(waypoints[:,2])

#Setting the 3d grid dimensions for hovering the quadcopter
def set_limit(x, y, z):
    ax = plt.gca()
    ax.set_xlim(x)
    ax.set_ylim(y)
    ax.set_zlim(z)

def anim_callback(i, get_world_frame):

    # For each frame number 'i' get a corresponding 'frame' and send it to 'set_frame' to draw it.
    frame = get_world_frame(i)
    set_frame(frame)

def set_frame(frame):
    # convert 3x6 world_frame matrix into three line_data objects which is 3x2 (row:point index, column:x,y,z)
    lines_data = [frame[:,[0,2]], frame[:,[1,3]], frame[:,[4,5]]]
    
    global history, count

    # It is updating the structure lines of quadcopter
    for line, line_data in zip(lines[:3], lines_data):
        x, y, z = line_data
        line.set_data(x, y)
        line.set_3d_properties(z)

    
    # plot history trajectory
    history[count] = frame[:,4]
    if count < np.size(history, 0) - 1:
        count += 1
    
    t_history.append(p_data['time'][-1])
    
    # Extracting the trajectory history in all three dimensions
    zline = history[:count,-1]
    xline = history[:count,0]
    yline = history[:count,1]
    
    # --------------This part added for this project------------
    # Updating the 4th line in "lines" with trajectory road of the quadcopter.
    lines[3].set_data(xline, yline)
    lines[3].set_3d_properties(zline)
    #--------------------------------------------------------


# --------------This part added for this project--------------
# A graph plotting function for just the position of quadcopter
def plot_position_over_time():
    
    # Initalization of some global variables in function
    global history, count, t_history

    # Checking whether if there is any data recorded or not
    if count == 0 or len(t_history) == 0:
        print("No position data recorded!")
        return

    # Initializing time, x, y and z coordinates of the quadcopter
    t = t_history[:count]
    x = history[:count, 0]
    y = history[:count, 1]
    z = history[:count, -1]

    # Building the structure of plot. Like adding each coordinate variable of quadcopter, 
    # label of x and y axes and title of graph
    plt.figure(figsize=(8,6))
    plt.plot(t,x, label='X Position')
    plt.plot(t,y, label='Y Position')
    plt.plot(t,z, label='Z Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Quadcopter Position vs Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
# -------------------------------------------------------------