import time
import math
import numpy as np
import pylab as py
import matplotlib.pyplot as plt
import scipy
import random as rand
import mpl_toolkits.mplot3d.axes3d as p3
import networkx as nx
import PySimpleGUI as sg
from N_Body_Functions import *
from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from future.moves import tkinter as Tkinter
np.random.seed(1)
Save_to_File="Off"
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
################_______________________________________########################

#Defining my layout condition for the window creation. This is where you setup word prompts and buttons:
sg.theme('DarkAmber')
Format1= [[sg.Text("Please Choose Your Scenario!")],[sg.Button("Random")],[sg.Button("Solar System")],[sg.Button("Cancel")]]
#Creates the window with the layout condition.
Screen1=sg.Window("Demokritos' Game of Gravitation",Format1,margins=(150,70))
#Creating an event loop

while True:
    event_series1,values=Screen1.read()
    if event_series1 == "Random":
        Format2 =  [ [sg.Text('Would you like to Customize or Run the Default?')],
                    [sg.Button("Customize")], [sg.Button("Default")], [sg.Button("Cancel")]]
        Screen2 = sg.Window("Demokritos' Game of Gravitation", Format2, margins=(150, 70))
        event_series2, values = Screen2.read()
        if event_series2 == "Default":
            t = 2000  # How many iterations
            DT = 300  # Your delta T jumps
            N = 6
            Mass = 5e28
            Pos_Bound = 3.84e7
            Vel_Bound = 300
            # Defining my storage area for my position values
            State, Mass, Soft = Create_Random(N, Mass, Pos_Bound, Vel_Bound)
            break
        elif event_series2 == "Customize":
            Format3 =  [ [sg.Text('Setup Your Parameters:')],
                        [sg.Text('How Many Bodies Would You Like to Simulate?'),                 sg.Input(key='-N-')],
                        [sg.Text('What Positional Bounds Would You Like to Use?  -\+ (m)'),      sg.Input(key='-Pos_Bound-')],
                        [sg.Text('What Velocity Bounds Would You Like to Use?  -\+(m/s)'),       sg.Input(key='-Vel_Bound-')],
                        [sg.Text('What are the Masses of the Bodies? (kg)'),                     sg.Input(key='-Mass-')],
                        [sg.Text('How Many Iterations Would You Like to Execute?'),              sg.Input(key='-t-')],
                        [sg.Text('How Far in Time Would You Like to Jump For Each Iteration?'),  sg.Input(key='-DT-')],
                        [sg.Button("Okay")], [sg.Button("Cancel")]]
            Screen3 = sg.Window("Demokritos' Game of Gravitation", Format3, margins=(150, 70))
            event_series3, values = Screen3.read()
            if event_series3=='Okay':
                N         = int(values['-N-'])
                Pos_Bound = int(values['-Pos_Bound-'])
                Vel_Bound = int(values['-Vel_Bound-'])
                Mass      = int(values['-Mass-'])
                t         = int(values['-t-'])
                DT        = int(values['-DT-'])
                print(type(N))
                State, Mass, Soft = Create_Random(N, Mass, Pos_Bound, Vel_Bound)
                break
            if event_series3=="Cancel" or event_series3 == sg.WIN_CLOSED:
                break
        elif event_series2 == "Cancel" or event_series2 == sg.WIN_CLOSED:
            break
        break
    elif event_series1 == "Solar System":
        t = 2000  # How many iterations
        DT = 36000  # Your delta T jumps
        N = 6
        State, Mass, Soft = Create_Solar_System()
        break
    elif event_series1 =="Cancel" or event_series1 == sg.WIN_CLOSED:

        break

#Window=sg.Window(title="Scenario Boot Up",layout=[[]],margins=(300,150)).read()

#Scenario_Type=input("Which Scenario Would you like to Run? R: Random, S: Solar System\n")
#if Scenario_Type=="R":
    #t = 2000  # How many iterations
    #DT = 300  # Your delta T jumps
    #N=6
    #Mass=5e28
    #Pos_Bound=3.84e7
    #Vel_Bound=300
    #Defining my storage area for my position values
    #State, Mass, Soft = Create_Random(N,Mass,Pos_Bound,Vel_Bound)
#elif Scenario_Type=="S":
   # t = 2000  # How many iterations
   # DT = 36000  # Your delta T jumps
 #   N = 6
   # State, Mass, Soft = Create_Solar_System()
#else: print("Not a Valid Entry!")

State_Store=np.zeros((N,6,t))
Accel = Get_Accel(N,State,Mass,Soft)
for k in range(t):
    State = Update_State(N,State,Accel,DT,Mass,Soft)
    State_Store[:,:,k]=State[:,0:]

#Saving the State Store Information to a File
    while Save_to_File=="On":
        write_to_file = True
        filename = 'n_body_dat_' + str(N) + Scenario_Type + '.npy'
        if write_to_file:
            with open(filename, 'wb') as f:
                np.save(f, State_Store)

Initial_Energy=Total_Energy(N, State_Store, Mass, 0)
Final_Energy=Total_Energy(N, State_Store, Mass, -1)
print(Initial_Energy)
print(Final_Energy)
#This is the animation stuff______________________________
# ANIMATION FUNCTION

fig1 = plt.figure()
ax1 = Axes3D(fig1, auto_add_to_figure=False)
fig1.add_axes(ax1)
def func(num, dataSet, line, N):
    # NOTE: there is no .set_data() for 3 dim data...
    for i in range(N):
        line[i].set_data(dataSet[i, 0:2, :num])
        line[i].set_3d_properties(dataSet[i, 2, :num])
    set_axes_equal(ax1)
    return line

line=[]

for i in range(N):
    line.append(plt.plot(State_Store[i,0,0], State_Store[i,1,0], State_Store[i,2,0],marker=".")[0])

anim = FuncAnimation(fig1, func, frames=t, repeat=True, interval=1, fargs=(State_Store, line, N))
#anim.save('rgb_cube.gif', dpi=80, writer='imagemagick', fps=24)
#___________________________________________________
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
ax1.set_zlabel("z (m)")
ax1.set_title("Orbital Trjectory")
ax1.grid()
ax1.legend(["Body 1", "Body 2","Body 3", "Body 4", "Body 5"])
print(State_Store[:,3:,:].max())
plt.show()




#9000, 900, 5, 5e22, -3.84e7...   -100..... Soft=1e2
#for i in range(N):
   # plot=ax1.plot3D(State_Store[i, 0,:], State_Store[i, 1,:], State_Store[i, 2,:],"o")




