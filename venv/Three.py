import time
import math
import numpy as np
import pylab as py
import matplotlib.pyplot as plt
import scipy
import random as rand
import mpl_toolkits.mplot3d.axes3d as p3
import networkx as nx
from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(1)
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
# Gives each particle a mass and a Random starting position and velocity. (m) and (m/s)
def Create(N, state, mass):
    #mass[0,0]=5e27    #if you activate this it'll let you have the first body be any particular value you want.
    mass[0,:]=5e22  #Every row in the mass array is 2   #1.67492747121e-27 #Mass of a neutron
    for i in range(N): #begin iterating i to 5
        for j in range(0, 3):  #Begin iterating j 0 to 2 for the position
            state[i, j] = np.random.uniform(-3.84e7, 3.84e7) #(m)
        for j in range(3, 6):  # Begin iterating j 3 to 5 for the velocity
            state[i,j]=np.random.uniform(-380, 300) #+rand.random())*10**(-3) Generates random velos for the particles in m/s
    #state[0,:]=[0,0,0,0,0,0]  #if you activate this it'll let you have the first body be any particular value you want.
    return state, mass

def Get_Accel(N,state,mass,accel):
    accel=np.zeros((N,3))
    for i in range(N):
        for j in range(N):
            if i != j:
                #Calculates the distance of a particular mass to another in each dimension.
                dx=state[i,0]-state[j,0]
                dy=state[i,1]-state[j,1]
                dz=state[i,2]-state[j,2]
                Dist=np.array([dx,dy,dz])
                Dist_Mag=np.sqrt((dx**2+dy**2+dz**2+Soft**2))
                ##Calculates the acceleration induced on any particular pair of bodies
                accel[i,:]-=G*mass[0,j]*np.array([(Dist[0])/(Dist_Mag**3),(Dist[1])/(Dist_Mag**3),(Dist[2])/(Dist_Mag**3)])
    return accel

def Update_State(N,state,accel,dt,mass):
    for i in range(N):
        state[i, 3:] = state[i, 3:] + ((dt/2)*accel[i,:]) #Updating the velocity based on the acceleration
        state[i, :3] = state[i, :3] + (dt) * state[i, 3:]  # Updating the position based on the velo
        accel = Get_Accel(N,state,mass,accel)
        state[i, 3:] = state[i, 3:] + ((dt/2)*accel[i,:])  # Updating the velocity based on the acceleration
    return state

def Total_Energy(n, state, mass, index):
    #Calculate the kinetic Energy
    KE=np.zeros((n,1))
    for i in range(n):
        KE[i,0]=mass[0,i]*((np.linalg.norm(state[i,3:,index]))**2)*.5
    Kinetic_Energy=np.sum(KE)
    # Calculate the Potential Energy
    PE = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                dx=state[i,0,index]-state[j,0,index]
                dy=state[i,1,index]-state[j,1,index]
                dz=state[i,2,index]-state[j,2,index]
                Dist=np.array([dx,dy,dz])
                Dist_Mag=np.sqrt((dx**2+dy**2+dz**2))
                PE+=G*mass[0,i]*mass[0,j]/Dist_Mag
    Total=Kinetic_Energy-PE
    return Total

################_______________________________________########################
G=6.67408e-11
AU=1.496e11 #m
#The number of bodies in the system
N=5
#Creating an Empty State Vector (x,y,z,vx,vy,vz)
State=np.zeros((N,6))  # (0,0,0,0,0,0)
#State=np.array([[0,0,0,0,0,0],[.41*AU,0,0,0,47.9e3,0],[.74*AU,0,0,0,35e3,0],[1.1*AU,0,0,0,29.8e3,0],[1.6*AU,0,0,0,24.1e3,0]])
print(State)
Mass=np.zeros((1,N))
#Mass=np.array([[1.989e30,0.33e24,4.87e24,5.97e24,0.642e28]])
#Creates an empty acceleration vector (ax,ay,az)
Accel=np.zeros((N,3))
#Time
t=800
DT=500
#Defining Softening factor
Soft=6378e3
#Defining my storage area for my position values
State_Store=np.zeros((N,6,t))

State, Mass = Create(N,State,Mass)
fig1 = plt.figure()
ax1 = Axes3D(fig1, auto_add_to_figure=False)
fig1.add_axes(ax1)


Accel = Get_Accel(N,State,Mass,Accel)
for k in range(t):
    # print('State Vector:  Pos x,y,z (m)   Vel x,y,z (m/s)')
    # print(State)
    State = Update_State(N,State,Accel,DT,Mass)
    # print('Acceleration: x,y,z (m/s/s)')
    # print(Accel)
    State_Store[:,:,k]=State[:,0:]

#for i in range(N):
   # plot=ax1.plot3D(State_Store[i, 0,:], State_Store[i, 1,:], State_Store[i, 2,:],"o")

Initial_Energy=Total_Energy(N, State_Store, Mass, 0)
Final_Energy=Total_Energy(N, State_Store, Mass, -1)
print(Initial_Energy)
print(Final_Energy)
#This is the animation stuff______________________________
# ANIMATION FUNCTION
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

anim = FuncAnimation(fig1, func, frames=t, repeat=True, interval=2, fargs=(State_Store, line, N))
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
A=4
print(A)




