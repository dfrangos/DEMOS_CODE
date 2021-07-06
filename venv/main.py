import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from Coordinate_Transform import Kep_Peri
from Coordinate_Transform import IJKPQW
from Coordinate_Transform import Peri_Inert
from Orbit_Integrators import *
from Solar_System import Julian_Date

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

#This is declaring what your state vector is in terms of keplarian elements.
a,e,i,ran,w,theta,mu = 6100e3+6378e3,.001,45*(np.pi/180),35*(np.pi/180),30*(np.pi/180),180*(np.pi/180),3.986004415e14
if e == 0 and i == 0:
    w = 0
    ran = 0
# This condition is for when you're circular and inclined.
if e == 0 and i > 0:
    w = 0
# This condition is for when you're elliptical and equatorial.
if e > 0 and i == 0:
    ran = 0
#Defining your start time (Date).
Epoch=[2001,4,2,12,10,10]
MD, JD=Julian_Date(Epoch)

#print("Here are your Keplerian Orbital Elements: ","\nSemimajor Axis:",a,"\nEccentricity:",e,"\nInclination: %.2g"%i,"\nRight Accension of Ascending Node:",ran,
      #"\nArgument of Perigee:",w,"\nTrue Anomaly:",theta)
#Invoking the kep peri function to output the state vector in terms
#of a position and velocity vector in perifocal coordinates.
r,v=Kep_Peri(a,e,i,ran,w,theta,mu)
print("This is your position",r,"\nThis is your velocity",v)

#Invoking the IJKPQW function to output the trasformation matrix that will be used to trasform perifocal to inertial.
ijkpqw= IJKPQW(i,ran,w)
print("Here is your Coordinate Transform:","\n",ijkpqw)

r_ijk,v_ijk=Peri_Inert(r,v,ijkpqw)
print("Inertial Position Vector:",r_ijk, "meters","\nInertial Velocity Vector:",v_ijk, "meters/second")

#Setting up the integrator.

t0=0
tf=6000
data_points=1000
t=np.linspace(t0, tf, data_points)
init_cond=r_ijk,v_ijk
init_cond=np.array([init_cond[0][0],init_cond[0][1],init_cond[0][2],init_cond[1][0],init_cond[1][1],init_cond[1][2]])
r= integrate.ode(Two_Body()).set_integrator("dop853")
r.set_initial_value(init_cond,t0).set_f_params()
y=np.zeros((len(t),len(init_cond)))

y[0,:]=init_cond
r_mag=np.zeros((1,len(t)))
r_mag[0,0]=a

for ii in range(1,len(t)):
      y[ii,:]=r.integrate(t[ii])
      r_mag[0,ii]=np.linalg.norm(y[ii,0:3])

#Creating the Altitude plot
fig4=plt.figure()
ax4=plt.axes()
ax4.set_xlabel("Time (Days)")
ax4.set_ylabel("Meters (m)")
ax4.set_title("Altitude (m) vs Time (Days)")
ax4.plot(t/86400,r_mag.T*(1e-3)-6378)
ax4.grid(axis='x')
ax4.grid(axis='y')


#Creating the 2D Position Plot
fig1=plt.figure()
ax1=plt.axes()
ax1.set_xlabel("Time (Seconds)")
ax1.set_ylabel("Meters (m)")
ax1.set_title("XYZ Position (m) vs Time (s)")
ax1.plot(t,y[:,0:3])
ax1.grid(axis='x')
ax1.grid(axis='y')
ax1.legend(["X","Y","Z"])

# #This section is obtaining the state vectors of the moon to include into the 3D plot.
# #This is declaring what your state vector is in terms of keplarian elements as.
# a_moon,e_moon,i_moon,ran_moon,w_moon,theta_moon,mu_moon = 384748e3,.05490,5.15*(np.pi/180),35*(np.pi/180),180*(np.pi/180),180*(np.pi/180),(6.674e-11*(7.34767309e22+5.972e24))
# #Invoking the kep peri function to output the state vector in terms
# #of a position and velocity vector in perifocal coordinates.
# r_moon,v_moon=Kep_Peri(a_moon,e_moon,i_moon,ran_moon,w_moon,theta_moon,mu_moon)
#
# #Invoking the IJKPQW function to output the trasformation matrix that will be used to trasform perifocal to inertial.
# ijkpqw_moon= IJKPQW(i_moon,ran_moon,w_moon)
#
# r_ijk_moon,v_ijk_moon=Peri_Inert(r_moon,v_moon,ijkpqw_moon)

# #Setting up the integrator for the moon.
# t0_moon=0
# tf_moon=1000000
# data_points_moon=1000
# t_moon=np.linspace(t0_moon, tf_moon, data_points_moon)
# init_cond_moon=r_ijk_moon,v_ijk_moon
# init_cond_moon=np.array([init_cond_moon[0][0],init_cond_moon[0][1],init_cond_moon[0][2],init_cond_moon[1][0],init_cond_moon[1][1],init_cond_moon[1][2]])
# r_moon= integrate.ode(Two_Body_Moon).set_integrator("dop853")
# r_moon.set_initial_value(init_cond_moon,t0_moon)
# y_moon=np.zeros((len(t_moon),len(init_cond_moon)))
#
# y_moon[0,:]=init_cond_moon
# for ii in range(1,len(t_moon)):
#       y_moon[ii,:]=r_moon.integrate(t_moon[ii])

#Creating a new 2d plot for the mmon

# fig3=plt.figure()
# ax3=plt.axes()
# ax3.set_xlabel("Time (Seconds)")
# ax3.set_ylabel("Meters (m)")
# ax3.set_title("Moon's XYZ Position (m) vs Time (s)")
# # ax3.plot(t_moon,y_moon[:,0:3])
# ax3.grid(axis='x')
# ax3.grid(axis='y')
# ax3.legend(["X","Y","Z"])


#Creating the 3D Position Plot

fig2=plt.figure()
ax2=plt.axes(projection = '3d')
ax2.plot3D(y[:,0],y[:,1],y[:,2], linewidth=.3)
#ax2.plot3D(y_moon[:,0],y_moon[:,1],y_moon[:,2])
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")
ax2.set_zlabel("z (m)")
ax2.set_title("Orbital Trjectory")
ax2.grid()
ax2.legend(["Sat1","Moon"])
set_axes_equal(ax2)
plt.show()

