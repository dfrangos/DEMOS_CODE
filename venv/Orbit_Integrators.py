import numpy as np
from Atmospheric_Model import density
from Solar_System import Julian_Date
from Solar_System import Sun_Position
from Solar_System import Moon_Position
sqrt=np.sqrt
J2 = 0.0010826269
R_e = 6378.137e3 # (meters)
G =6.674e-11
mu = 3.986004415e14  # (m^3/s^2)
mu_moon = 4.90486959e12 # (m^3/s^2)
mu_sun = 1.32712440018e20 # (m^3/s^2)
Cd = 2.0 #Coefficient of Drag (Unitless)
A= 1.0 #(Square Meter)
m = 20 #Mass of the Spacecraft (Kg)
w_Earth=np.array([0,0,2*np.pi/(24*60*60)]) #Angular Velocity of Earth in Radians per sec
B=(Cd*A)/m #The Ballistic coefficient of your sat
M_sun= 1.989100e30 #kg

def Two_Body (t,X):

    x,y,z,vx,vy,vz = X

    r3=(sqrt(x**2+y**2+z**2))**3

    ax = -((mu*x)/r3)
    ay = -((mu * y) / r3)
    az = -((mu * z) / r3)
    X_dot=np.array([vx,vy,vz,ax,ay,az])

    return X_dot

def Two_Body_Moon(t, X):
    x, y, z, vx, vy, vz = X

    r3 = (sqrt(x ** 2 + y ** 2 + z ** 2)) ** 3

    ax = -((mu_moon * x) / r3)
    ay = -((mu_moon * y) / r3)
    az = -((mu_moon * z) / r3)
    X_dot = np.array([vx, vy, vz, ax, ay, az])

    return X_dot

def Two_Body_J2 (t,X):

    x,y,z,vx,vy,vz = X

    r3=(sqrt(x**2+y**2+z**2))**3

    ax = -((mu*x)/r3)
    ay = -((mu * y) / r3)
    az = -((mu * z) / r3)

    r_mag=sqrt(x ** 2 + y ** 2 + z ** 2)

    ax_J2 = -(3*J2*mu*R_e**2*x) / (2 * r_mag ** 5) * (1 - ((5 * z ** 2) / (r_mag ** 2)))
    ay_J2 = -(3*J2*mu*R_e**2*y) / (2 * r_mag ** 5) * (1 - ((5 * z ** 2) / (r_mag ** 2)))
    az_J2 = -(3*J2*mu*R_e**2*z) / (2 * r_mag ** 5) * (3 - ((5 * z ** 2) / (r_mag ** 2)))
    X_dot=np.array([vx,vy,vz,ax+ax_J2,ay+ay_J2,az+az_J2])

    return X_dot

def Two_Body_J2_Drag (t,X):

    x,y,z,vx,vy,vz = X

    r3=(sqrt(x**2+y**2+z**2))**3

    ax = -((mu*x)/r3)
    ay = -((mu * y) / r3)
    az = -((mu * z) / r3)

    r_mag=sqrt(x ** 2 + y ** 2 + z ** 2)

    ax_J2 = -(3*J2*mu*R_e**2*x) / (2 * r_mag ** 5) * (1 - ((5 * z ** 2) / (r_mag ** 2)))
    ay_J2 = -(3*J2*mu*R_e**2*y) / (2 * r_mag ** 5) * (1 - ((5 * z ** 2) / (r_mag ** 2)))
    az_J2 = -(3*J2*mu*R_e**2*z) / (2 * r_mag ** 5) * (3 - ((5 * z ** 2) / (r_mag ** 2)))

    #Calculating what the relative velocity of your satellite is
    # with respect to the motion of the earths atmosphere
    vx_rel = (vx + w_Earth[2] * y)
    vy_rel = (vy - w_Earth[2] * x)
    vz_rel = (vz)

    #The magnitude
    v_rel  = sqrt(vx_rel**2+vy_rel**2+vz_rel**2)
    v_rel = X[3:]-np.cross(w_Earth,X[0:3])

    #Getting the acceleration casued by the drag of the atmo
    ax_Drag = -.5 * B * density(r_mag) * v_rel * (vx_rel)
    ay_Drag = -.5 * B * density(r_mag) * v_rel * (vy_rel)
    az_Drag = -.5 * B * density(r_mag) * v_rel * (vz_rel)

    a_drag = -.5 * B * density(r_mag) * v_rel * np.linalg.norm(v_rel)**2/np.linalg.norm(v_rel)
    ax_Drag, ay_Drag, az_Drag= a_drag

    X_dot=np.array([vx,vy,vz,ax+ax_J2+ax_Drag,ay+ay_J2+ay_Drag,az+az_J2+az_Drag])

    return X_dot

def N_Body_J2_Drag (t,X, JD):
    x, y, z, vx, vy, vz = X

    r3 = (sqrt(x ** 2 + y ** 2 + z ** 2)) ** 3
    r_mag = sqrt(x ** 2 + y ** 2 + z ** 2)
    ax = -((mu * x) / r3)
    ay = -((mu * y) / r3)
    az = -((mu * z) / r3)

#_________________________________________
    # Acceleration Casued by the sun

    #Calling the Solar System file to get the position vector at the starting time
    r_sun = Sun_Position(JD+(t/86400))
    #Breaking that up into components
    r_sun_x = r_sun[0]
    r_sun_y = r_sun[1]
    r_sun_z = r_sun[2]
    #Getting the magnitude
    r_sun_mag= (sqrt(r_sun[0] ** 2 + r_sun[1] ** 2 + r_sun[2] ** 2))

    #Creating the position vector from the sat to the sun.
    r_sat_sun_x = r_sun_x - x
    r_sat_sun_y = r_sun_y - y
    r_sat_sun_z = r_sun_z - z
    r_sat_sun = np.array([r_sat_sun_x, r_sat_sun_y, r_sat_sun_z])
    r_sat_sun_mag=(sqrt(r_sat_sun[0] ** 2 + r_sat_sun[1] ** 2 + r_sat_sun[2] ** 2))


    # Calculating Acceleration components Casued by the sun
    a_sun_x = mu_sun * ((r_sat_sun_x / (r_sat_sun_mag ** 3)) - (r_sun_x / (r_sun_mag ** 3)))
    a_sun_y = mu_sun * ((r_sat_sun_y / (r_sat_sun_mag ** 3)) - (r_sun_y / (r_sun_mag ** 3)))
    a_sun_z = mu_sun * ((r_sat_sun_z / (r_sat_sun_mag ** 3)) - (r_sun_z / (r_sun_mag ** 3)))
    # _________________________________________________________________
    # Accel caused by the moon
    # Calling the Solar System file to get the position vector at the starting time
    r_moon, r_moon_mag = Moon_Position(JD + (t / 86400))
    # Breaking that up into components
    r_moon_x = r_moon[0]
    r_moon_y = r_moon[1]
    r_moon_z = r_moon[2]

    # Creating the position vector from the sat to the moon.
    r_sat_moon_x = r_moon_x - x
    r_sat_moon_y = r_moon_y - y
    r_sat_moon_z = r_moon_z - z
    r_sat_moon = np.array([r_sat_moon_x, r_sat_moon_y, r_sat_moon_z])
    r_sat_moon_mag = (sqrt(r_sat_moon[0] ** 2 + r_sat_moon[1] ** 2 + r_sat_moon[2] ** 2))

    # Calculating Acceleration components Casued by the moon
    a_moon_x = mu_moon * ((r_sat_moon_x / (r_sat_moon_mag ** 3)) - (r_moon_x / (r_moon_mag ** 3)))
    a_moon_y = mu_moon * ((r_sat_moon_y / (r_sat_moon_mag ** 3)) - (r_moon_y / (r_moon_mag ** 3)))
    a_moon_z = mu_moon * ((r_sat_moon_z / (r_sat_moon_mag ** 3)) - (r_moon_z / (r_moon_mag ** 3)))

    #_________________________________________________________________
    # Calculating Acceleration components Casued by the J2 Pertubation
    ax_J2 = -(3 * J2 * mu * R_e ** 2 * x) / (2 * r_mag ** 5) * (1 - ((5 * z ** 2) / (r_mag ** 2)))
    ay_J2 = -(3 * J2 * mu * R_e ** 2 * y) / (2 * r_mag ** 5) * (1 - ((5 * z ** 2) / (r_mag ** 2)))
    az_J2 = -(3 * J2 * mu * R_e ** 2 * z) / (2 * r_mag ** 5) * (3 - ((5 * z ** 2) / (r_mag ** 2)))
    # _________________________________________________________________
    #Accel caused by drag
    # Calculating what the relative velocity of your satellite is
    # with respect to the motion of the earths atmosphere
    vx_rel = (vx + w_Earth[2] * y)
    vy_rel = (vy - w_Earth[2] * x)
    vz_rel = (vz)

    # The magnitude
    v_rel = sqrt(vx_rel ** 2 + vy_rel ** 2 + vz_rel ** 2)
    v_rel = X[3:] - np.cross(w_Earth, X[0:3])

    # Getting the acceleration casued by the drag of the atmo
    ax_Drag = -.5 * B * density(r_mag) * v_rel * (vx_rel)
    ay_Drag = -.5 * B * density(r_mag) * v_rel * (vy_rel)
    az_Drag = -.5 * B * density(r_mag) * v_rel * (vz_rel)

    a_drag = -.5 * B * density(r_mag) * v_rel * np.linalg.norm(v_rel) ** 2 / np.linalg.norm(v_rel)
    ax_Drag, ay_Drag, az_Drag = a_drag


    # Including all calculated accelerations into the state vector.
    X_dot = np.array([vx, vy, vz, ax + a_sun_x+a_moon_x+ax_J2+ax_Drag, ay + a_sun_y+a_moon_y+ay_J2+ay_Drag, az + a_sun_z+a_moon_z+az_J2+az_Drag])
    #X_dot = np.array([vx, vy, vz, ax + a_sun_x*0 + a_moon_x*0 + ax_J2*0 + ax_Drag*0, ay + a_sun_y*0 + a_moon_y*0 + ay_J2*0 + ay_Drag*0,
                      #az + a_sun_z*0 + a_moon_z*0 + az_J2*0 + az_Drag*0])
    return X_dot