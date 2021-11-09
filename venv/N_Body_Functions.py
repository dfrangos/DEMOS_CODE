import time
import math
import numpy as np
import Constants as C
from Coordinate_Transform import *
import scipy
import matplotlib
import tkinter

#Just doing some short hand stuff for SOCAHTOA
sin=np.sin
cos=np.cos
tan=np.tan
sqrt=np.sqrt

M_sun=1.988473e30# kg
M_Earth = 5.972e24
M_Moon = 7.34767309e22
AU=149597870700 #m
State_Norm=1#AU
Mass_Norm=1#M_sun
G = ((6.67408e-11) / (State_Norm ** 3)) * Mass_Norm

#Your Scenario Functions________________________________________________________________________________________________
def Create_Random(N,Mass,Pos_Bound,Vel_Bound):
    state = np.zeros((N, 6)) # Creating an Empty State Vector (x,y,z,vx,vy,vz), (0,0,0,0,0,0)
    mass = np.zeros((1, N)) # Creating an Empty Mass Vector (0,0,0,0....)
    soft = 6378e3 # Defining Softening factor
    mass[0,:]=Mass  #Every value in the mass array is:
    for i in range(N): #begin iterating i to N
        for j in range(0, 3):  #Begin iterating j 0 to 2 for the position values of the state vector
            state[i, j] = np.random.uniform(-Pos_Bound, Pos_Bound) #(m) random creation of position values between...
        for j in range(3, 6):  # Begin iterating j 3 to 5 for the velocity
            state[i,j]=np.random.uniform(-Vel_Bound, Vel_Bound) #+rand.random())*10**(-3) Generates random velos for the particles in m/s
    mass[0,:]=5e22  #if you activate this it'll let you have the first body be any particular value you want.
    #state=[0,0,0,0,0,0]  #if you activate this it'll let you have the first body be any particular value you want.
    return state, mass, soft

def Create_Solar_System(N,JD):
    #Defining Softening factor
    soft = 6378e5/State_Norm
    state=np.zeros((N,6))
    mass=np.ones((1,N))
    mass=mass*Mass_Norm
    for i in range(1,N):
        state[i,:], mass[0,i] = any_planet(JD, i, "")
    #state=np.array([[0,0,0,0,0,0],[(.41*AU),0,0,0,47.9e3,0],[.74*AU,0,0,50,35e3,-150],[1.0*AU,.02*AU,.01*AU,100,29.8e3,-100]])/AU

    return state, mass, soft

def Create_Mars_System(N):
    state = np.zeros((N, 6))  # Creating an Empty State Vector (x,y,z,vx,vy,vz), (0,0,0,0,0,0)
    mass = np.zeros((1, N))  # Creating an Empty Mass Vector (0,0,0,0....)
    soft = 100  # Defining Softening factor
    mass  = np.array([[C.C["Mars"]["Mass"],C.C["Phobos"]["Mass"],C.C["Deimos"]["Mass"],C.C["Craft1"]["Mass"]]])  #if you activate this it'll let you have the first body be any particular value you want.
    #mass=mass.T
    #mass=np.transpose(mass,(1,3))
    print(np.shape(mass))
    state = np.array([[0,0,0,0,0,0],[8227315.12407213,4282153.56786034,-170726.75882666,-977.28363949,1926.43920718,-8.28578831],[11485996.03326884,20455037.49258747,425987.6488949,-1177.95580571,660.73160754,34.42792028],[8227315.12407213*1.3,4282153.56786034*1.1,-170726.75882666,-977.28363949,1926.43920718,-8.28578831]])  #if you activate this it'll let you have the first body be any particular value you want.
    return state, mass, soft

def Create_Earth_Moon_System(N):
    state = np.zeros((N, 6))  # Creating an Empty State Vector (x,y,z,vx,vy,vz), (0,0,0,0,0,0)
    mass = np.zeros((1, N))  # Creating an Empty Mass Vector (0,0,0,0....)
    soft = 100  # Defining Softening factor
    mass  = np.array([[C.C["Earth"]["Mass"],C.C["Moon"]["Mass"],C.C["Craft1"]["Mass"]]])  #if you activate this it'll let you have the first body be any particular value you want.
    Epoch1 = [1994, 4, 28, 12, 0, 0]
    MD, JD1 = Julian_Date(Epoch1)
    Moon_Position1, Radius_Moon_Mag1 = Moon_Position(JD1)
    Epoch2 = [1994, 4, 28, 12, 0, 1]
    MD, JD2 = Julian_Date(Epoch2)
    Moon_Position2, Radius_Moon_Mag2 = Moon_Position(JD2)
    Moon_Velocity=(Moon_Position2-Moon_Position1)
    Moon_State_Vector=np.ndarray.tolist(np.hstack([Moon_Position1,Moon_Velocity]))
    print(Moon_State_Vector)
    Moon_Keplerian, e_vec_Moon = Inert_Kep(Moon_State_Vector, C.C["Earth"]["Mu"])
    Craft1_State_Vector=Kep_Inert(150e3+6378e3,0.0001,(Moon_Keplerian[2]),(Moon_Keplerian[3]),(Moon_Keplerian[4]),np.pi/2,C.C["Earth"]["Mu"]) #[a,e,i,w,ran,f,M,E]
    state = np.array([[0, 0, 0, 0, 0, 0],Moon_State_Vector,Craft1_State_Vector])
    return state, mass, soft

def Create_TRAPIST_1(N):
    state = np.zeros((N, 6))  # Creating an Empty State Vector (x,y,z,vx,vy,vz), (0,0,0,0,0,0)
    mass = np.zeros((1, N))  # Creating an Empty Mass Vector (0,0,0,0....)
    soft = 1988500e24  # Defining Softening factor
    mass = np.array([[C.C["Body1"]["Mass"], C.C["Body2"]["Mass"],C.C["Body3"]["Mass"],C.C["Body4"]["Mass"]]])*Mass_Norm  # if you activate this it'll let you have the first body be any particular value you want.
    state = np.array([[1*AU, 1.2*AU, .02*AU, -1e3, -.3e3, -.02e3], [-.01*AU, .09*AU, -.03*AU, 2e3, .5e3, .02e3],[-15*AU, 1*AU, .2*AU, -.8e3, .4e3, -.02e3],[-550*AU,0,0,0.005e3,4e3,.1e3]])/State_Norm  # if you activate this it'll let you have the first body be any particular value you want.
    return state, mass, soft

def Create_The_Jovian_System(N):
    state = np.zeros((N, 6))  # Creating an Empty State Vector (x,y,z,vx,vy,vz), (0,0,0,0,0,0)
    mass = np.zeros((1, N))  # Creating an Empty Mass Vector (0,0,0,0....)
    soft = 1988500e24  # Defining Softening factor
    mass = np.array([[C.C["Body1"]["Mass"], C.C["Body2"]["Mass"],C.C["Body3"]["Mass"],C.C["Body4"]["Mass"]]])*Mass_Norm  # if you activate this it'll let you have the first body be any particular value you want.
    state = np.array([[1*AU, 1.2*AU, .02*AU, -1e3, -.3e3, -.02e3], [-.01*AU, .09*AU, -.03*AU, 2e3, .5e3, .02e3],[-15*AU, 1*AU, .2*AU, -.8e3, .4e3, -.02e3],[-550*AU,0,0,0.005e3,4e3,.1e3]])/State_Norm  # if you activate this it'll let you have the first body be any particular value you want.
    return state, mass, soft

#Propegation Functions#_______________________________________________________________________________________________

def Get_Accel(N,state,mass,soft):
    accel=np.zeros((N,3))
    for i in range(N):
        for j in range(N):
            if i != j:
                #Calculates the distance of a particular mass to another in each dimension.
                dx=state[i,0]-state[j,0]
                dy=state[i,1]-state[j,1]
                dz=state[i,2]-state[j,2]
                dist=np.array([dx,dy,dz])
                dist_mag=np.sqrt((dx**2+dy**2+dz**2+soft**2))
                ##Calculates the acceleration induced on any particular pair of bodies
                accel[i,:]-=G*mass[0,j]*np.array([(dist[0])/(dist_mag**3),(dist[1])/(dist_mag**3),(dist[2])/(dist_mag**3)])

    return accel

def Update_DT(n,state):
    relative_dist_state = []
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculates the distance of a particular mass to another in each dimension.
                dx = state[i, 0] - state[j, 0]
                dy = state[i, 1] - state[j, 1]
                dz = state[i, 2] - state[j, 2]
                dist = np.array([dx, dy, dz])
                dist_mag_no_soft = np.linalg.norm(dist)
                # Adds that distance into the Relative Dist State array
                relative_dist_state.append(dist_mag_no_soft)
    # Base altitude (m) [h_0]
    dist_list = [150e3,300e3,600e3,1200e3,10e6,30e6,70e6,150e6,300e6,500e6,750e6,1.5e9]
    dt_list= [10   ,25   ,50   ,120   ,170 ,220 ,320 ,420  ,550  ,680  ,800  ,2500]
    z=np.min(relative_dist_state)-C.C["Earth"]["Radius"]
    # error check
    if z > dist_list[len(dist_list)-1]:
        z = dist_list[len(dist_list)-1]
    elif z < dist_list[0]:
        z = dist_list[0]
    ii = 0
    for jj in range(0, int(len(dist_list))):
        if dist_list[jj] <= z < dist_list[jj + 1]:
            ii = jj

    if z >= dist_list[len(dist_list)-1]:
        ii = int(dist_list[len(dist_list)-1])
    dt=dt_list[int(ii)]
    print(dt)

    return dt

def Update_State(n,state,accel,dt,mass,soft,flag):

#Flag's Structure
    burn_flag,target_body,dv_mag,origin_body,normal_direction,velocity_direction=flag

    if flag[0]==0:
        for i in range(n):
            accel = Get_Accel(n, state, mass, soft)
            state[i, 3:] = state[i, 3:] + ((dt)*accel[i,:]) #Updating the velocity based on the acceleration
            state[i, :3] = state[i, :3] + (dt) * state[i, 3:]  # Updating the position based on the velo
            # accel        = Get_Accel(n,state,mass,soft)
            # state[i, 3:] = state[i, 3:] + ((dt/2)*accel[i,:])  # Updating the velocity based on the acceleration
    elif flag[0]==1:
        relative_state         = state[target_body,:]-state[origin_body,:]
        r_mag                  = np.linalg.norm(relative_state[:3])
        r_unit                 = relative_state[:3]/r_mag
        v_mag                  = np.linalg.norm(relative_state[3:])
        v_unit                 = relative_state[3:] / v_mag
        h_unit                 = np.cross(r_unit,v_unit)
        dv_vec_unit            = (v_unit * velocity_direction + h_unit * normal_direction)/np.linalg.norm(v_unit * velocity_direction + h_unit * normal_direction)
        relative_state[3:]    += dv_vec_unit*dv_mag
        state[target_body, 3:] = relative_state[3:]+state[origin_body,3:]

        for i in range(n):
            accel = Get_Accel(n, state, mass, soft)
            state[i, 3:] = state[i, 3:] + ((dt) * accel[i, :])  # Updating the velocity based on the acceleration
            state[i, :3] = state[i, :3] + (dt) * state[i, 3:]  # Updating the position based on the velo
            # accel = Get_Accel(n, state, mass, soft)
            # state[i, 3:] = state[i, 3:] + ((dt / 2) * accel[i, :])  # Updating the velocity based on the acceleration
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

def Any_Planet(JD, Planet, Frame):
    T = (JD - 2451545.0) / 36525
    deg2rad = np.pi / 180

    mu_s = 1.32712440017987e20

    coeff = Ephemeride_Coeff(Planet)

    Tvec = np.array([1, T, T ** 2, T ** 3])

    # Mean longitude of Planet
    L = coeff.L @ Tvec * deg2rad

    # Semimajor axis of the orbit
    a = coeff.a @ Tvec * AU

    # Eccentricity of the orbit
    ecc = coeff.e @ Tvec

    # Inclination of the orbit
    inc = coeff.i @ Tvec * deg2rad

    # Longitude of the Ascending Node
    W = coeff.W @ Tvec * deg2rad

    # Longitude of the Perihelion
    P = coeff.P @ Tvec * deg2rad

    # Argument of perihelion
    w = P - W

    # Mean anomaly of orbit
    M = L - P

    # True anomaly of orbit
    Ccen = (2 * ecc - ecc ** 3 / 4 + 5 / 96 * ecc ** 5) * np.sin(M) + (
                5 / 4 * ecc ** 2 - 11 / 24 * ecc ** 4) * np.sin(
        2 * M) + (13 / 12 * ecc ** 3 - 43 / 64 * ecc ** 5) * np.sin(3 * M) + 103 / 96 * ecc ** 4 * np.sin(
        4 * M) + 1097 / 960 * ecc ** 5 * np.sin(5 * M)

    nu = M + Ccen
    kep = [a, ecc, inc, W, w, nu]
    r_pqw, v_pqw = Kep_Peri(a, ecc, inc, W, w, nu,mu_s)
    ijkpqw = IJKPQW(inc, W, w)
    r_ijk, v_ijk=Peri_Inert(r_pqw, v_pqw, ijkpqw)
    rv=np.hstack([r_ijk,v_ijk])

    # Convert to EME2000 if necessary
    # if Frame in 'EME2000':
    #     theta = 23.4393 * np.pi / 180
    #     C = np.matrix([[1, 0, 0],
    #                    [0, cos(theta), - sin(theta)],
    #                    [0, sin(theta), cos(theta)]])
    #     R = C @ R
    #     V = C @ V

    mu_p = coeff.mu_p
    return rv/State_Norm, coeff.mass/Mass_Norm

def Ephemeride_Coeff(Planet):
    class PlanetEphem:
        pass

    coeff = PlanetEphem()
    if Planet == 1:
        # Venus
        L = [252.250906, 149472.6746358, -0.00000535, 0.000000002]
        a = [0.387098310, 0.0, 0.0, 0.0]
        e = [0.20563175, 0.000020406, -0.0000000284, -0.00000000017]
        i = [7.004986, -0.0059516, 0.00000081, 0.000000041]
        W = [48.330893, -0.1254229, -0.00008833, -0.000000196]
        P = [77.456119, 0.1588643, -0.00001343, 0.000000039]
        mu_p = 2.20320804864179e4
        mass = 3.3011e23
    elif Planet == 2:
        # Venus
        L = [181.979801, 58517.8156760, 0.00000165, -0.000000002]
        a = [0.72332982, 0.0, 0.0, 0.0]
        e = [0.00677188, -0.000047766, 0.0000000975, 0.00000000044]
        i = [3.394662, -0.0008568, -0.00003244, 0.000000010]
        W = [76.679920, -0.2780080, -0.00014256, -0.000000198]
        P = [131.563707, 0.0048646, -0.00138232, -0.000005332]
        mu_p = 3.2485859882646e5
        mass= 4.8675e24
    elif Planet == 3:
        # Earth
        L = [100.466449, 35999.3728519, -0.00000568, 0.0]
        a = [1.000001018, 0.0, 0.0, 0.0]
        e = [0.01670862, - 0.000042037, -0.0000001236, 0.00000000004]
        i = [0.0, 0.0130546, - 0.00000931, -0.000000034]
        W = [174.873174, - 0.2410908, 0.00004067, -0.000001327]
        P = [102.937348, 0.3225557, 0.00015026, 0.000000478]
        mu_p = 3.98600432896939e5
        mass = 5.97237e24
    elif Planet == 4:
        # Mars
        L = [355.433275, 19140.2993313, 0.00000261, -0.000000003]
        a = [1.523679342, 0.0, 0.0, 0.0]
        e = [0.09340062, 0.000090483, -0.0000000806, -0.00000000035]
        i = [1.849726, - 0.0081479, -0.00002255, -0.000000027]
        W = [49.558093, - 0.2949846, -0.00063993, -0.000002143]
        P = [336.060234, 0.4438898, -0.00017321, 0.000000300]
        mu_p = 4.28283142580671e4
        mass = 6.4171e23
    elif Planet == 5:
        # Jupiter
        L = [34.351484, 3034.9056746, -0.00008501, 0.000000004]
        a = [5.202603191, 0.0000001913, 0.0, 0.0]
        e = [0.04849485, 0.000163244, -0.0000004719, -0.00000000197]
        i = [1.303270, - 0.0019872, 0.00003318, 0.000000092]
        W = [100.464441, 0.1766828, 0.00090387, -0.000007032]
        P = [14.331309, 0.2155525, 0.00072252, -0.000004590]
        mu_p = 1.26712767857796e8
        mass =1.8982e27
    elif Planet == 6:
        # Saturn
        L = [50.077471, 1222.1137943, 0.00021004, -0.000000019]
        a = [9.554909596, -0.0000021389, 0.0, 0.0]
        e = [0.05550862, -0.000346818, -0.0000006456, 0.00000000338]
        i = [2.488878, 0.0025515, -0.00004903, 0.000000018]
        W = [113.665524, -0.2566649, -0.00018345, 0.000000357]
        P = [93.056787, 0.5665496, 0.00052809, 0.000004882]
        mu_p = 3.79406260611373e7
        mass = 5.6834e26
    elif Planet == 7:
        # Uranus
        L = [314.055005, 428.4669983, -0.00000486, 0.000000006]
        a = [19.218446062, -0.0000000372, 0.00000000098, 0]
        e = [0.04629590, -0.000027337, 0.0000000790, 0.00000000025]
        i = [0.773196, -0.0016869, 0.00000349, 0.000000016]
        W = [74.005947, 0.0741461, 0.00040540, 0.000000104]
        P = [173.005159, 0.0893206, -0.00009470, 0.000000413]
        mu_p = 5.79454900707188e6
        mass = 8.6810e25
    elif Planet == 8:
        # Neptune
        L = [304.348665, 218.4862002, 0.00000059, -0.000000002]
        a = [30.110386869, -0.0000001663, 0.00000000069, 0.0]
        e = [0.00898809, 0.000006408, - 0.0000000008, -0.00000000005]
        i = [1.769952, 0.0002257, 0.00000023, 0.0]
        W = [131.784057, -0.0061651, -0.00000219, -0.000000078]
        P = [48.123691, 0.0291587, 0.00007051, -0.000000023]
        mu_p = 6.83653406387926e6
        mass = 1.02413e26
    elif Planet == 9:
        # Pluto
        L = [238.92903833, 145.20780515, 0.0, 0.0]
        a = [39.48211675, -0.00031596, 0.0, 0.0]
        e = [0.24882730, 0.00005170, 0.0, 0.0]
        i = [17.14001206, 0.00004818, 0.0, 0.0]
        W = [110.30393684, -0.01183482, 0.0, 0.0]
        P = [224.06891629, -0.04062942, 0.0, 0.0]
        mu_p = 9.81600887707005e2
        mass = 1.303e22

    coeff.L = L
    coeff.a = a
    coeff.e = e
    coeff.i = i
    coeff.W = W
    coeff.P = P
    coeff.mu_p = mu_p * 1e9
    coeff.mass = mass

    return coeff

def Moon_Position (JD):
    T_ut1= (JD - 2451545.0) / 36525.0
    T_TBD=T_ut1
    #Calculating Lambda
    Lambda_e = 218.32 + 481267.8813 * T_TBD + 6.29 * sind(134.9 + 477198.85 * T_TBD)\
                -1.27*sind(259.2-413335.39*T_TBD)+.66*sind(235.7+890534.23*T_TBD)\
                +.21*sind(269.9+954397.7*T_TBD)-.19*sind(357.5+35999.05*T_TBD)\
                -.11*sind(186.6+966404.05*T_TBD)
    #Lambda_e *= np.pi / 180
    # Calculating Phi_e
    Phi_e=5.13*sind(93.3+483202.03*T_TBD)+.28*sind(228.2+960400.87*T_TBD)
    -.28*sind(318.3+6003.18*T_TBD)-.17*sind(217.6-407332.20*T_TBD)

    #Phi_e*= np.pi / 180
    # Calculating WOW
    WOW=.9508+.0518*cosd(134.9+477198.85*T_TBD)
    +.0095*cosd(259.2-413335.38*T_TBD)+.0078*cosd(235.7+890534.23*T_TBD)
    +.0028*cosd(269.9+954397.7*T_TBD)
    #WOW*= np.pi / 180
    #Calcualating ep
    ep = 23.439291 - .0130042 * T_TBD- 1.64e-7*T_TBD**2 + 5.04e-7*T_TBD**3

    #ep *= np.pi / 180
    #Calculating r_moon magnitude
    r_moon_mag=(1/sind(WOW))*6378e3
    # Calculating the position vector of the moon
    r_moon=r_moon_mag*np.array([
    cosd(Phi_e)*cosd(Lambda_e),
    cosd(ep)*cosd(Phi_e)*sind(Lambda_e)-sind(ep)*sind(Phi_e),
    sind(ep)*cosd(Phi_e)*sind(Lambda_e)+cosd(ep)*sind(Phi_e)
    ])


    return r_moon, r_moon_mag

def Julian_Date (Timestamp):

#Taking the values within Timestamp and assigning them names.
    year, month, day, hour, minute, second=Timestamp
#Calculating your Julian date using the timestamp inputs
    JD= 367 * (year) - ( (7/4) * (year + ( (month+9) / 12) ) ) + ( (275 * month) / 9  ) + day + 1721013.5 + ( ( ( (second / 60) + minute) / 60 + hour )/24)

#Doing a modified MJD because the normal value is very large
    MJD=JD-2400000.5

    return MJD,JD

def Sun_Position (JD):

    T_ut1= (JD-2451545.0)/36525
    Lambda_M = 280.4606184 + 36000.77005361 * T_ut1
    M=357.5277233+35999.05034* T_ut1
    M*=np.pi/180
    Lambda_e= Lambda_M + 1.914666471 * np.sin(M) + .019994643 * np.sin(2 * M)
    Lambda_e*= np.pi / 180
    r_mag= 1.000140612 - .016708617 * np.cos(M) - .000139589 * np.cos(2 * M)
    ep = 23.439291 - .0130042 * T_ut1
    ep*=np.pi/180
    r= np.array([r_mag * np.cos(Lambda_e), r_mag * np.cos(ep) * np.sin(Lambda_e), r_mag * np.sin(ep) * np.sin(Lambda_e)]) * AU

    return r

#Auxillary Functions and Features_______________________________________________________________________________________

def GET_SOI(M,m,a): #Returns the sphere of influence of the body in question
    r_SOI=a*((m/M)**(2/5))
    print(r_SOI)
    return r_SOI

def Abs_Rel(Parent_Body,State_Store):
    #n_origin is the N value of the parent body
    Relative_Store=np.zeros(State_Store.shape)
    for i in range(State_Store.shape[0]):
        Relative_Store[i,:,:]=State_Store[i,:,:]-State_Store[Parent_Body,:,:]

    return Relative_Store

    #Deining the transformation matrix that you'll multiply your r and v perifocal vectors by.
    #This matrix will need to be converted from a 1 x 6 to a 3x3.

def Inert_Kep(state_vec,mu):
    #https://downloads.rene-schwarz.com/download/M002-Cartesian_State_Vectors_to_Keplerian_Orbit_Elements.pdf
    #This takes a state vector in inertial, (doesnt matter if its abs or rel) and out puts the cooresponding values of the keplerian orbit
    r_ijk = state_vec[:3]
    v_ijk = state_vec[3:]
    h     = np.cross(r_ijk,v_ijk)
    e_vec = (np.cross(v_ijk,h)/C.C["Earth"]["Mu"])-r_ijk/(np.linalg.norm(r_ijk))
    e=np.linalg.norm(e_vec)
    n_vec=np.cross(np.array([0, 0, 1]), h)
    n=np.linalg.norm(n_vec)
    f=np.arccos(np.dot(e_vec, r_ijk) / (e * np.linalg.norm(r_ijk)))
    if np.dot(r_ijk,v_ijk)<0:
        f= 2 * np.pi - f
    i = np.arccos(h[2]/np.linalg.norm(h))
    E=2*np.arctan2(np.tan(f / 2), np.sqrt((1 + e) / (1 - e)))
    ran=np.arccos(n_vec[0] / n)
    if n_vec[1]<0:
        ran=2*np.pi-ran
    w=np.arccos(np.dot(n_vec,e_vec) / (n*e))
    if e_vec[2]<0:
        w=2*np.pi-w

    M=E-e*np.sin(E)
    if M<=0:
        M=M+(2*np.pi)
    if f<=0:
        f=f+(2*np.pi)
    if E<=0:
        E=E+(2*np.pi)
    a=1/ (   (2/np.linalg.norm(r_ijk) ) - ( (np.linalg.norm(v_ijk)**2)/mu  ) )

    return [a,e,i,w,ran,f,M,E],e_vec

def Kep_Inert(a,e,i,w,ran,theta,mu):
    # Defining Semilatus Rectum
    p = a * (1 - (e ** 2))

    # Deining the perifocal (PQW) position and velocity vector.
    r_pqw = np.array([(p * cos(theta)) / (1 + e * cos(theta)), (p * sin(theta)) / (1 + e * cos(theta)), (0)])
    v_pqw = np.array([-sqrt(mu / p) * sin(theta), sqrt(mu / p) * (e + cos(theta)), (0)])
    ijkpqw = np.array([[(cos(ran) * cos(w) - sin(ran) * sin(w) * cos(i)),
                        (-cos(ran) * sin(w) - sin(ran) * cos(w) * cos(i)), (sin(ran) * sin(i))],
                       [(sin(ran) * cos(w) + cos(ran) * sin(w) * cos(i)),
                        (-sin(ran) * sin(w) + cos(ran) * cos(w) * cos(i)), (-cos(ran) * sin(i))],
                       [(sin(w) * sin(i)), (cos(w) * sin(i)), (cos(i))]])
    r_ijk = np.matmul(ijkpqw, r_pqw)
    v_ijk = np.matmul(ijkpqw, v_pqw)

    print("Inertial Position Vector:",r_ijk, "meters","\nInertial Velocity Vector:",v_ijk, "meters/second")
    State_Vect=np.hstack([r_ijk, v_ijk])
    return State_Vect

def Get_Angular(r_ijk,v_ijk):
    h = np.cross(r_ijk, v_ijk)
    return h

def Get_Period(a,M): #a is the semi major and M is the mass of the parent body.
    T=(2*np.pi*(a**(3/2)))/(np.sqrt(G*M))
    return T

def Parent_Body_Check(State_Store_wrt_Body1,State_Store_wrt_Body2,t,ROI_List): #N_Craft is the number of the craft in the relative store
    #First Column 0: The Craft has escaped every body in the system
    #First Column 1: The Craft is within the SOI of the first body
    #First Column 2: The Craft is within the SOI of the second body.
    #Second Column 0: The Craft does not have an orbit around anything.
    #Second COlumn 1: The Craft has a closed-stable Orbit.
    #Second Column 2: The craft has an eccentric orbit.
    Orbit_Flag_Store=np.zeros((t,2))
    for i in range(t): #Creating the first Parameter which body is considered the parent body at that particular time step.
        mag_0=np.linalg.norm(State_Store_wrt_Body1[2,:3, i])
        mag_1=np.linalg.norm(State_Store_wrt_Body2[2,:3, i])
        if mag_0<ROI_List[0] and mag_1>ROI_List[1]:
            Orbit_Flag_Store[i,0]=1
        elif mag_0<ROI_List[0] and mag_1<ROI_List[1]:
            Orbit_Flag_Store[i,0]=2
        else:
            Orbit_Flag_Store[i,:] = 0

        #Now Creating the tool that will check the eccentricity with respect to the parent body.
        if Orbit_Flag_Store[i,0]==1:
            r_ijk = State_Store_wrt_Body1[2,:3, i]
            v_ijk = State_Store_wrt_Body1[2,3:, i]
            h = np.cross(r_ijk, v_ijk)
            e_vec = (np.cross(v_ijk, h) / C.C["Earth"]["Mu"]) - r_ijk / (np.linalg.norm(r_ijk))
            e = np.linalg.norm(e_vec)
            if e<=1:
                Orbit_Flag_Store[i, 1] = 1
            elif e>=1:
                Orbit_Flag_Store[i, 1] = 2
            elif e<=0:
                Orbit_Flag_Store[i, 1] = 0
        if Orbit_Flag_Store[i,0]==2:
            r_ijk = State_Store_wrt_Body2[2,:3, i]
            v_ijk = State_Store_wrt_Body2[2,3:, i]
            h = np.cross(r_ijk, v_ijk)
            e_vec = (np.cross(v_ijk, h) / C.C["Moon"]["Mu"]) - r_ijk / (np.linalg.norm(r_ijk))
            e = np.linalg.norm(e_vec)
            if e<=1:
                Orbit_Flag_Store[i, 1] = 1
            elif e>=1:
                Orbit_Flag_Store[i, 1] = 2
            elif e <= 0:
                Orbit_Flag_Store[i, 1] = 0

    return Orbit_Flag_Store

def Transfer_Time(t,state_vec,State_Store_wrt_Body1,DT):
    r_ijk = State_Store_wrt_Body1[2, :3, 1]
    v_ijk = State_Store_wrt_Body1[2, 3:, 1]
    h = np.cross(r_ijk, v_ijk)
    h_mag=np.linalg.norm(h)
    e_vec = (np.cross(v_ijk, h) / C.C["Earth"]["Mu"]) - r_ijk / (np.linalg.norm(r_ijk))
    e = np.linalg.norm(e_vec)
    R_1=((h_mag**2/C.C["Earth"]["Mu"])/(1+e*np.cos(0))) #Perigee Circular Alt of Spacecraft
    R_2_Store = np.zeros((t,1))
    T_Store=np.zeros((t,1))
    for i in range(t):
        R_2_Store[i] = np.linalg.norm(State_Store_wrt_Body1[1, :3, i])
        a=(R_1+R_2_Store[i])/2
        #T_Store[i]=((np.pi/np.sqrt(C.C["Earth"]["Mu"]))*(a**(3/2)))/(3600*24)
        T_Store[i]=np.pi*np.sqrt((a**3)/(C.C["Earth"]["Mu"]))/(3600*24)

    return R_1,R_2_Store,T_Store #transfer arc times with their respective true anomaly degree value. (deg,sec)

def Delta_V(t,R_1,R_2_Store):
    DV_1=np.zeros((t,1))
    DV_2=np.zeros((t,1))
    DV_Total=np.zeros((t,1))
    for i in range(t):
        DV_1[i]=np.sqrt(C.C["Earth"]["Mu"]/R_1)*(np.sqrt(2*R_2_Store[i]/(R_1+R_2_Store[i]))-1)
        DV_2[i]=np.sqrt(C.C["Earth"]["Mu"]/R_2_Store[i])*(1-np.sqrt(2 * R_1/(R_1 + R_2_Store[i])))
        DV_Total[i]=DV_1[i]+DV_2[i]
    return DV_1,DV_2,DV_Total

def Phasing(t,T_Store,State_Store_wrt_Body1,DT): # The objective is to determine exactly when the differenace between the moons angle from an earth centered coordindate frame is (at the intercept time) exactly 180 degrees off of where the crafts current position is.
    #Moons Angle
    # Get the moons coord after predicted transfer time (neglect the z position)
    Angle_Diff = np.zeros((t, 1))
    for i in range(t):
        if int(i+(T_Store[i]*(3600*24))/DT)>=t:
            Angle_Diff[i]= 0
        else:
            #Calculate its angle with repect to the earth reference frame.
            Rvec_Moon_Intercept = State_Store_wrt_Body1[1, :3, i + int(T_Store[i] * (3600 * 24) / DT)]
            # Calculate the crafts angle with repect to the earth reference frame.
            Rvec_Craft_Current = State_Store_wrt_Body1[2, :3, i]
            Angle_Diff[i]=np.arccos(np.dot(Rvec_Moon_Intercept,Rvec_Craft_Current)/(np.linalg.norm(Rvec_Moon_Intercept)*np.linalg.norm(Rvec_Craft_Current)))*180/np.pi
            #Angle_Diff[i]=np.arctan2(np.linalg.norm(np.cross(Rvec_Moon_Intercept,Rvec_Craft_Current)), np.dot(Rvec_Moon_Intercept,Rvec_Craft_Current))*180/np.pi

    return Angle_Diff


#Extra Functions________________________________________________________________________________________________________

def Peri_Inert(r_pqw, v_pqw, ijkpqw):
    r_ijk = np.matmul(ijkpqw, r_pqw)
    v_ijk = np.matmul(ijkpqw, v_pqw)
    return r_ijk, v_ijk

def Kep_Peri(a, e, i, ran, w, theta, mu):
    # Defining Semilatus Rectum
    p = a * (1 - (e ** 2))

    # Deining the perifocal (PQW) position and velocity vector.
    r_pqw = np.array([(p * cos(theta)) / (1 + e * cos(theta)), (p * sin(theta)) / (1 + e * cos(theta)), (0)])
    v_pqw = np.array([-sqrt(mu / p) * sin(theta), sqrt(mu / p) * (e + cos(theta)), (0)])

    return r_pqw, v_pqw

def IJKPQW(i,ran,w):
    ijkpqw = np.array([[ (cos(ran)*cos(w)-sin(ran)*sin(w)*cos(i)),   (-cos(ran)*sin(w)-sin(ran)*cos(w)*cos(i)),    (sin(ran)*sin(i))],
                         [(sin(ran)*cos(w)+cos(ran)*sin(w)*cos(i)),   (-sin(ran)*sin(w)+cos(ran)*cos(w)*cos(i)),    (-cos(ran)*sin(i))],
                                    [(sin(w)*sin(i)),                           (cos(w)*sin(i)),                      (cos(i))]             ])
    print(ijkpqw.shape)
    return ijkpqw

def sind (x):
    deg=np.sin(x*(np.pi/180.0))
    return deg
def cosd (x):
    deg=np.cos(x*(np.pi/180.0))
    return deg

#Force Feeler Array________________________________________
def Create_Force_Feeler_Array(N_Feeler,X_Bound,Y_Bound,Z_Bound):
    state = np.zeros((N_Feeler, 3))  # Creating an Empty State Vector (x,y,z,vx,vy,vz), (0,0,0,0,0,0)
    x=np.linspace(-X_Bound,X_Bound,int(np.ceil(N_Feeler**(1/3))))
    y=np.linspace(-Y_Bound,Y_Bound,int(np.ceil(N_Feeler**(1/3))))
    z=np.linspace(-Z_Bound,Z_Bound,int(np.ceil(N_Feeler**(1/3))))
    state=[]
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                state.append([x[i],y[j],z[k]])

    return np.array(state)

def Feel_Accel(feeler_array,State_Store,Mass):
    soft     = 6378e3
    t        =State_Store.shape[2]
    n        =feeler_array.shape[0]
    accel    =np.zeros((n,t))
    threshold=.01
    mask     =np.zeros((n,t))
    for tt in range(t):
        for i in range(n):
            accel_vec=np.zeros((3,1))
            for j in range(State_Store.shape[0]):
                dx = feeler_array[i, 0] - State_Store[j, 0, tt]
                dy = feeler_array[i, 1] - State_Store[j, 1, tt]
                dz = feeler_array[i, 2] - State_Store[j, 2, tt]
                dist=np.array([dx,dy,dz])
                dist_mag=np.linalg.norm(np.array([dx,dy,dz,soft]))
                dist_unit_vect=dist/dist_mag
                accel_vec[:,0]+=((-G*Mass[0,j])/dist_mag**2)*dist_unit_vect
            accel[i,tt]=np.linalg.norm(accel_vec)
            if accel[i,tt]<=threshold:
                mask[i,tt]=1

    return accel, mask

