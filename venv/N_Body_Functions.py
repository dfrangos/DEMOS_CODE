import time
import math
import numpy as np
M_sun=1.988473e30# kg
AU=149597870700 #m
G = ((6.67408e-11) / (AU ** 3)) * M_sun

#Your Scenario Functions_________________________
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
    #mass[0,0]=5e27  #if you activate this it'll let you have the first body be any particular value you want.
    #state[0,:]=[0,0,0,0,0,0]  #if you activate this it'll let you have the first body be any particular value you want.
    return state, mass, soft

def Create_Solar_System():
    # Defining Softening factor
    soft = 6378e5/AU
    #state=np.array([[0,0,0,0,0,0],[(.41*AU),0,0,0,47.9e3,0],[.74*AU,0,0,50,35e3,-150],[1.0*AU,.02*AU,.01*AU,100,29.8e3,-100]])/AU
    #mass=np.array([[1.989e30,0.33e24,1.84767309e28,5.97e24]])/M_sun
    state=np.array([[0,0,0,0,0,0],[(.41*AU),0,0,0,47.9e3,0],[.74*AU,0,0,0,35e3,0],[1.0*AU,0,0,0,29.8e3,0],[(1.0*AU)+404e6,0,0,0,29.8e3+np.sqrt(3.986e14/404e6),0],[1.6*AU,0,0,0,24.1e3,0]])/AU
    mass=np.array([[1.989e30,0.33e24,4.87e24,5.97e24,7.34767309e22,0.642e24]])/M_sun

    return state, mass, soft

#__________________________________________________________

def Get_Accel(N,state,mass,soft):
    accel=np.zeros((N,3))
    for i in range(N):
        for j in range(N):
            if i != j:
                #Calculates the distance of a particular mass to another in each dimension.
                dx=state[i,0]-state[j,0]
                dy=state[i,1]-state[j,1]
                dz=state[i,2]-state[j,2]
                Dist=np.array([dx,dy,dz])
                Dist_Mag=np.sqrt((dx**2+dy**2+dz**2+soft**2))
                ##Calculates the acceleration induced on any particular pair of bodies
                accel[i,:]-=G*mass[0,j]*np.array([(Dist[0])/(Dist_Mag**3),(Dist[1])/(Dist_Mag**3),(Dist[2])/(Dist_Mag**3)])
    return accel

def Update_State(N,state,accel,dt,mass,soft):
    for i in range(N):
        state[i, 3:] = state[i, 3:] + ((dt/2)*accel[i,:]) #Updating the velocity based on the acceleration
        state[i, :3] = state[i, :3] + (dt) * state[i, 3:]  # Updating the position based on the velo
        accel = Get_Accel(N,state,mass,soft)
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
