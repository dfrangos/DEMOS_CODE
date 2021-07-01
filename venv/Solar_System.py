import numpy as np
AU_meter=149597871e3

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
    r= np.array([r_mag * np.cos(Lambda_e), r_mag * np.cos(ep) * np.sin(Lambda_e), r_mag * np.sin(ep) * np.sin(Lambda_e)]) * AU_meter

    return r

def sind (x):
    deg=np.sin(x*(np.pi/180))
    return deg
def cosd (x):
    deg=np.sin(x*(np.pi/180))
    return deg

def Moon_Position (JD):

    T_ut1= (JD - 2451545.0) / 36525
    T_TBD=T_ut1
    #Calculating Lambda
    Lambda_e = 218.32 + 481267.8813 * T_TBD + 6.29 * sind(134.9 + 477198.85 * T_TBD)
    -1.27*sind(259.2-413335.39*T_TBD)+.66*sind(235.7+890534.23*T_TBD)
    +.21*sind(269.9+954397.7*T_TBD)-.19*sind(357.5+35999.05*T_TBD)
    -.77*sind(186.6+966404.05*T_TBD)
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
    ep = 23.439291 - .0130042 * T_TBD
    #ep *= np.pi / 180
    #Calculating r_moon magnitude
    r_moon_mag=(1/sind(WOW))#*6378e3
    # Calculating the position vector of the moon
    r_moon=r_moon_mag*np.array([
    cosd(Phi_e)*cosd(Lambda_e),
    cosd(ep)*cosd(Phi_e)*sind(Lambda_e)-sind(ep)*sind(Phi_e),
    sind(ep)*cosd(Phi_e)*sind(Lambda_e)+cosd(ep)*sind(Phi_e)
    ])


    return r_moon, r_moon_mag

Epoch=[1994,4,28,12,0,0]
MD,JD=Julian_Date (Epoch)
# Position=Sun_Position (JD)
# print(Position)
# print(JD)
Position,b=Moon_Position (JD)
print(Position)

