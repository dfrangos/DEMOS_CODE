from N_Body_Functions import *
import Constants as C


Day=86400/DT
def Transfer_Time(t,State_Vec_2):
    R_Store_1=np.zeros((360,1))
    for i in range(t):
        Relative
        State_Vec_1=Kep_Inert(600e3+6378e3,.03,0.43379,0,0,n,C.C["Earth"]["Mu"])
        R_Store_1[n,0]=np.linalg.norm(State_Vec_1[0:3])

    R_Store_2=np.zeros((360,1))
    for n in range(360):
        State_Vec_1=Kep_Inert(600e3+6378e3,.03,0.43379,0,0,n,C.C["Earth"]["Mu"])
        R_Store_1[n,0]=np.linalg.norm(State_Vec_1[0:3])


    return T_Store #transfer arc times with their respective true anomaly degree value. (deg,sec)