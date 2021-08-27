import numpy as np
import scipy
import matplotlib
import tkinter

#Just doing some short hand stuff for SOCAHTOA
sin=np.sin
cos=np.cos
tan=np.tan
sqrt=np.sqrt


def Kep_Peri(a,e,i,ran,w,theta,mu):
    #Defining Semilatus Rectum
    p = a*(1-(e**2))

    #Deining the perifocal (PQW) position and velocity vector.
    r_pqw = np.array([ (p * cos(theta))/(1 + e * cos(theta)), (p * sin(theta)) / (1 + e * cos(theta)), (0)  ])
    v_pqw = np.array([ -sqrt(mu/p)*sin(theta), sqrt(mu/p)*(e + cos(theta)), (0) ])
    return r_pqw, v_pqw
#__________________________________________________________________
    #Deining the transformation matrix that you'll multiply your r and v perifocal vectors by.
    #This matrix will need to be converted from a 1 x 6 to a 3x3.
def IJKPQW(i,ran,w):
    ijkpqw = np.array([[ (cos(ran)*cos(w)-sin(ran)*sin(w)*cos(i)),   (-cos(ran)*sin(w)-sin(ran)*cos(w)*cos(i)),    (sin(ran)*sin(i))],
                         [(sin(ran)*cos(w)+cos(ran)*sin(w)*cos(i)),   (-sin(ran)*sin(w)+cos(ran)*cos(w)*cos(i)),    (-cos(ran)*sin(i))],
                                    [(sin(w)*sin(i)),                           (cos(w)*sin(i)),                      (cos(i))]             ])
    print(ijkpqw.shape)
    return ijkpqw

def Peri_Inert (r_pqw,v_pqw,ijkpqw):
    r_ijk = np.matmul(ijkpqw, r_pqw)
    v_ijk = np.matmul(ijkpqw, v_pqw)
    return r_ijk, v_ijk
