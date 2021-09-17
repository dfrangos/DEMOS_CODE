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
import Constants as C
import os
import csv
from vpython import *
from N_Body_Functions import *
from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from future.moves import tkinter as Tkinter
from Solar_System import *
from Coordinate_Transform import *

np.random.seed(1)
Save_to_File      = "Off"
print(os.getcwd())
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
#Creating my layout condition for the window creation. This is where you setup word prompts and buttons:
sg.theme('DarkAmber')
Format_Main_Menu= [[sg.Text("Which Scenario Would You Like to Eplore?!?!")],[sg.Button("Random")],[sg.Button("Solar System")],[sg.Button("Earth-Moon System")],[sg.Button("Martian System")],[sg.Button("Jovian System")],[sg.Button("TRAPIST-1 System")],[sg.Button("Cancel")]]
#Creates the window with the layout condition.
Screen_Main_Menu=sg.Window("Demokritos' Game of Gravitation", Format_Main_Menu, margins=(150, 70))
#Creating an event loop
while True:
    #https://pypi.org/project/PySimpleGUI/#:~:text=PySimpleGUI%20is%20a%20Python%20package%20that%20enables%20Python,frameworks%20to%20display%20and%20interact%20with%20your%20window.
    event_series1,values=Screen_Main_Menu.read()
    if event_series1 == "Random":
        Format_Random =  [ [sg.Text('Would you like to Customize or Run the Default?')],
                         [sg.Button("Customize")], [sg.Button("Default")], [sg.Button("Cancel")]]
        Screen_Random = sg.Window("Demokritos' Game of Gravitation", Format_Random, margins=(150, 70))
        event_series2, values = Screen_Random.read()
        if event_series2 == "Default":
            t = 2000  # How many iterations
            DT = 15  # Your delta T jumps
            N = 4
            Mass = 5e24
            Pos_Bound = 6.84e4
            Vel_Bound = 300
            # Defining my storage area for my position values
            State, Mass, Soft = Create_Random(N,Mass,Pos_Bound,Vel_Bound)
            State_Store = np.zeros((N, 6, t))
            Accel = Get_Accel(N, State, Mass, Soft)
            Burn_Index = 1870
            Burn_Index_2 = 6290
            for k in range(t):
                # Structure for flag: burn_flag, target_body, dv_mag, origin_body, direction, time
                if k == Burn_Index:
                    Flag = [0, 2, 300, 0, 1]
                elif k == Burn_Index_2:
                    Flag = [0, 2, 355, 1, -1]
                else:
                    Flag = [0, 0, 0, 0, 0]
                State = Update_State(N, State, Accel, DT, Mass, Soft, Flag)
                State_Store[:, :, k] = State[:, 0:]
                # Saving the State Store Information to a File
                while Save_to_File == "On":
                    write_to_file = True
                    filename = 'n_body_dat_' + str(N) + Scenario_Type + '.npy'
                    if write_to_file:
                        with open(filename, 'wb') as f:
                            np.save(f, State_Store)

            # This is the animation stuff______________________________
            # ANIMATION FUNCTION
            # ___________________________________________________________________________

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


            line = []

            for i in range(N):
                line.append(plt.plot(State_Store[i, 0, 0], State_Store[i, 1, 0], State_Store[i, 2, 0], marker=".")[0])

            anim = FuncAnimation(fig1, func, frames=t, repeat=True, interval=1, fargs=(State_Store, line, N))
            # anim.save('rgb_cube.gif', dpi=80, writer='imagemagick', fps=24)
            ax1.set_xlabel("x (m)")
            ax1.set_ylabel("y (m)")
            ax1.set_zlabel("z (m)")
            ax1.set_title("Orbital Trjectory")
            ax1.grid()
            ax1.legend(["Body 1", "Body 2", "Body 3", "Body 4", "Body 5"])
            plt.show()

            State_Store_wrt_Earth  = Abs_Rel(0, State_Store)
            State_Store_wrt_Moon   = Abs_Rel(1, State_Store)
            State_Store_wrt_Craft1 = Abs_Rel(2, State_Store)
            break
        elif event_series2 == "Customize":
            #This section will read the user data that was inputed in the previous run
            #https://pythonspot.com/reading-csv-files-in-python/
            UserData  = []
            N=          []
            Pos_Bound = []
            Vel_Bound = []
            Mass =      []
            t =         []
            DT =        []
            with open('User_Input_Custom.txt','r') as File:
                lines=File.readlines()
                N        = int(lines[0])
                Pos_Bound= float(lines[1])
                Vel_Bound= float(lines[2])
                Mass     = float(lines[3])
                t        = int(lines[4])
                DT       = float(lines[5])
            File.close()

            Format3 =  [ [sg.Text('Setup Your Parameters:')],
                         [sg.Text('How Many Bodies Would You Like to Simulate? (int)'),          sg.InputText(str(N),         key='-N-')         ],
                         [sg.Text('What Positional Bounds Would You Like to Use?  -\+ (m)'),     sg.InputText(str(Pos_Bound), key='-Pos_Bound-') ],
                         [sg.Text('What Velocity Bounds Would You Like to Use?  -\+(m/s)'),      sg.InputText(str(Vel_Bound), key='-Vel_Bound-') ],
                         [sg.Text('What are the Masses of the Bodies? (kg)'),                    sg.InputText(str(Mass),      key='-Mass-')      ],
                         [sg.Text('How Many Iterations Would You Like to Execute? (int)'),       sg.InputText(str(t),         key='-t-')         ],
                         [sg.Text('How Far in Time Would You Like to Jump For Each Iteration?'), sg.InputText(str(DT),        key='-DT-')        ],
                         [sg.Button("Okay")],
                         [sg.Button("Cancel")]]

            Screen3 = sg.Window("Demokritos' Game of Gravitation", Format3, margins=(150, 70))
            event_series3, values = Screen3.read()
            if event_series3=='Okay':
                N         = int(values['-N-'])
                Pos_Bound = float(values['-Pos_Bound-'])
                Vel_Bound = float(values['-Vel_Bound-'])
                Mass      = float(values['-Mass-'])
                t         = int(values['-t-'])
                DT        = float(values['-DT-'])
                User_Input_Data = N, Pos_Bound, Vel_Bound, Mass, t, DT
                with open("User_Input_Custom.txt", "w") as File:
                # https://www.youtube.com/watch?v=irnj19jz8uI
                    for i in range(len(User_Input_Data)):
                        Output = ""
                        Output += str(User_Input_Data[i])
                        Output += '\n'
                        File.write(Output)
                File.close()
                # Defining my storage area for my position values
                State, Mass, Soft = Create_Random(N,Mass,Pos_Bound,Vel_Bound)
                State_Store = np.zeros((N, 6, t))
                Accel = Get_Accel(N, State, Mass, Soft)
                Burn_Index = 1870
                Burn_Index_2 = 6290
                for k in range(t):
                    # Structure for flag: burn_flag, target_body, dv_mag, origin_body, direction, time
                    if k == Burn_Index:
                        Flag = [0, 2, 300, 0, 1]
                    elif k == Burn_Index_2:
                        Flag = [0, 2, 355, 1, -1]
                    else:
                        Flag = [0, 0, 0, 0, 0]
                    State = Update_State(N, State, Accel, DT, Mass, Soft, Flag)
                    State_Store[:, :, k] = State[:, 0:]
                    # Saving the State Store Information to a File
                    while Save_to_File == "On":
                        write_to_file = True
                        filename = 'n_body_dat_' + str(N) + Scenario_Type + '.npy'
                        if write_to_file:
                            with open(filename, 'wb') as f:
                                np.save(f, State_Store)

                # This is the animation stuff______________________________
                # ANIMATION FUNCTION
                # ___________________________________________________________________________

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


                line = []

                for i in range(N):
                    line.append(
                        plt.plot(State_Store[i, 0, 0], State_Store[i, 1, 0], State_Store[i, 2, 0], marker=".")[0])

                anim = FuncAnimation(fig1, func, frames=t, repeat=True, interval=1, fargs=(State_Store, line, N))
                # anim.save('rgb_cube.gif', dpi=80, writer='imagemagick', fps=24)
                ax1.set_xlabel("x (m)")
                ax1.set_ylabel("y (m)")
                ax1.set_zlabel("z (m)")
                ax1.set_title("Orbital Trjectory")
                ax1.grid()
                ax1.legend(["Body 1", "Body 2", "Body 3", "Body 4", "Body 5"])
                plt.show()

                State_Store_wrt_Earth = Abs_Rel(0, State_Store)
                State_Store_wrt_Moon = Abs_Rel(1, State_Store)
                State_Store_wrt_Craft1 = Abs_Rel(2, State_Store)
                break

            if event_series3=="Cancel" or event_series3 == sg.WIN_CLOSED:
                break
        elif event_series2 == "Cancel" or event_series2 == sg.WIN_CLOSED:
            break
    elif event_series1 == "Solar System":
        t = 5000  # How many iterations
        DT = 36000  # Your delta T jumps
        N = 6
        # year, month, day, hour, minute, second=Timestamp
        year=2010
        month=2
        day=12
        hour=3
        minute=5
        second=56
        MJD, JD = Julian_Date([year,month,day,hour,minute,second])
        State, Mass, Soft = Create_Solar_System(N,JD)
        # Defining my storage area for my position values
        State_Store = np.zeros((N, 6, t))
        Accel = Get_Accel(N, State, Mass, Soft)

        for k in range(t):
            State = Update_State(N, State, Accel, DT, Mass, Soft)
            State_Store[:, :, k] = State[:, 0:]

            # Saving the State Store Information to a File
            while Save_to_File == "On":
                write_to_file = True
                filename = 'n_body_dat_' + str(N) + Scenario_Type + '.npy'
                if write_to_file:
                    with open(filename, 'wb') as f:
                        np.save(f, State_Store)

        # This is the animation stuff______________________________
        # ANIMATION FUNCTION
        # ___________________________________________________________________________

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


        line = []

        for i in range(N):
            line.append(plt.plot(State_Store[i, 0, 0], State_Store[i, 1, 0], State_Store[i, 2, 0], marker=".")[0])

        anim = FuncAnimation(fig1, func, frames=t, repeat=True, interval=1, fargs=(State_Store, line, N))
        # anim.save('rgb_cube.gif', dpi=80, writer='imagemagick', fps=24)
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        ax1.set_zlabel("z (m)")
        ax1.set_title("Orbital Trjectory")
        ax1.grid()
        ax1.legend(["Body 1", "Body 2", "Body 3", "Body 4", "Body 5"])
        plt.show()
        # VPTHON EXPERIMENT
        # ____________________________

        running = True
        def Run(b):
            global running
            running = not running
            if running:
                b.text = "Pause"
            else:
                b.text = "Run"
        button(text="Pause", pos=scene.title_anchor, bind=Run)


        EARTH = sphere(pos=vector(State_Store[3, 0, 0], State_Store[3, 1, 0], State_Store[3, 2, 0]), radius=6378e6,
                       color=color.blue, make_trail=True, trail_type='points', interval=10, retain=25)
        VENUS = sphere(pos=vector(State_Store[2, 0, 0], State_Store[2, 1, 0], State_Store[2, 2, 0]), radius=2356e6,
                       color=color.orange, make_trail=True, trail_type='points', interval=10, retain=25)
        MERCURY = sphere(pos=vector(State_Store[1, 0, 0], State_Store[1, 1, 0], State_Store[1, 2, 0]), radius=7356e5,
                         color=color.white, make_trail=True, trail_type='points', interval=10, retain=25)
        SUN = sphere(pos=vector(State_Store[0, 0, 0], State_Store[0, 1, 0], State_Store[0, 2, 0]), radius=7356e6,
                     color=color.yellow, make_trail=True, trail_type='points', interval=10, retain=25)
        lamp = local_light(pos=vector(State_Store[0, 0, 0], State_Store[0, 1, 0], State_Store[0, 2, 0]),
                           color=color.yellow)

        Elabel = label(pos=vector(State_Store[3, 0, 0], State_Store[3, 1, 0], State_Store[3, 2, 0]), text='EARTH',
                       xoffset=10, height=10, color=color.blue)
        Vlabel = label(pos=vector(State_Store[2, 0, 0], State_Store[2, 1, 0], State_Store[2, 2, 0]), text='VENUS',
                       xoffset=10, height=10, color=color.yellow)
        Mlabel = label(pos=vector(State_Store[1, 0, 0], State_Store[1, 1, 0], State_Store[1, 2, 0]), text='MERCURY',
                       xoffset=10, height=10, color=color.white)
        Slabel = label(pos=vector(State_Store[0, 0, 0], State_Store[0, 1, 0], State_Store[0, 2, 0]), text='SUN',
                       xoffset=10, height=10, color=color.blue)

        k = 0
        while k < t:
            if running:
                rate(10)
                EARTH.pos = vector(State_Store[3, 0, k], State_Store[3, 1, k], State_Store[3, 2, k])
                VENUS.pos = vector(State_Store[2, 0, k], State_Store[2, 1, k], State_Store[2, 2, k])
                MERCURY.pos = vector(State_Store[1, 0, k], State_Store[1, 1, k], State_Store[1, 2, k])
                SUN.pos = vector(State_Store[0, 0, k], State_Store[0, 1, k], State_Store[0, 2, k])
                Elabel.pos = vector(State_Store[3, 0, k], State_Store[3, 1, k], State_Store[3, 2, k])
                Mlabel.pos = vector(State_Store[1, 0, k], State_Store[1, 1, k], State_Store[1, 2, k])
                Vlabel.pos = vector(State_Store[2, 0, k], State_Store[2, 1, k], State_Store[2, 2, k])
                Slabel.pos = vector(State_Store[0, 0, k], State_Store[0, 1, k], State_Store[0, 2, k])
                k += 1

        break
    elif event_series1 == "Earth-Moon System":
        with open('User_Input_Earth_Moon.txt', 'r') as File:
            lines = File.readlines()
            N         = int(lines[0])
            Pos_Bound = float(lines[1])
            Vel_Bound = float(lines[2])
            Mass      = float(lines[3])
            t         = int(lines[4])
            DT        = float(lines[5])
        File.close()
        t = 3000  # How many iterations
        DT = 400  # Your delta T jumps
        N = 3 #Number of Bodies
        ROI_Moon = GET_SOI(C.C["Earth"]["Mass"], C.C["Moon"]["Mass"], 384399e3)
        # Defining my storage area for my position values
        State, Mass, Soft = Create_Earth_Moon_System(N)
        State_Store = np.zeros((N, 6, t))
        Accel = Get_Accel(N, State, Mass, Soft)
        Burn_Index=200
        Burn_Index_2=400
        for k in range(t):
            #Structure for flag: burn_flag, target_body, dv_mag, origin_body, direction, time
            if k==Burn_Index:
                Flag=[1,2,1.5e3,0,1]
            elif k==Burn_Index_2:
                Flag=[1,2,.700e3,1,1]
            else:
                Flag=[0,0,0,0,0]
            State = Update_State(N, State, Accel, DT, Mass, Soft,Flag)
            State_Store[:, :, k] = State[:, 0:]
            # Saving the State Store Information to a File
            while Save_to_File == "On":
                write_to_file = True
                filename = 'n_body_dat_' + str(N) + Scenario_Type + '.npy'
                if write_to_file:
                    with open(filename, 'wb') as f:
                        np.save(f, State_Store)

        # This is the animation stuff______________________________
        # ANIMATION FUNCTION
        # ___________________________________________________________________________

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


        line = []

        for i in range(N):
            line.append(plt.plot(State_Store[i, 0, 0], State_Store[i, 1, 0], State_Store[i, 2, 0], marker=".")[0])

        anim = FuncAnimation(fig1, func, frames=t, repeat=True, interval=1, fargs=(State_Store, line, N))
        # anim.save('rgb_cube.gif', dpi=80, writer='imagemagick', fps=24)
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        ax1.set_zlabel("z (m)")
        ax1.set_title("Orbital Trjectory")
        ax1.grid()
        ax1.legend(["Body 1", "Body 2", "Body 3", "Body 4", "Body 5"])
        plt.show()


        State_Store_wrt_Earth  = Abs_Rel(0, State_Store)
        State_Store_wrt_Moon   = Abs_Rel(1, State_Store)
        State_Store_wrt_Craft1 = Abs_Rel(2, State_Store)

        # VPTHON Section
        # ____________________________

        Craft1_Orb_Elements_Ex,Craft1_e_Vec_Ex=Inert_Kep(State_Store_wrt_Earth[2,:,0],C.C["Earth"]["Mu"])
        Craft1_Major_Axis_Ex=float(Craft1_Orb_Elements_Ex[0])
        Craft1_T_Ex=Get_Period(Craft1_Major_Axis_Ex,C.C["Earth"]["Mass"])

        running = True
        #This function binds the action of the pause button that will be defined in the future
        def Run(b):
            global running
            running = not running
            if running:
                b.text = "Pause"
            else:
                b.text = "Run"

        button(text="Pause", pos=scene.title_anchor, bind=Run)

        #This function creates the menu that will be used to select which object you want to snap the view to.
        valnum=0

        def Target_Menu(m):
            val=m.selected
            global valnum
            if val=="Earth":
                scene.camera.follow(EARTH)
                valnum=0
            elif val=="Moon":
                scene.camera.follow(MOON)
                valnum = 1
            elif val=="Craft1":
                scene.camera.follow(CRAFT1)
                valnum = 2
            if val == "Home":
                scene.camera.center=vector(0,0,0)
                #scene.camera.follow(None)

        labels = ["Earth", "Moon", "Craft1"]
        menu(choices=['Choose an object', 'Earth', 'Moon', 'Craft1','Home'], bind=Target_Menu, right=30, pos=scene.title_anchor)

        # This function creates the menu that will be used to select the parent body .
        Parent_Num = 0
        def Parent_Menu(m):
            val = m.selected
            global Parent_Num
            if val == "Earth":
                Parent_Num = 0
            elif val == "Moon":
                Parent_Num = 1
            elif val == "Craft1":
                Parent_Num = 2


        menu(choices=['Choose a Parent Body', 'Earth', 'Moon', 'Craft1'], bind=Parent_Menu, right=90, pos=scene.title_anchor)

        # This function creates the menu that will be used to select if the user wants to see relative or abs orbit trails
        Trail_Flag = 2

        def Orb_Menu(m):
            val = m.selected
            global Trail_Flag
            if val == "Relative":
                Trail_Flag = 0
            elif val == "Absolute":
                Trail_Flag = 1
            elif val == "Both":
                Trail_Flag = 2

        menu(choices=['Choose a Trail Type', 'Relative', 'Absolute','Both'], bind=Orb_Menu, right=60, pos=scene.title_anchor)


        # This function creates the slide bar that will allow the user to change the rate at which the animation is played.
        def setspeed(s):
            wt.text = '{:1.2f}'.format(s.value)

        playrate = slider(min=1, max=1000, value=10, length=600, bind=setspeed, right=15, pos=scene.title_anchor)
        wt = wtext(text='{:1.2f}'.format(playrate.value),pos=scene.title_anchor)

        str_format = '''Time: {:.1f} JD
                        ----------
                        <b>{:s}</b> 
                        Absolute                 Relative     
                        X: {:.2f} m              X: {:.2f} m
                        Y: {:.2f} m              Y: {:.2f} m
                        Z: {:.2f} m              Z: {:.2f} m
                        RMag: {:.2f} m           RMag: {:.2f} m
                        VX: {:.2f} m/s           VX: {:.2f} m/s
                        VY: {:.2f} m/s           VY: {:.2f} m/s
                        VZ: {:.2f} m/s           VZ: {:.2f} m/s
                        VMag: {:.2f} m/s         VMag: {:.2f} m/s'''
        # BOOTING UP THE BODIES.
        #_______________________________________________________
        EARTH        = sphere(pos=vector(State_Store[0, 0, 0], State_Store[0, 1, 0], State_Store[0, 2, 0]),radius=C.C["Earth"]["Radius"], make_trail=True, trail_type='curve', interval=30, retain=600,shininess=.1, texture={'file': "\Images\Earth.jpg"})
        MOON         = sphere(pos=vector(State_Store[1, 0, 0], State_Store[1, 1, 0], State_Store[1, 2, 0]),radius=C.C["Moon"]["Radius"], make_trail=True, trail_type='curve', interval=30, retain=1400,shininess=0.1, texture={'file': "\Images\Moon.jpg"})
        CRAFT1       = sphere(pos=vector(State_Store[2, 0, 0], State_Store[2, 1, 0], State_Store[2, 2, 0]), radius=C.C["Craft1"]["Radius"], color=color.gray(0.5), make_trail=True,    trail_type='curve', interval=1, retain=600, shininess=0.1)
        # BOOTING UP THE LABELS.
        #______________________________________________________
        Elabel       = label(pos=vector(State_Store[0, 0, 0], State_Store[0, 1, 0], State_Store[0, 2, 0]), text='Earth',  xoffset=10, height=10, color=color.blue)
        Mlabel       = label(pos=vector(State_Store[1, 0, 0], State_Store[1, 1, 0], State_Store[1, 2, 0]), text='Moon',xoffset=10, height=10, color=color.white)
        C1label      = label(pos=vector(State_Store[2, 0, 0], State_Store[2, 1, 0], State_Store[2, 2, 0]), text='Craft1',xoffset=10, height=10, color=color.green)
        # BOOTING UP THE RELATIVE ORBITS.

        # ______________________________________________________
        CRAFT1_Rel_Orb        = curve(vector(State_Store_wrt_Earth[2, 0, 0], State_Store_wrt_Earth[2, 1, 0], State_Store_wrt_Earth[2, 2, 0]),radius=200e3, retain=(Craft1_T_Ex/DT)+1   )
        CRAFT1_Rel_Orb.origin = vector(State_Store[0, 0, 0], State_Store[0, 1, 0], State_Store[0, 2, 0])
        #BOOTING UP THE SPHERES OF INFLUENCE
        # ______________________________________________________
        ROI_MOON = sphere(pos=vector(State_Store[1, 0, 0], State_Store[1, 1, 0], State_Store[1, 2, 0]), radius=ROI_Moon, color=color.white,opacity=.08)

        #SETTING UP THE INITIAL CAMERA CONDITIONS
        k=0
        scene.forward    = vector(0,0,1)
        scene.up         = vector(1,0,0)
        scale            = 1e-10 / 1e2
        scene.range      = 1000000
        scene.fov        = .0001
        scene.autoscale  = False
        scene.width = 1400
        scene.height = 700
        #BEGINING THE ANIMATION
        while k < t:
            if running:
                Craft1_Orb_Elements_In, Craft1_e_Vec_In = Inert_Kep(State_Store_wrt_Earth[2, :, k], C.C["Earth"]["Mu"])
                Craft1_Major_Axis_In = float(Craft1_Orb_Elements_In[0])
                Craft1_T_In = Get_Period(Craft1_Major_Axis_In, C.C["Earth"]["Mass"])
                rate(playrate.value)
                if Trail_Flag==0:
                    CRAFT1_Rel_Orb.visible=True
                    CRAFT1.clear_trail()
                    CRAFT1.make_trail=False
                elif Trail_Flag==1:
                    CRAFT1_Rel_Orb.visible = False
                    CRAFT1_Rel_Orb.clear()
                    CRAFT1.make_trail = True
                elif Trail_Flag==2:
                    CRAFT1_Rel_Orb.visible = True
                    CRAFT1.make_trail = True

                # PROPAGATING THE BODIES.
                # _______________________________________________________
                EARTH.pos    = vector(State_Store[0, 0, k], State_Store[0, 1, k], State_Store[0, 2, k])
                MOON.pos     = vector(State_Store[1, 0, k], State_Store[1, 1, k], State_Store[1, 2, k])
                CRAFT1.pos   = vector(State_Store[2, 0, k], State_Store[2, 1, k], State_Store[2, 2, k])
                # PROPAGATING THE LABELS.
                # _______________________________________________________
                Elabel.pos   = vector(State_Store[0, 0, k], State_Store[0, 1, k], State_Store[0, 2, k])
                Mlabel.pos   = vector(State_Store[1, 0, k], State_Store[1, 1, k], State_Store[1, 2, k])
                C1label.pos  = vector(State_Store[2, 0, k], State_Store[2, 1, k], State_Store[2, 2, k])
                # PROPAGATING THE RELATIVE ORBITS.
                # _______________________________________________________
                if Craft1_Major_Axis_In>=Craft1_Major_Axis_Ex*1.015 or Craft1_Major_Axis_In<=Craft1_Major_Axis_Ex*0.985:
                    Craft1_Major_Axis_Ex=Craft1_Major_Axis_In
                    CRAFT1_Rel_Orb.retain=(Craft1_T_In/DT)
                CRAFT1_Rel_Orb.append(pos=vector(State_Store_wrt_Earth[2, 0, k], State_Store_wrt_Earth[2, 1, k], State_Store_wrt_Earth[2, 2, k]),radius=200e3, color=color.red)
                CRAFT1_Rel_Orb.origin = vector(State_Store[0, 0, k], State_Store[0, 1, k], State_Store[0, 2, k])

                # PROPAGATING THE SPHERES OF INFLUENCE.
                # _______________________________________________________
                ROI_MOON.pos = vector(State_Store[1, 0, k], State_Store[1, 1, k], State_Store[1, 2, k])

                #Creating the text at the bottom.
                if Parent_Num==0:
                    scene.caption=(str_format.format(k,labels[valnum],
                                                       State_Store[valnum, 0, k],State_Store_wrt_Earth[valnum, 0, k],
                                                       State_Store[valnum, 1, k],State_Store_wrt_Earth[valnum, 1, k],
                                                       State_Store[valnum, 2, k],State_Store_wrt_Earth[valnum, 2, k],
                                                       np.linalg.norm(State_Store[valnum,:3, k]), np.linalg.norm(State_Store_wrt_Earth[valnum,:3, k]),
                                                       State_Store[valnum, 3, k],State_Store_wrt_Earth[valnum, 3, k],
                                                       State_Store[valnum, 4, k],State_Store_wrt_Earth[valnum, 4, k],
                                                       State_Store[valnum, 5, k],State_Store_wrt_Earth[valnum, 5, k],
                                                       np.linalg.norm(State_Store[valnum,3:, k]), np.linalg.norm(State_Store_wrt_Earth[valnum,3:, k])))
                elif Parent_Num==1:
                    scene.caption = (str_format.format(k, labels[valnum],
                                                       State_Store[valnum, 0, k], State_Store_wrt_Moon[valnum, 0, k],
                                                       State_Store[valnum, 1, k], State_Store_wrt_Moon[valnum, 1, k],
                                                       State_Store[valnum, 2, k], State_Store_wrt_Moon[valnum, 2, k],
                                                       np.linalg.norm(State_Store[valnum, :3, k]),
                                                       np.linalg.norm(State_Store_wrt_Moon[valnum, :3, k]),
                                                       State_Store[valnum, 3, k], State_Store_wrt_Moon[valnum, 3, k],
                                                       State_Store[valnum, 4, k], State_Store_wrt_Moon[valnum, 4, k],
                                                       State_Store[valnum, 5, k], State_Store_wrt_Moon[valnum, 5, k],
                                                       np.linalg.norm(State_Store[valnum, 3:, k]),
                                                       np.linalg.norm(State_Store_wrt_Moon[valnum, 3:, k])))
                elif Parent_Num == 2:
                    scene.caption = (str_format.format(k, labels[valnum],
                                                       State_Store[valnum, 0, k], State_Store_wrt_Craft1[valnum, 0, k],
                                                       State_Store[valnum, 1, k], State_Store_wrt_Craft1[valnum, 1, k],
                                                       State_Store[valnum, 2, k], State_Store_wrt_Craft1[valnum, 2, k],
                                                       np.linalg.norm(State_Store[valnum, :3, k]),
                                                       np.linalg.norm(State_Store_wrt_Craft1[valnum, :3, k]),
                                                       State_Store[valnum, 3, k], State_Store_wrt_Craft1[valnum, 3, k],
                                                       State_Store[valnum, 4, k], State_Store_wrt_Craft1[valnum, 4, k],
                                                       State_Store[valnum, 5, k], State_Store_wrt_Craft1[valnum, 5, k],
                                                       np.linalg.norm(State_Store[valnum, 3:, k]),
                                                       np.linalg.norm(State_Store_wrt_Craft1[valnum, 3:, k])))
                k +=1
                if k==t-1:
                    k=0
        break
    elif event_series1 == "Martian System":
        t = 2000  # How many iterations
        DT = 20  # Your delta T jumps
        N = 4
        M_Mars = .64171e24
        M_Phobos = 1.066e16
        M_Deimos = 1.476e10
        ROI_Phobos = GET_SOI(M_Mars, M_Phobos, 9377.07e3)
        ROI_Deimos = GET_SOI(M_Mars, M_Deimos, 23462.89e3)
        # Defining my storage area for my position values
        State, Mass, Soft = Create_Mars_System(N)
        State_Store = np.zeros((N, 6, t))
        Accel = Get_Accel(N, State, Mass, Soft)

        for k in range(t):
            State = Update_State(N, State, Accel, DT, Mass, Soft)
            State_Store[:, :, k] = State[:, 0:]

            # Saving the State Store Information to a File
            while Save_to_File == "On":
                write_to_file = True
                filename = 'n_body_dat_' + str(N) + Scenario_Type + '.npy'
                if write_to_file:
                    with open(filename, 'wb') as f:
                        np.save(f, State_Store)

        # This is the animation stuff______________________________
        # ANIMATION FUNCTION
        # ___________________________________________________________________________

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


        line = []

        for i in range(N):
            line.append(plt.plot(State_Store[i, 0, 0], State_Store[i, 1, 0], State_Store[i, 2, 0], marker=".")[0])

        anim = FuncAnimation(fig1, func, frames=t, repeat=True, interval=1, fargs=(State_Store, line, N))
        # anim.save('rgb_cube.gif', dpi=80, writer='imagemagick', fps=24)
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        ax1.set_zlabel("z (m)")
        ax1.set_title("Orbital Trjectory")
        ax1.grid()
        ax1.legend(["Body 1", "Body 2", "Body 3", "Body 4", "Body 5"])
        plt.show()

        # VPTHON Section
        # ____________________________
        # L = 2
        # scene.range = L
        #scene.forward = vector(State_Store[1, 0, 0], State_Store[1, 1, 0], State_Store[1, 2, 0])
        running = True
        def Run(b):
            global running
            running = not running
            if running:
                b.text = "Pause"
            else:
                b.text = "Run"

        button(text="Pause", pos=scene.title_anchor, bind=Run)


        SPACECRAFT = sphere(pos=vector(State_Store[3, 0, 0], State_Store[3, 1, 0], State_Store[3, 2, 0]), radius=10e1,
                       color=color.blue, make_trail=True, trail_type='points', interval=10, retain=25,shininess=0)
        DEIMOS = sphere(pos=vector(State_Store[2, 0, 0], State_Store[2, 1, 0], State_Store[2, 2, 0]), radius=6.2e3,
                       color=color.orange, make_trail=True, trail_type='points', interval=10, retain=25,shininess=0)
        PHOBOS = sphere(pos=vector(State_Store[1, 0, 0], State_Store[1, 1, 0], State_Store[1, 2, 0]), radius=11.267e3,
                         color=color.white, make_trail=True, trail_type='points', interval=10, retain=25,shininess=0)
        MARS = sphere(pos=vector(State_Store[0, 0, 0], State_Store[0, 1, 0], State_Store[0, 2, 0]), radius=3.3895e6,
                     color=color.red, make_trail=True, trail_type='points', interval=10, retain=25,shininess=.1)

        Slabel = label(pos=vector(State_Store[3, 0, 0], State_Store[3, 1, 0], State_Store[3, 2, 0]), text='SPACECRAFT',
                       xoffset=10, height=10, color=color.blue)
        Dlabel = label(pos=vector(State_Store[2, 0, 0], State_Store[2, 1, 0], State_Store[2, 2, 0]), text='DEIMOS',
                       xoffset=10, height=10, color=color.yellow)
        Plabel = label(pos=vector(State_Store[1, 0, 0], State_Store[1, 1, 0], State_Store[1, 2, 0]), text='PHOBOS',
                       xoffset=10, height=10, color=color.white)
        Mlabel = label(pos=vector(State_Store[0, 0, 0], State_Store[0, 1, 0], State_Store[0, 2, 0]), text='MARS',
                       xoffset=10, height=10, color=color.red)

        #BOOTING UP THE SPHERES OF INFLUENCE

        ROI_DEIMOS = sphere(pos=vector(State_Store[2, 0, 0], State_Store[2, 1, 0], State_Store[2, 2, 0]), radius=ROI_Deimos,
                         color=color.white,opacity=.12)
        ROI_PHOBOS = sphere(pos=vector(State_Store[1, 0, 0], State_Store[1, 1, 0], State_Store[1, 2, 0]), radius=ROI_Phobos,
                       color=color.white,opacity=.12)
        k=0
        scale = 1e-10/1e2
        scene.range = 100000
        scene.fov=.01
        while k < t:
            if running:
                rate(10)
                SPACECRAFT.pos = vector(State_Store[3, 0, k], State_Store[3, 1, k], State_Store[3, 2, k])
                DEIMOS.pos = vector(State_Store[2, 0, k], State_Store[2, 1, k], State_Store[2, 2, k])
                PHOBOS.pos = vector(State_Store[1, 0, k], State_Store[1, 1, k], State_Store[1, 2, k])
                ROI_DEIMOS.pos = vector(State_Store[2, 0, k], State_Store[2, 1, k], State_Store[2, 2, k])
                ROI_PHOBOS.pos = vector(State_Store[1, 0, k], State_Store[1, 1, k], State_Store[1, 2, k])
                #scene.camera.follow(PHOBOS)
                MARS.pos = vector(State_Store[0, 0, k], State_Store[0, 1, k], State_Store[0, 2, k])
                Slabel.pos = vector(State_Store[3, 0, k], State_Store[3, 1, k], State_Store[3, 2, k])
                Dlabel.pos = vector(State_Store[2, 0, k], State_Store[2, 1, k], State_Store[2, 2, k])
                Plabel.pos = vector(State_Store[1, 0, k], State_Store[1, 1, k], State_Store[1, 2, k])
                Mlabel.pos = vector(State_Store[0, 0, k], State_Store[0, 1, k], State_Store[0, 2, k])
                k +=1

            #scene.forward = vector(State_Store[1, 0, k], State_Store[1, 1, k], State_Store[1, 2, k])


        break
    elif event_series1 == "Jovian System":
        t = 12*80  # How many iterations
        DT = 3600*24*28  # Your delta T jumps
        N = 4
        # Defining my storage area for my position values
        State, Mass, Soft = Create_The_Three_Body_Problem(N)
        State_Store = np.zeros((N, 6, t))
        Accel = Get_Accel(N, State, Mass, Soft)

        for k in range(t):
            State = Update_State(N, State, Accel, DT, Mass, Soft)
            State_Store[:, :, k] = State[:, 0:]

            # Saving the State Store Information to a File
            while Save_to_File == "On":
                write_to_file = True
                filename = 'n_body_dat_' + str(N) + Scenario_Type + '.npy'
                if write_to_file:
                    with open(filename, 'wb') as f:
                        np.save(f, State_Store)

        # This is the animation stuff______________________________
        # ANIMATION FUNCTION
        # ___________________________________________________________________________

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


        line = []

        for i in range(N):
            line.append(plt.plot(State_Store[i, 0, 0], State_Store[i, 1, 0], State_Store[i, 2, 0], marker=".")[0])

        anim = FuncAnimation(fig1, func, frames=t, repeat=True, interval=1, fargs=(State_Store, line, N))
        # anim.save('rgb_cube.gif', dpi=80, writer='imagemagick', fps=24)
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        ax1.set_zlabel("z (m)")
        ax1.set_title("Orbital Trjectory")
        ax1.grid()
        ax1.legend(["Body 1", "Body 2", "Body 3", "Body 4"])
        plt.show()

        State_Store_wrt_Body1 = Abs_Rel(0, State_Store)
        State_Store_wrt_Body2 = Abs_Rel(1, State_Store)
        State_Store_wrt_Body3 = Abs_Rel(2, State_Store)

        # VPTHON Section
        # ____________________________
        running = True

        # This function binds the action of the pause button that will be defined in the future
        def Run(b):
            global running
            running = not running
            if running:
                b.text = "Pause"
            else:
                b.text = "Run"

        button(text="Pause", pos=scene.title_anchor, bind=Run)

        # This function creates the menu that will be used to select which object you want to snap the view to.
        valnum = 0

        def Menu(m):
            val = m.selected
            global valnum
            if val == "Body1":
                scene.camera.follow(BODY1)
                valnum = 0
            elif val == "Body2":
                scene.camera.follow(BODY2)
                valnum = 1
            elif val == "Body3":
                scene.camera.follow(BODY3)
                valnum = 2
            elif val == "Body4":
                scene.camera.follow(BODY4)
                valnum = 3


        labels = ["Body1","Body2","Body3","Body4"]
        menu(choices=['Choose an Object','Body1','Body2','Body3','Body4'], bind=Menu, right=30, pos=scene.title_anchor)

        # This function creates the slide bar that will allow the user to change the rate at which the animation is played.
        def setspeed(s):
            wt.text = '{:1.2f}'.format(s.value)

        playrate = slider(min=1, max=3000, value=10, length=220, bind=setspeed, right=15, pos=scene.title_anchor)
        wt = wtext(text='{:1.2f}'.format(playrate.value), pos=scene.title_anchor)

        str_format = '''Time: {:.1f} JD
                                ----------
                                <b>{:s}</b> 
                                Absolute                 Relative     
                                X: {:.2f} m              X: {:.2f} m
                                Y: {:.2f} m              Y: {:.2f} m
                                Z: {:.2f} m              Z: {:.2f} m
                                VX: {:.2f} m/s           VX: {:.2f} m/s
                                VY: {:.2f} m/s           VY: {:.2f} m/s
                                VZ: {:.2f} m/s           VZ: {:.2f} m/s'''



        BODY1 = sphere(pos=vector(State_Store[0, 0, 0], State_Store[0, 1, 0], State_Store[0, 2, 0]),
                       radius=C.C["Body1"]["Radius"], make_trail=True, trail_type='curve', interval=30, retain=250,
                       shininess=.1, texture={'file': "\Images\Sun.jpg"})
        BODY2 = sphere(pos=vector(State_Store[1, 0, 0], State_Store[1, 1, 0], State_Store[1, 2, 0]),
                      radius=C.C["Body2"]["Radius"], make_trail=True, trail_type='curve', interval=30, retain=300,
                      shininess=0.1, texture={'file': "\Images\Sun.jpg"})
        BODY3 = sphere(pos=vector(State_Store[2, 0, 0], State_Store[2, 1, 0], State_Store[2, 2, 0]),
                        radius=C.C["Body3"]["Radius"], make_trail=True, trail_type='curve',
                        interval=30, retain=2500, shininess=0.1,texture={'file': "\Images\Sun.jpg"})
        BODY4 = sphere(pos=vector(State_Store[3, 0, 0], State_Store[3, 1, 0], State_Store[3, 2, 0]),
                        radius=C.C["Body3"]["Radius"], make_trail=True, trail_type='curve',
                        interval=30, retain=2500, shininess=0.1,texture={'file': "\Images\Earth.jpg"})
        B1label = label(pos=vector(State_Store[0, 0, 0], State_Store[0, 1, 0], State_Store[0, 2, 0]), text='Body1',
                       xoffset=10, height=10, color=color.white)
        B2label = label(pos=vector(State_Store[1, 0, 0], State_Store[1, 1, 0], State_Store[1, 2, 0]), text='Body2',
                       xoffset=10, height=10, color=color.white)
        B3label = label(pos=vector(State_Store[2, 0, 0], State_Store[2, 1, 0], State_Store[2, 2, 0]), text='Body3',
                       xoffset=10, height=10, color=color.white)
        B4label = label(pos=vector(State_Store[3, 0, 0], State_Store[3, 1, 0], State_Store[3, 2, 0]), text='Body4',
                        xoffset=10, height=10, color=color.blue)
        k = 0
        #Changes the axes to the proper visual orientation
        scene.forward = vector(0, 0, 1)
        scene.up = vector(1, 0, 0)
        #Changes the fov and scale to attempt to fit everything into the image
        scale = 1e-10 / 1e2
        scene.range = 1000000
        scene.fov = .0001

        while k < t:
            if running:
                rate(playrate.value)

                BODY1.pos = vector(State_Store[0, 0, k], State_Store[0, 1, k], State_Store[0, 2, k])
                BODY2.pos = vector(State_Store[1, 0, k], State_Store[1, 1, k], State_Store[1, 2, k])
                BODY3.pos = vector(State_Store[2, 0, k], State_Store[2, 1, k], State_Store[2, 2, k])
                BODY4.pos = vector(State_Store[3, 0, k], State_Store[3, 1, k], State_Store[3, 2, k])
                B1label.pos = vector(State_Store[0, 0, k], State_Store[0, 1, k], State_Store[0, 2, k])
                B2label.pos = vector(State_Store[1, 0, k], State_Store[1, 1, k], State_Store[1, 2, k])
                B3label.pos = vector(State_Store[2, 0, k], State_Store[2, 1, k], State_Store[2, 2, k])
                B4label.pos = vector(State_Store[3, 0, k], State_Store[3, 1, k], State_Store[3, 2, k])

                # Creating the text at the bottom.
                # if valnum==0:
                scene.caption = (str_format.format(k, labels[valnum],
                                                   State_Store[valnum, 0, k], State_Store_wrt_Earth[valnum, 0, k],
                                                   State_Store[valnum, 1, k], State_Store_wrt_Earth[valnum, 1, k],
                                                   State_Store[valnum, 2, k], State_Store_wrt_Earth[valnum, 2, k],
                                                   State_Store[valnum, 3, k], State_Store_wrt_Earth[valnum, 3, k],
                                                   State_Store[valnum, 4, k], State_Store_wrt_Earth[valnum, 4, k],
                                                   State_Store[valnum, 5, k], State_Store_wrt_Earth[valnum, 5, k]))
                k += 1
                if k == t - 1:
                    k = 0
        break
    elif event_series1 == "TRAPIST-1 System":
        t = 12*80  # How many iterations
        DT = 3600*24*28  # Your delta T jumps
        N = 4
        # Defining my storage area for my position values
        State, Mass, Soft = Create_The_Three_Body_Problem(N)
        State_Store = np.zeros((N, 6, t))
        Accel = Get_Accel(N, State, Mass, Soft)

        for k in range(t):
            State = Update_State(N, State, Accel, DT, Mass, Soft)
            State_Store[:, :, k] = State[:, 0:]

            # Saving the State Store Information to a File
            while Save_to_File == "On":
                write_to_file = True
                filename = 'n_body_dat_' + str(N) + Scenario_Type + '.npy'
                if write_to_file:
                    with open(filename, 'wb') as f:
                        np.save(f, State_Store)

        # This is the animation stuff______________________________
        # ANIMATION FUNCTION
        # ___________________________________________________________________________

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


        line = []

        for i in range(N):
            line.append(plt.plot(State_Store[i, 0, 0], State_Store[i, 1, 0], State_Store[i, 2, 0], marker=".")[0])

        anim = FuncAnimation(fig1, func, frames=t, repeat=True, interval=1, fargs=(State_Store, line, N))
        # anim.save('rgb_cube.gif', dpi=80, writer='imagemagick', fps=24)
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        ax1.set_zlabel("z (m)")
        ax1.set_title("Orbital Trjectory")
        ax1.grid()
        ax1.legend(["Body 1", "Body 2", "Body 3", "Body 4"])
        plt.show()

        State_Store_wrt_Body1 = Abs_Rel(0, State_Store)
        State_Store_wrt_Body2 = Abs_Rel(1, State_Store)
        State_Store_wrt_Body3 = Abs_Rel(2, State_Store)

        # VPTHON Section
        # ____________________________
        running = True

        # This function binds the action of the pause button that will be defined in the future
        def Run(b):
            global running
            running = not running
            if running:
                b.text = "Pause"
            else:
                b.text = "Run"

        button(text="Pause", pos=scene.title_anchor, bind=Run)

        # This function creates the menu that will be used to select which object you want to snap the view to.
        valnum = 0

        def Menu(m):
            val = m.selected
            global valnum
            if val == "Body1":
                scene.camera.follow(BODY1)
                valnum = 0
            elif val == "Body2":
                scene.camera.follow(BODY2)
                valnum = 1
            elif val == "Body3":
                scene.camera.follow(BODY3)
                valnum = 2
            elif val == "Body4":
                scene.camera.follow(BODY4)
                valnum = 3


        labels = ["Body1","Body2","Body3","Body4"]
        menu(choices=['Choose an Object','Body1','Body2','Body3','Body4'], bind=Menu, right=30, pos=scene.title_anchor)

        # This function creates the slide bar that will allow the user to change the rate at which the animation is played.
        def setspeed(s):
            wt.text = '{:1.2f}'.format(s.value)

        playrate = slider(min=1, max=3000, value=10, length=220, bind=setspeed, right=15, pos=scene.title_anchor)
        wt = wtext(text='{:1.2f}'.format(playrate.value), pos=scene.title_anchor)

        str_format = '''Time: {:.1f} JD
                                ----------
                                <b>{:s}</b> 
                                Absolute                 Relative     
                                X: {:.2f} m              X: {:.2f} m
                                Y: {:.2f} m              Y: {:.2f} m
                                Z: {:.2f} m              Z: {:.2f} m
                                VX: {:.2f} m/s           VX: {:.2f} m/s
                                VY: {:.2f} m/s           VY: {:.2f} m/s
                                VZ: {:.2f} m/s           VZ: {:.2f} m/s'''



        BODY1 = sphere(pos=vector(State_Store[0, 0, 0], State_Store[0, 1, 0], State_Store[0, 2, 0]),
                       radius=C.C["Body1"]["Radius"], make_trail=True, trail_type='curve', interval=30, retain=250,
                       shininess=.1, texture={'file': "\Images\Sun.jpg"})
        BODY2 = sphere(pos=vector(State_Store[1, 0, 0], State_Store[1, 1, 0], State_Store[1, 2, 0]),
                      radius=C.C["Body2"]["Radius"], make_trail=True, trail_type='curve', interval=30, retain=300,
                      shininess=0.1, texture={'file': "\Images\Sun.jpg"})
        BODY3 = sphere(pos=vector(State_Store[2, 0, 0], State_Store[2, 1, 0], State_Store[2, 2, 0]),
                        radius=C.C["Body3"]["Radius"], make_trail=True, trail_type='curve',
                        interval=30, retain=2500, shininess=0.1,texture={'file': "\Images\Sun.jpg"})
        BODY4 = sphere(pos=vector(State_Store[3, 0, 0], State_Store[3, 1, 0], State_Store[3, 2, 0]),
                        radius=C.C["Body3"]["Radius"], make_trail=True, trail_type='curve',
                        interval=30, retain=2500, shininess=0.1,texture={'file': "\Images\Earth.jpg"})
        B1label = label(pos=vector(State_Store[0, 0, 0], State_Store[0, 1, 0], State_Store[0, 2, 0]), text='Body1',
                       xoffset=10, height=10, color=color.white)
        B2label = label(pos=vector(State_Store[1, 0, 0], State_Store[1, 1, 0], State_Store[1, 2, 0]), text='Body2',
                       xoffset=10, height=10, color=color.white)
        B3label = label(pos=vector(State_Store[2, 0, 0], State_Store[2, 1, 0], State_Store[2, 2, 0]), text='Body3',
                       xoffset=10, height=10, color=color.white)
        B4label = label(pos=vector(State_Store[3, 0, 0], State_Store[3, 1, 0], State_Store[3, 2, 0]), text='Body4',
                        xoffset=10, height=10, color=color.blue)
        k = 0
        #Changes the axes to the proper visual orientation
        scene.forward = vector(0, 0, 1)
        scene.up = vector(1, 0, 0)
        #Changes the fov and scale to attempt to fit everything into the image
        scale = 1e-10 / 1e2
        scene.range = 1000000
        scene.fov = .0001

        while k < t:
            if running:
                rate(playrate.value)

                BODY1.pos = vector(State_Store[0, 0, k], State_Store[0, 1, k], State_Store[0, 2, k])
                BODY2.pos = vector(State_Store[1, 0, k], State_Store[1, 1, k], State_Store[1, 2, k])
                BODY3.pos = vector(State_Store[2, 0, k], State_Store[2, 1, k], State_Store[2, 2, k])
                BODY4.pos = vector(State_Store[3, 0, k], State_Store[3, 1, k], State_Store[3, 2, k])
                B1label.pos = vector(State_Store[0, 0, k], State_Store[0, 1, k], State_Store[0, 2, k])
                B2label.pos = vector(State_Store[1, 0, k], State_Store[1, 1, k], State_Store[1, 2, k])
                B3label.pos = vector(State_Store[2, 0, k], State_Store[2, 1, k], State_Store[2, 2, k])
                B4label.pos = vector(State_Store[3, 0, k], State_Store[3, 1, k], State_Store[3, 2, k])

                # Creating the text at the bottom.
                # if valnum==0:
                scene.caption = (str_format.format(k, labels[valnum],
                                                   State_Store[valnum, 0, k], State_Store_wrt_Earth[valnum, 0, k],
                                                   State_Store[valnum, 1, k], State_Store_wrt_Earth[valnum, 1, k],
                                                   State_Store[valnum, 2, k], State_Store_wrt_Earth[valnum, 2, k],
                                                   State_Store[valnum, 3, k], State_Store_wrt_Earth[valnum, 3, k],
                                                   State_Store[valnum, 4, k], State_Store_wrt_Earth[valnum, 4, k],
                                                   State_Store[valnum, 5, k], State_Store_wrt_Earth[valnum, 5, k]))
                k += 1
                if k == t - 1:
                    k = 0
        break
    elif event_series1 == "Cancel" or event_series1 == sg.WIN_CLOSED:
        break

# ___________________________________________________________________________
# Initial_Energy = Total_Energy(N, State_Store, Mass, 0)
# Final_Energy = Total_Energy(N, State_Store, Mass, -1)
# #print(Initial_Energy)hjjbh
# #print(Final_Energy)





