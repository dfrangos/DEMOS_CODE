import pandas as pd
G=6.67408e-11

#Units are in kg, meters, coverted to seconds and then, (m^3/s^2)
C={ "Sun":      {"Mass": 1988500e24,      "Radius": 695700e3,       "Rot_Rate": 609.12*3600,            "Mu": 1.32712e20},
    "Mercury":  {"Mass": 0.33011e24,      "Radius": 2439.7e3,       "Rot_Rate": 1407.6*3600,            "Mu": 2.20329e13},
    "Venus":    {"Mass": 4.8675e24,       "Radius": 6051.8e3,       "Rot_Rate": -243.025*24*3600,       "Mu": 3.248599e14},
    "Earth":    {"Mass": 5.972e24,        "Radius": 6378.1363e3,    "Rot_Rate": 23.9345*3600,           "Mu": 3.9860044188e14},
    "Moon":     {"Mass": 7.34767309e22,   "Radius": 1738.1e3,       "Rot_Rate": 27.3217*24*3600,        "Mu": 4.90486959e12},
    "Mars":     {"Mass": .64171e24,       "Radius": 3376.2e3,       "Rot_Rate": 24.6229*3600,           "Mu": 4.28284e13},
    "Phobos":   {"Mass": 1.066e16,        "Radius": 11.267e3,       "Rot_Rate": 459*60,                 "Mu": 7.161e5},
    "Deimos":   {"Mass": 1.5e15,          "Radius": 6.2e3,          "Rot_Rate": 30.26*3600,             "Mu": 100111.2},
    "Jupiter":  {"Mass": 1.8988e27,       "Radius": 69.911e3,       "Rot_Rate": 9.9*3600,               "Mu": 1.26727431e17},
    "Io":       {"Mass": 8.9314e22,       "Radius": 1823.6e3,       "Rot_Rate": 42.45*3600,             "Mu": 5.96088781e12},
    "Saturn":   {"Mass": 5.683e26,        "Radius": 58232e3,        "Rot_Rate": 10.7*3600,              "Mu": 3.79287e16},
    "Uranus":   {"Mass": 8.681e25,        "Radius": 25362e3,        "Rot_Rate": 17.22*3600,             "Mu": 5.7937688480e15},
    "Neptune":  {"Mass": 1.024e26,        "Radius": 24622e3,        "Rot_Rate": 16.1*3600,              "Mu": 6.8342579200e15},
    "Pluto":    {"Mass": 1.30900e22,      "Radius": 1188.3e3,       "Rot_Rate": 6.4*24*3600,            "Mu": 8.7363707e11},
    "Craft1":   {"Mass": 500,             "Radius": 50,             "Rot_Rate": 2*3600,                 "Mu": 4.28284e13},
    "Body1":    {"Mass": 1988500e24*1.09, "Radius": 695700e3*1.227, "Rot_Rate": 38.483*24*3600,         "Mu": 4.28284e13},
    "Body2":    {"Mass": 1988500e24*.92,  "Radius": 695700e3*.865,  "Rot_Rate": 35.438*24*3600,         "Mu": 4.28284e13},
    "Body3":    {"Mass": 1988500e24*.42,  "Radius": 695700e3*.465,  "Rot_Rate": 32.31*3600,             "Mu": 4.28284e13},
    "Body4":    {"Mass": 5.972e24,        "Radius": 6378.1363e3,    "Rot_Rate": 23.9345*3600,           "Mu": 3.9860044188e14},}

print(C["Earth"]["Mass"])

Data = {'Body':[
          'Sun',      'Mercury',    'Venus',    'Earth',  'Moon',        'Mars',     'Phobos',  'Deimos', 'Jupiter',  'Io',      'Saturn',  'Uranus',   'Neptune', 'Pluto',    'Craft1', 'Body1',          'Body2',        'Body3',        'Body4'],
'Mass':[  1988500e24,  0.33011e24,  4.8675e24,  5.972e24, 7.34767309e22, .64171e24,  1.066e16,  1.5e15,   1.8988e27,  8.9314e22, 5.683e26,   8.681e25,  1.024e26,  1.30900e22,  500,      1988500e24*1.09, 1988500e24*.92, 1988500e24*.42, 5.972e24]}

Data = pd.DataFrame(Data)

df=pd.DataFrame(C)

#Data.iloc[[1,2]]

# Data = pd.DataFrame({"Name": ["Sun", 'Mercury'],"Mass":[198,.33]}
# Cdfmercury = Cdf.loc['Name'=='Mercury']
#print(df.iloc[[1,2]])

