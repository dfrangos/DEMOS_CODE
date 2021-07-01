import numpy as np

r_Earth=6378e3

def density(r):
    #The input r is the magnitude of your sat position vector in ECI (meters)

    z=(r-r_Earth)*(1e-3)

    # base altitude (km) [h_0]
    h0 = [0, 25, 30, 40, 50, 60, 70,
         80,  90, 100, 110, 120, 130, 140,
         150, 180, 200, 250, 300, 350, 400,
         450, 500, 600, 700, 800, 900, 1000]

    # nominal density (kg/m^3) [rho_0]
    rho0 = [1.225, 4.008e-2, 1.841e-2, 3.996e-3, 1.027e-3, 3.097e-4,
         8.283e-5, 1.846e-5, 3.416e-6, 5.606e-7, 9.708e-8, 2.222e-8,
         8.152e-9, 3.831e-9, 2.076e-9, 5.194e-10, 2.541e-10, 6.073e-11,
         1.916e-11, 7.014e-12, 2.803e-12, 1.184e-12, 5.215e-13,
         1.137e-13, 3.070e-14, 1.136e-14, 5.759e-15, 3.561e-15]

    # scale height (km) [H]
    H = [7.310, 6.427, 6.546, 7.360, 8.342, 7.583, 6.661,
         5.927, 5.533, 5.703, 6.782, 9.973, 13.243, 16.322,
         21.652, 27.974, 34.934, 43.342, 49.755, 54.513, 58.019,
         60.980, 65.654, 76.377, 100.587, 147.203, 208.020]

     # error check
    if z > 1000:
        z = 1000
    elif z < 0:
        z = 0

    ii = 0
    for jj in range(0, 26):
        if h0[jj] <= z < h0[jj + 1]:
            ii = jj

    if z == 1000:
        ii = 26

    # rho = rho_0 * exp( -( h_ellp - h_0 ) / H )
    density = rho0[ii] * np.exp(-(z - h0[ii]) / H[ii])

    return density
















