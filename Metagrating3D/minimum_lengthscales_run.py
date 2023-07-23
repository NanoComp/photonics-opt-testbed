import numpy as np
import os
import imageruler

theta_d = np.radians(50.0)  # deflection angle
wvl = 1050  # wavelength
px = wvl/np.sin(theta_d)  # period in x
py = 0.5*wvl

path = 'designs/'
files = os.listdir(path)
files.sort()
print('Design file, solid minimum lengthscale (nm), void minimum lengthscale (nm), minimum lengthscale (nm)')

for file in files:  
    file_name = str(file)
    if file_name[-3:] == 'csv':
        design_pattern = np.loadtxt(path+file_name, delimiter=',')
        design_dimension = np.ndim(design_pattern)

        if design_dimension == 2: design_size = (px, py)
        elif design_dimension == 1: design_size = px
        else: AssertionError("Invalid dimension of the design pattern.")

        solid_mls, void_mls = imageruler.minimum_length_solid_void(design_pattern, design_size, periodic_axes=(0,1))
        print(file_name, solid_mls, void_mls, min(solid_mls, void_mls))