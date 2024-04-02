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
        if design_pattern.ndim == 1:
            design_pattern = design_pattern[:, np.newaxis]
        pixel_size = px / design_pattern.shape[0]

        binary_design_pattern = design_pattern > 0.5
        solid_mls_pixels, void_mls_pixels = imageruler.minimum_length_scale(
            binary_design_pattern, periodic=(True, True)
        )
        solid_mls = solid_mls_pixels * pixel_size
        void_mls = void_mls_pixels * pixel_size
        print(path+file_name, solid_mls, void_mls, min(solid_mls, void_mls))
