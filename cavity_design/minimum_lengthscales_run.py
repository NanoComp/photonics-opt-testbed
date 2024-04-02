import numpy as np
import os
import imageruler

print('Design file, solid minimum lengthscale (nm), void minimum lengthscale (nm), minimum lengthscale (nm)')
pixel_size = 5

for path in ['Mo/', 'Göktuğ/']:
    files = os.listdir(path)
    files.sort()

    for file in files:  
        file_name = str(file)
        if file_name[-3:] == 'csv':
            design_pattern = np.genfromtxt(path+file_name, delimiter=',')
            
            binary_design_pattern = design_pattern > 0.5
            solid_mls_pixels, void_mls_pixels = imageruler.minimum_length_scale(binary_design_pattern)
            solid_mls = solid_mls_pixels * pixel_size
            void_mls = void_mls_pixels * pixel_size
            print(path+file_name, solid_mls, void_mls, min(solid_mls, void_mls))
