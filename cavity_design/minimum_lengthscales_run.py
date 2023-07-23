import numpy as np
import os
import imageruler

print('Design file, solid minimum lengthscale (nm), void minimum lengthscale (nm), minimum lengthscale (nm)')
design_size = (1550, 1550)

for path in ['Mo/', 'Göktuğ/']:
    files = os.listdir(path)
    files.sort()

    for file in files:  
        file_name = str(file)
        if file_name[-3:] == 'csv':
            design_pattern = np.genfromtxt(path+file_name, delimiter=',')
            solid_mls, void_mls = imageruler.minimum_length_solid_void(design_pattern, design_size)
            print(path+file_name, solid_mls, void_mls, min(solid_mls, void_mls))
