import numpy as np
import os
import imageruler

print('Design file, solid minimum lengthscale (nm), void minimum lengthscale (nm), minimum lengthscale (nm)')

for path in ['Ex/', 'Ez/']:
    files = os.listdir(path)
    files.sort()

    for file in files:  
        file_name = str(file)
        if file_name[-3:] == 'csv':
            design_pattern = np.genfromtxt(path+file_name, delimiter=',')

            if file_name[0].upper() == 'M': design_size = 20*np.array(design_pattern.shape)
            elif file_name[0].upper() in ('R', 'W'): design_size = 10*np.array(design_pattern.shape)
            else: AssertionError("Unknown file name.")

            solid_mls, void_mls = imageruler.minimum_length_solid_void(design_pattern, design_size)
            print(path+file_name, solid_mls, void_mls, min(solid_mls, void_mls))
