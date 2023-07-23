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
            if file_name[0].upper() == 'M': design_size = (10400, 1400)
            elif file_name[0].upper() == 'R': design_size = (11000, 2000)
            elif file_name[0].upper() == 'W': design_size = (11600, 1160)
            else: AssertionError("Unknown file name.")

            design_pattern = np.genfromtxt(path+file_name, delimiter=',')
            solid_mls, void_mls = imageruler.minimum_length_solid_void(design_pattern, design_size)
            print(path+file_name, solid_mls, void_mls, min(solid_mls, void_mls))
