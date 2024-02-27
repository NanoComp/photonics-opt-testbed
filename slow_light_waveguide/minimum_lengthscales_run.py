import numpy as np
import os
import imageruler

path = './'
files = os.listdir(path)
files.sort()
print('Design file, solid minimum lengthscale (a), void minimum lengthscale (a), minimum lengthscale (a)')

design_size = (1, 10.2)

for file_name in ['Design_Dnum_2.csv', 'HigRes_DesMatch_Opt_Dnum_2.csv']:  
    design_pattern = np.loadtxt(path+file_name, delimiter=',')
    design_dimension = np.ndim(design_pattern)

    solid_mls, void_mls = imageruler.minimum_length_solid_void(design_pattern, design_size, periodic_axes=(0,1))
    print(file_name, solid_mls, void_mls, min(solid_mls, void_mls))