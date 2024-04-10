import numpy as np
import os
import imageruler

path = './'
files = os.listdir(path)
files.sort()
print('Design file, solid minimum lengthscale (a), void minimum lengthscale (a), minimum lengthscale (a)')

for file_name in ['Design_Dnum_2.csv', 'HigRes_DesMatch_Opt_Dnum_2.csv']:  
    design_permittivity = np.loadtxt(path+file_name, delimiter=',')

    midpoint = (np.amax(design_permittivity) + np.amin(design_permittivity)) / 2
    binary_design_pattern = design_permittivity > midpoint
    solid_mls, void_mls = imageruler.minimum_length_scale(binary_design_pattern, periodic=(True, True))
    print(file_name, solid_mls, void_mls, min(solid_mls, void_mls))
