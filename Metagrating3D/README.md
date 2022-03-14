This is an example of 3D metagrating deflector designs. Evan Wang [@evanwang13](https://github.com/evanwang13) generated designs using his own code and reported the FOMs. The simulation is performed by [RETICOLO](https://www.lp2n.institutoptique.fr/equipes-de-recherche-du-lp2n/light-complex-nanostructures) RCWA. 

A schematic of the problem setup is shown below. 

![schematic](https://https://github.com/jiaqi-jiang/photonics-opt-testbed/blob/jiaqi-jiang-metagrating3d/Metagrating3D/metagrating3d.png)


The 3D metagratings consist of freeform silicon patterns and deflect normally-incident light to the +1 diffraction order. The structure is periodic in x, y directions, and uniform in z direction. The relevant parameters are defined below:

-Refractive index: The refractive index of silicon is taken from this GitHub repository, and only the real part of the index is used to simplify the design problem.

-Deflection angle: The desired deflection angle, 𝜃, is given by the angle to the normal. The azimuth, φ, is assumed to be zero.

-Period: Each period is defined in the x-direction, along the plane of deflection, and the y-direction, perpendicular to the plane of deflection. The grating period in the x-direction, Px, is related to the desired deflection angle 𝜃 by Px = 𝜆/sin(𝜃). The grating period in the y-direction is typically subwavelength to prevent diffraction in the y-direction.

-Thickness: Thickness of the silicon device region in the z-direction, given in nm.

-Polarization: Polarization is defined relative to the deflection plane. In TE polarization, the electric field is perpendicular to the deflection plane. In TM polarization, the magnetic field is perpendicular to the deflection plane.

-Unit Cell: The metagrating unit cell is subdivided into a Nx by Ny grid. The unit cell is defined by a binary Nx × Ny matrix, with a 1 representing silicon and a 0 representing air.

-Symmetry: Reflection symmetry in the y-direction (across the x-axis) is enforced in all devices.

-Efficiency: Deflection efficiency is defined as the intensity of light deflected to the desired diffraction order, normalized to the light intensity incident from within a semi-infinite silica substrate.

The reported FOM is calculated as an average of |E|^2 over the three wavelengths, and normalized by the intensity when no lens is present. Rasmus designed structures with lengthscales 123nm, 209nm, and 256nm, and reported FOMs of 16, 11.7, and 8.1, respectively. Mo validated the designs on Meep and found FOMs of 14.75, 10.7, and 7.8.

