This is an example of 3D metagrating deflector designs. Evan Wang [@evanwang13](https://github.com/evanwang13) generated designs using his own code and reported the FOMs. The simulation is performed by [RETICOLO](https://www.lp2n.institutoptique.fr/equipes-de-recherche-du-lp2n/light-complex-nanostructures) RCWA. 

A schematic of the problem setup is shown below. 

![schematic](/Metagrating3D/metagrating3d.png)


The 3D metagratings consist of freeform silicon patterns and deflect normally-incident light (plane wave) to the +1 diffraction order. FOM of this problem is the deflection efficiency in the desired drection for both TE and TM polarization light. The structure is periodic in x, y directions, and uniform in z direction. The relevant parameters are defined below:

- **Refractive index**: The refractive index of silicon is 3.45, and refractive index of silica is 1.45.

- **Deflection angle**: The desired deflection angle, 𝜃, is given by the angle to the normal. The azimuth, φ, is assumed to be zero.

- **Period**: Each period is defined in the x-direction, along the plane of deflection, and the y-direction, perpendicular to the plane of deflection. The grating period in the x-direction, Px, is related to the desired deflection angle 𝜃 by Px = 𝜆/sin(𝜃). The grating period in the y-direction is typically subwavelength to prevent diffraction in the y-direction.

- **Thickness**: Thickness of the silicon device region in the z-direction, given in nm.

- **Polarization**: Polarization is defined relative to the deflection plane. In TE polarization, the electric field is perpendicular to the deflection plane. In TM polarization, the magnetic field is perpendicular to the deflection plane.

- **Unit Cell**: The metagrating unit cell is subdivided into a Nx by Ny grid. The unit cell is defined by a binary Nx × Ny matrix, with a 1 representing silicon and a 0 representing air.

- **Symmetry**: Reflection symmetry in the y-direction (across the x-axis) is enforced in all devices.

- **Efficiency**: Deflection efficiency is defined as the intensity of light deflected to the desired diffraction order, normalized to the light intensity incident from within a semi-infinite silica substrate.

As an example, optimized metagrating designs with following parameters can be found in this repo:

- **Wavelength**: 1050 nm
- **Deflection angle**: 50 degree
- **Period**: Px = 1050/sin(50) nm, Py = 0.5* 1050 nm
- **Thickness**: 325 nm
- **Polarization**: TE and TM
- **Unit Cell**: Nx = 118, Ny = 45

The deflection efficiencies for the example device in this repo are, TE: 95.14%, TM: 91.14%. 

