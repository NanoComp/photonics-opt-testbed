This is an example of 3D metagrating deflector designs. Evan Wang [@evanwang13](https://github.com/jonfanlab/Metagrating-Topology-Optimization) generated designs using his own code and reported the FOMs. The simulation is performed by [RETICOLO](https://www.lp2n.institutoptique.fr/equipes-de-recherche-du-lp2n/light-complex-nanostructures) RCWA. 

A schematic of the problem setup is shown below. 

![schematic](/Metagrating3D/metagrating3d.png)


The 3D metagratings consist of freeform silicon patterns and deflect normally-incident light (plane wave) to the +1 diffraction order. The FOM of this problem is the diffraction efficiency in the desired direction for TM polarization light. The structure is periodic in x and y directions, and uniform in z direction. The relevant parameters are defined below:

- **Refractive index**: The refractive index of silicon is 3.45, and refractive index of silica is 1.45.

- **Deflection angle**: The desired deflection angle, 𝜃, is given by the angle to the normal. The azimuth, φ, is assumed to be zero.

- **Period**: Each period is defined in the x-direction, along the plane of deflection, and the y-direction, perpendicular to the plane of deflection. The grating period in the x-direction, Px, is related to the desired deflection angle 𝜃 by Px = 𝜆/sin(𝜃). The grating period in the y-direction is typically subwavelength to prevent diffraction in the y-direction.

- **Thickness**: Thickness of the silicon device region in the z-direction, given in nm.

- **Polarization**: Polarization is defined relative to the deflection plane. In TE polarization, the electric field is perpendicular to the deflection plane. In TM polarization, the magnetic field is perpendicular to the deflection plane.

- **Unit Cell**: The metagrating unit cell is subdivided into a Nx by Ny grid. The unit cell is defined by a binary Nx × Ny matrix, with a 1 representing silicon and a 0 representing air.

- **Symmetry**: Reflection symmetry in the y-direction (across the x-axis) is enforced in all devices.

- **Efficiency**: Diffraction efficiency is defined as the intensity of light deflected to the desired diffraction order, normalized to the light intensity incident from within a semi-infinite silica substrate. 

As an example, optimized metagrating designs with following parameters can be found in this repo:

- **Wavelength**: 1050 nm
- **Deflection angle**: 50 degree
- **Period**: Px = 1050/sin 50 $^\circ$ nm, Py = 0.5 × 1050 nm
- **Thickness**: 325 nm
- **Polarization**: TM
- **Unit Cell**: Nx = 472, Ny = 180

The diffraction efficiencies for the example devices in this repo are:
- **Device1**: TM 95.7% (RETICOLO/RCWA), 95.5% (MEEP/FDTD)
- **Device2**: TM 93.3% (RETICOLO/RCWA), 93.8% (MEEP/FDTD)
- **Device3**: TM 96.6% (RETICOLO/RCWA), 95.0% (MEEP/FDTD)
- **Device4**: TM 93.3% (RETICOLO/RCWA), 92.5% (MEEP/FDTD)
- **Device5**: TM 84.1% (RETICOLO/RCWA), 84.3% (MEEP/FDTD)

For RETICOLO results, `device1.mat`, `device2.mat`, and `device3.mat` contain all optimization parameters. The final design patterns (Nx = 118, Ny = 45) are in `device1.csv`, `device2.csv`, and `device3.csv`, while `device1_interpolated.csv`, `device2_interpolated.csv`, and `device3_interpolated.csv` contain design patterns interpolated to a high resolution (Nx = 472, Ny = 180).

For MEEP results, `device4.csv` and `device5.csv` contain final design patterns while `device4_interpolated.csv` and `device5_interpolated.csv` contain design patterns interpolated to a high resolution. The files `device5.csv` and `device5_interpolated.csv` contain 1d arrays, which correspond to design patterns composed of stripes.

The file `metagrating_meep.py` is the MEEP script that computes the diffraction efficiency of a metagrating with a given design pattern. The files `metagrating_meep_opt_2d.py` and `metagrating_meep_opt_1d.py` are the MEEP scripts for optimization, which maximize the diffraction efficiencies based on 2D and 1D design patterns, respectively.