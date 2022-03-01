This is an example of RGB metalens designs. Rasmus [@raelch](https://github.com/raelch) generated designs using his own code and reported the FOMs, which were validated on Meep by Mo [@mochen4](https://github.com/mochen4). Mo also generated designs using Meep.

A schematic of the problem setup is shown below. The design region is 10μm by 1μm, with a material of index n=2.4, and lies on a substrate of the same material. The source is a planewave with in-plane (Ex) polarization. The lens tries to focus incident light at 2.4μm away for wavelengths of 450nm, 550nm, and 650nm.

<p align="center">
<img width="594" alt="Screen Shot 2022-03-01 at 11 19 20" src="https://user-images.githubusercontent.com/25192039/156206561-cd0fe0f2-a889-49c8-a377-ee085f62df20.png">
</p>

The script [metalens_check.py](https://github.com/mochen4/photonics-opt-testbed/blob/RGB/RGB_metalens/metalens_check.py) checks different designs under this setup, and reports FOMs and lengthscales. Sample usage: ``python3 metalens_check.py path/to/design_file [--resolution RESOLUTION]``, where the design_file should be in ``.csv`` format, and ``resolution`` is an optional argument for the resolution of the simulation, with a default of 50.

The reported FOM is calculated as an average of |E|^2 over the three wavelengths, and normalized by the intensity when no lens is present. Rasmus designed structures with lengthscales 123nm, 209nm, and 256nm, and reported FOMs of 16, 11.7, and 8.1, respectively. Mo validated the designs on Meep and found FOMs of 14.75, 10.7, and 7.8.

Mo generated structures with lengthscales 88nm, 120nm, and 216nm in Meep, and found FOMs of 12.6, 11.2, and 7.7. A plot of FOMs vs Lengthscale is shown below. 
<p align="center">
<img src="https://user-images.githubusercontent.com/25192039/156216247-450186af-5c7f-4460-9d78-563cfb53e1da.png" width="600" />
</p>
