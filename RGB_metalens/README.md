This is an example of RGB metalens designs. Rasmus [@raelch](https://github.com/raelch) generated designs using his own code and reported the FOMs, which were validated on Meep by [@Mo](https://github.com/mochen4). Mo also generated designs using Meep. The results are plotted for comparison.

A schematic of the problem setup is shown below. The design region is 10μm by 1μm, with a material of index n=2.4, and lies on a substrate of the same material. The source is a planewave with in-plane (Ex) polarization. The lens tries to focus incident light at 2.4μm away for wavelengths of 450nm, 550nm, and 650nm.

<p align="center">
<img width="594" alt="Screen Shot 2022-03-01 at 11 19 20" src="https://user-images.githubusercontent.com/25192039/156206561-cd0fe0f2-a889-49c8-a377-ee085f62df20.png">
</p>




``python3 metalens_check.py path/to/design_file [--resolution]`` checks designs (in .csv format) on Meep, and reports FOMs and lengthscales.
