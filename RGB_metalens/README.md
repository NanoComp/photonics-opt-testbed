This is an example of RGB metalens designs. 

For Ex polarization and designing lens with substrate, Rasmus [@raelch](https://github.com/raelch) generated designs using his own FEM code and reported the FOMs, which were validated on Meep by Mo [@mochen4](https://github.com/mochen4). Mo also generated designs using Meep and were validated by Rasmus.

A schematic of the problem setup and numerical results are shown below. The design region is 10μm by 1μm, with a material of index n=2.4, and lies on a substrate of the same material. The lens tries to focus incident light at 2.4μm away for wavelengths of 450nm, 550nm, and 650nm.

<p align="center">
<img width="1113" alt="rgb_ex" src="https://github.com/mochen4/photonics-opt-testbed/assets/25192039/1ce726f5-ec61-4b17-b3fc-4f96a6d5affc">
<img width="550" alt="Screen Shot 2023-05-29 at 21 13 47" src="https://github.com/mochen4/photonics-opt-testbed/assets/25192039/7b3a2989-d126-46ff-9b92-e5a2ff08a187">
<img width="546" alt="Screen Shot 2023-05-29 at 21 13 57" src="https://github.com/mochen4/photonics-opt-testbed/assets/25192039/9790cd46-76ce-47c8-9957-caf1f6108120">
</p>


For Ez polarization and designing lens without substrate, Wenjin Xue (wenjin.xue@yale.edu) generated a design using her BEM code without enforcing lengthscale constraints, and reported the FOM, which were validated on Meep by Mo. Mo also generated designs using Meep and enforced constraints, and were validated by Wenjin.

A schematic of the problem setup and numerical results are shown below. The design region is 10μm by 1μm, with a material of index n=2.4, and lies on a substrate of the same material. The lens tries to focus incident light at 2.4μm away for wavelengths of 450nm, 550nm, and 650nm.

<p align="center">
<img width="1125" alt="rgb_ez" src="https://github.com/mochen4/photonics-opt-testbed/assets/25192039/fd2dbff4-3c70-4efa-a3ae-3bc679dcd43a">
<img width="552" alt="Screen Shot 2023-05-29 at 21 14 11" src="https://github.com/mochen4/photonics-opt-testbed/assets/25192039/984a9cb2-64e4-40b5-8562-f3be2c5f345f">
<img width="560" alt="Screen Shot 2023-05-29 at 21 14 17" src="https://github.com/mochen4/photonics-opt-testbed/assets/25192039/50237dbe-f375-4d06-a665-ba70df8c2f5c">
</p>
