
This is an example of the inverse design of slow light waveguide based on the supercell calculation for the Hz polarization. Fengwen Wang [@fengwen1206](https://github.com/fengwen1206) generated the design using her own  matlab code. 

The considerd supercell, initial design and corresponding design domain are illustrated in the figure below.

![schematic](/slow_light_waveguide/Illustration.png)

The design problem is stated as

![schematic](/slow_light_waveguide/Optimizationformulation.PNG)

 The relevant parameters are defined below:
 - **Discretization**: 408x40 bilinear quadrilateral elements
 - **Regularization**: density filter (filter radius: 1/8a) + projection
 - **Continuation scheme in the projection**: 	
 
     For every 40th iteration or if  ( ($\Delta \rho < 1e-3$  || $\Delta f < 1e-3 $ )& $\beta $ < 50),   set $\beta=1.3 \beta$.   
     If  $\Delta \rho < 1e-4 $ ||  $\Delta f$ < 1e-4,  terminate. 
 
- **Interpolation of the relative permittivity of element e**:

    $\frac{1}{\varepsilon^{\eta}_e}=(1-\bar{\rho}^{\eta}_e)\frac{1}{\varepsilon_A}+\overline{\rho}^{\eta}_e  \frac{1}{\varepsilon_S}, \quad   \varepsilon_A=1 \quad,\varepsilon_S=3.476^2$
  
- **Robust formulation**: $ \eta \in [0.35, 0.5, 0.65]$.
- **Target group index**: $n^*_g=25$
- **Target $k$ points**: Target $k$ points
- **Initial guess**:  $a_1=0.9$ and $a_2=1.1$  

The blue print design  with $\eta=0.5$ obtained using the robust optimization formulation considering the parameters above and corresponding performance are shown in the figure below. [Design_Dnum_2.csv](/slow_light_waveguide/Design_Dnum_2.csv) is the corresponding csv format design pattern. [Opt_Band.csv (/slow_light_waveguide/Opt_Band.csv) is the corresponding band structure in csv format with first column for k and [Opt_Group_index.csv](/slow_light_waveguide/Opt_Group_index.csv) is the corresponding group index in csv format with first column for k. [HigRes_DesMatch_Opt_Dnum_2.csv](/slow_light_waveguide/HigRes_DesMatch_Opt_Dnum_2.csv) is the design pattern in high resolution extracted using the contour form  the optimized design. 


![schematic](/slow_light_waveguide/Resp_Dnum_2_FF.png)

The optimization history is shown in the figure below.

![schematic](/slow_light_waveguide/Opt_History_SlowLight.png)
