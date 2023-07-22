This is an example of the inverse design of slow light waveguide based on the supercell calculation for the Hz polarization. Fengwen Wang [@fengwen1206](https://github.com/fengwen1206) generated the design using her own  matlab code. 

The considerd supercell, initial design and corresponding design domain are illustrated in the figure below.

![schematic](/slow_light_waveguide/Illustration.png)

The design problem is stated as

![schematic](/slow_light_waveguide/Optimizationformulation.png)

 The relevant parameters are defined below:
 - **Discretization**: 408x40 bilinear quadrilateral elements
 - **Regularization**: density filtering (filter radius: $0.125a$) + projection
 - **Continuation scheme in the projection**: 	
 
     For $\beta < 50$, set $\beta=1.3 \beta$ every 40 iterations or if  $\max (\Delta \rho ,\Delta f) < 10^{-3}$ and $\min (\Delta \rho ,\Delta f) \geq 10^{-4}$.   
     If  $\beta \geq 50$ or $\min (\Delta \rho ,\Delta f) < 10^{-4}$,  terminate. 
 
- **Interpolation of the relative permittivity of element e**:

    $\frac{1}{\varepsilon^{\eta}_e}=(1-\hat{\rho}^{\eta}_e)\frac{1}{\varepsilon_A}+\hat{\rho}^{\eta}_e  \frac{1}{\varepsilon_S}, \quad   \varepsilon_A=1 \quad,\varepsilon_S=3.476^2$
  
- **Robust formulation**: $\eta\in [0.35, 0.5, 0.65]$.
- **Target group index**: $n^*_g=25$
- **Target $k$ points**: $ka/(2\pi) = 0.3875, 0.4, 0.4125, 0.425, 0.4375, 0.45, 0.4625$
- **Initial guess**:  $a_1=0.9$ and $a_2=1.1$  

The blue print design  with $\eta=0.5$ obtained using the robust optimization formulation considering the parameters above and corresponding performance are shown in the figure below. [Design_Dnum_2.csv](/slow_light_waveguide/Design_Dnum_2.csv) is the corresponding csv format design pattern. 
 [Opt_Band.csv](/slow_light_waveguide/Opt_Band.csv) is the corresponding band structure in csv format with first column for k and [Opt_Group_index.csv](/slow_light_waveguide/Opt_Group_index.csv) is the corresponding group index in csv format with first column for k. [HigRes_DesMatch_Opt_Dnum_2.csv](/slow_light_waveguide/HigRes_DesMatch_Opt_Dnum_2.csv) is the design pattern in high resolution extracted using the contour form  the optimized design.

![schematic](/slow_light_waveguide/Resp_Dnum_2_FF.png)

The optimization history is shown in the figure below.

![schematic](/slow_light_waveguide/Opt_History_SlowLight.png)