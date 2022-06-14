
This is an example of the inverse design of slow light waveguide based on the supercell calculation for the Hz polarization. Fengwen Wang [@fengwen1206](https://github.com/fengwen1206) generated the design using her own  matlab code. 

The considerd supercell, initial design and corresponding design domain are illustrated in the figure below.

![schematic](/slow_light_waveguide/Illustration.png)

The design problem is stated as

![schematic](/slow_light_waveguide/Optimizationformulation.PNG)

 The relevant parameters are defined below:
 - **Discretization**: 408x40 bilinear quadrilateral elements
 - **Regularization**: density filter (filter radius: 1/8a) + projection
 - **Continuation scheme in the projection**: 	 For every 40th iteration or if  ({ $\Delta \rho < 1e-3$  || $\Delta f < 1e-3 $ } &&  $\beta < 50$),   set $\beta=1.3 \beta$.   
  If $\Delta \rho < 1e-4$ || $\Delta f < 1e-4 $,  terminate. 
 
- **Interpolation of the relative permittivity of element e **:  

   $\frac{1}{\varepsilon^{\eta}_e}$
   
   $= \left(1- \overline{\rho}^{\eta}_e \right)\frac{1}{\varepsilon_{\rm{air}}} +\overline{\rho}^{\eta}_e  \frac{1}{\varepsilon_{\rm{Si}}}$
   - 
