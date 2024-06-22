# This python script is for maximizing the diffraction efficiency of the m=+1 order
# for a 1D metagrating.

import meep as mp
import meep.adjoint as mpa
import numpy as np
import autograd.numpy as npa
from autograd import tensor_jacobian_product
import nlopt

Si = mp.Medium(index=3.45)
SiO2 = mp.Medium(index=1.45)

resolution = 86  # pixels/μm
design_region_resolution = (resolution, 0)

theta_d = np.radians(50.0)  # deflection angle
wvl = 1.05  # wavelength
fcen = 1/wvl
px = wvl/np.sin(theta_d)  # period in x

dpml = 1.0  # PML thickness
gh = 0.325  # grating height
dsub = 5.0  # substrate thickness
dair = 5.0  # air padding
sy = dpml+dsub+gh+dair+dpml
cell_size = mp.Vector3(px,sy)
boundary_layers = [mp.PML(thickness=dpml,direction=mp.Y)]

# periodic boundary conditions
k_point = mp.Vector3()

# plane of incidence is XY
P_pol = True
src_cmpt = mp.Ex if P_pol else mp.Ez
src_pt = mp.Vector3(0,-0.5*sy+dpml+0.5*dsub)
sources = [mp.Source(src=mp.GaussianSource(fcen,fwidth=0.1*fcen),
                     size=mp.Vector3(px,0),
                     center=src_pt,
                     component=src_cmpt)]

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    sources=sources,
                    default_material=SiO2,
                    boundary_layers=boundary_layers,
                    k_point=k_point)

flux = sim.add_mode_monitor(fcen,0,1,
                            mp.ModeRegion(center=mp.Vector3(0,0.5*sy-dpml),
                                          size=mp.Vector3(px,0)))

stop_cond = mp.stop_when_fields_decayed(10,src_cmpt,src_pt,1e-7)
sim.run(until_after_sources=stop_cond)

input_flux = mp.get_fluxes(flux)
sim.reset_meep()

eta_i = 0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
filter_radius = 0.06

diffraction_order = 1
kdiff = mp.Vector3(diffraction_order/px,np.sqrt(fcen**2-(diffraction_order/px)**2))

def J(emc):
    return npa.abs(emc[0])**2 / input_flux[0]

def mapping(x):
    filtered_field = mpa.conic_filter(x,filter_radius,px,gh,design_region_resolution, periodic_axes=0)
    filtered_field_symm = (filtered_field+npa.fliplr(filtered_field))/2
    return filtered_field_symm.flatten()

evaluation_history, iteration = [], [0]
Nx, Ny = int(round(design_region_resolution[0]*px))+1, int(round(design_region_resolution[1]*gh))+1

design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), mp.air, Si, grid_type="U_MEAN")
design_region = mpa.DesignRegion(design_variables,
                                 volume=mp.Volume(center=mp.Vector3(0,-0.5*sy+dpml+dsub+0.5*gh),
                                                  size=mp.Vector3(px,gh)))

geometry = [mp.Block(center=mp.Vector3(0,-0.5*sy+0.5*(dpml+dsub)),
                     size=mp.Vector3(px,dpml+dsub),
                     material=SiO2),
            mp.Block(center=design_region.center, 
                     size=design_region.size, 
                     material=design_variables)]

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    sources=sources,
                    geometry=geometry,
                    boundary_layers=boundary_layers,
                    k_point=k_point,
                    eps_averaging=False)

emc = mpa.EigenmodeCoefficient(
    sim,
    mp.Volume(
        center=mp.Vector3(0, 0.5*sy - dpml),
        size=mp.Vector3(px, 0)
    ),
    mode=1,
    eig_parity=mp.EVEN_Z if P_pol else mp.ODD_Z,
    kpoint_func=lambda *not_used: kdiff,
    eig_vol=mp.Volume(
        center=mp.Vector3(0, 0.5*sy - dpml),
        size=mp.Vector3(1 / resolution, 0, 0)
    )
)
ob_list = [emc]

opt = mpa.OptimizationProblem(simulation=sim,objective_functions=J,objective_arguments=ob_list,
                              design_regions=[design_region],fcen=fcen,df=0,nf=1,decay_by=1e-6)

def f(v, gradient):
    print("Current iteration:",iteration[0])
    iteration[0] = iteration[0]+1
    v_mapped = mapping(v)
    eff,dJ_dn = opt([v_mapped])
    if gradient.size > 0:
        gradient[:] = tensor_jacobian_product(mapping, 0)(v, dJ_dn)
    evaluation_history.append(eff)
    np.savetxt('objective_1d.dat',evaluation_history) # evaluation history of the objective, i.e., the diffraction efficiency
    np.savetxt('pattern_1d.dat',np.vstack((v,v_mapped))) # unfiltered and filtered design patterns at the current iteration
    return eff

n = Nx*Ny
lb, ub = np.zeros(n), np.ones(n) # lower and upper bounds
x = 0.5*np.ones(n) # initial guess

betas = [4, 8, 16, 32, 64, 96, 128, 256, float("inf")]
max_evals = [80, 80, 80, 80, 80, 80, 80, 80, 200]

for beta, max_eval in zip(betas,max_evals):
    print("Current beta:",beta)
    design_variables.beta = beta
    if beta <= 32:
        design_variables.do_averaging = False
        design_variables.damping = 0.05*2*np.pi*fcen
    else:
        design_variables.do_averaging = True
        design_variables.damping = 0

    solver = nlopt.opt(nlopt.LD_MMA, n)
    solver.set_lower_bounds(lb)
    solver.set_upper_bounds(ub)
    solver.set_max_objective(f)
    solver.set_maxeval(max_eval)
    solver.set_xtol_rel(1e-8)
    x[:] = solver.optimize(x)

x = np.sign(mapping(x)-0.5)/2+0.5
x = x.astype(int)
np.savetxt('optimized_pattern_1d.dat', x)
