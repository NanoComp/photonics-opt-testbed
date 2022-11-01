"""Topology optimization of the waveguide mode converter using
   Meep's adjoint solver from A M. Hammond et al., Optics Express,
   Vol. 30, pp. 4467-4491 (2022). doi.org/10.1364/OE.442074

The worst-case optimization is based on minimizing the maximum
of R + (1-T) where R (reflectance) is $|S_{11}|^2$ for mode 1
and T (transmittance) is $|S_{21}|^2$ for mode 2 across six
different wavelengths. The optimization uses the method of moving
asymptotes (MMA) algorithm from NLopt. The minimum linewidth
constraint is based on A.M. Hammond et al., Optics Express,
Vol. 29, pp. 23916-23938, (2021). doi.org/10.1364/OE.431188
"""

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from autograd import numpy as npa, tensor_jacobian_product, grad
import nlopt
import meep as mp
import meep.adjoint as mpa

resolution = 50  # pixels/μm

w = 0.4          # waveguide width
l = 3.0          # waveguide length (on each side of design region)
dpad = 0.6       # padding length above/below design region
dpml = 1.0       # PML thickness
dx = 1.6         # length of design region
dy = 1.6         # width of design region

sx = dpml+l+dx+l+dpml
sy = dpml+dpad+dy+dpad+dpml
cell_size = mp.Vector3(sx,sy,0)

pml_layers = [mp.PML(thickness=dpml)]

# wavelengths for minimax optimization
wvls = (1.265, 1.270, 1.275, 1.285, 1.290, 1.295)
frqs = [1/wvl for wvl in wvls]

minimum_length = 0.05  # minimum length scale (μm)
eta_i = (
    0.5 # blueprint design field thresholding point (between 0 and 1)
)
eta_e = 0.75       # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
print(f"filter_radius:, {filter_radius:.6f}")

# pulsed source center frequency and bandwidth
wvl_min = 1.26
wvl_max = 1.30
frq_min = 1/wvl_max
frq_max = 1/wvl_min
fcen = 0.5*(frq_min+frq_max)
df = frq_max-frq_min

eig_parity = mp.ODD_Z
src_pt = mp.Vector3(-0.5*sx+dpml,0,0)

nSiO2 = 1.5
SiO2 = mp.Medium(index=nSiO2)
nSi = 3.5
Si = mp.Medium(index=nSi)

design_region_size = mp.Vector3(dx,dy,0)
design_region_resolution = int(2*resolution)
Nx = int(design_region_size.x*design_region_resolution)
Ny = int(design_region_size.y*design_region_resolution)

# impose a 1-pixel thick bit "mask" around the edges
# of the design region in order to prevent violations
# of the mininum linewidth constraint.

x_g = np.linspace(
    -design_region_size.x / 2,
    design_region_size.x / 2,
    Nx,
)
y_g = np.linspace(
    -design_region_size.y / 2,
    design_region_size.y / 2,
    Ny,
)
X_g, Y_g = np.meshgrid(
    x_g,
    y_g,
    sparse=True,
    indexing="ij",
)

left_wg_mask = (X_g == -design_region_size.x / 2) & (np.abs(Y_g) <= w / 2)
right_wg_mask = (X_g == design_region_size.x / 2) & (np.abs(Y_g) <= w / 2)
Si_mask = left_wg_mask | right_wg_mask

border_mask = (
    (X_g == -design_region_size.x / 2) |
    (X_g == design_region_size.x / 2) |
    (Y_g == -design_region_size.y / 2) |
    (Y_g == design_region_size.y / 2)
)
SiO2_mask = border_mask.copy()
SiO2_mask[Si_mask] = False

refl_pt = mp.Vector3(-0.5*sx+dpml+0.5*l)
tran_pt = mp.Vector3(0.5*sx-dpml-0.5*l)

stop_cond = mp.stop_when_fields_decayed(50, mp.Ez, refl_pt, 1e-8)

def mapping(x, eta, beta):
    """A differentiable mapping function which applies, in order,
       the following sequence of transformations to the design weights:
       (1) a bit mask for the edge pixels, (2) convolution with a
       conic filter, and (3) projection via a hyperbolic tangent.

    Args:
      x: design weights as a 1d array of size Nx*Ny.
      eta: erosion/dilation parameter for the projection.
      beta: bias parameter for the projection.
    """
    x = npa.where(
        Si_mask.flatten(),
        1,
        npa.where(
            SiO2_mask.flatten(),
            0,
            x
        )
    )

    filtered_field = mpa.conic_filter(
        x,
        filter_radius,
        design_region_size.x,
        design_region_size.y,
        design_region_resolution,
    )

    projected_field = mpa.tanh_projection(
        filtered_field,
        beta,
        eta,
    )

    return projected_field.flatten()


def f(x, grad):
    """Objective function for the epigraph formulation.

    Args:
      x: 1d array of size 1+Nx*Ny containing epigraph variable (first element)
         and design weights (remaining elements).
      grad: the gradient as 1d array of size 1+Nx*Ny.
    """
    t = x[0]  # epigraph variable
    v = x[1:] # design weights
    if grad.size > 0:
        grad[0] = 1
        grad[1:] = 0
    return t


def c(result, x, gradient, eta, beta):
    """Constraint function for the epigraph formulation.

       Args:
         result: the result of the function evaluation modified in place.
         x: 1d array of size 1+Nx*Ny containing epigraph variable (first
            element) and design weights (remaining elements).
         gradient: the Jacobian matrix with dimensions (1+Nx*Ny,
                   num. wavelengths).
         eta: erosion/dilation parameter for projection.
         beta: bias parameter for projection.
    """
    print(f"iteration: {cur_iter[0]}, eta: {eta}, beta: {beta}")

    t = x[0]  # epigraph variable
    v = x[1:] # design weights

    f0, dJ_du = opt([mapping(v, eta, beta)])

    # backpropagate the gradients through mapping function
    my_grad = np.zeros(dJ_du.shape)
    for k in range(opt.nf):
        my_grad[:, k] = tensor_jacobian_product(mapping, 0)(
            v,
            eta,
            beta,
            dJ_du[:, k],
        )

    if gradient.size > 0:
        gradient[:, 0] = -1  # gradient w.r.t. epigraph variable ("t")
        gradient[:, 1:] = my_grad.T  # gradient w.r.t. each frequency objective

    result[:] = np.real(f0) - t

    evaluation_history.append(np.real(f0))

    cur_iter[0] = cur_iter[0] + 1


def glc(result, x, gradient, beta):
    """Constraint function for the minimum linewidth.

       Args:
         result: the result of the function evaluation modified in place.
         x: 1d array of size 1+Nx*Ny containing epigraph variable (first
            element) and design weights (remaining elements).
         gradient: the Jacobian matrix with dimensions (1+Nx*Ny,
                   num. wavelengths).
         beta: bias parameter for projection.
    """
    t = x[0]  # dummy parameter
    v = x[1:] # design parameters
    a1 = 1e-4 # hyper parameter (primary)
    b1 = 0    # hyper parameter (secondary)
    gradient[:,0] = -a1

    filter_f = lambda a: mpa.conic_filter(
        a.reshape(Nx,Ny),
        filter_radius,
        design_region_size.x,
        design_region_size.y,
        design_region_resolution,
    )
    threshold_f = lambda a: mpa.tanh_projection(a,beta,0.5)
    c0 = (filter_radius*1/resolution)**4

    M1 = lambda a: mpa.constraint_solid(a,c0,eta_e,filter_f,threshold_f,1)
    M2 = lambda a: mpa.constraint_void(a,c0,eta_d,filter_f,threshold_f,1)

    g1 = grad(M1)(v)
    g2 = grad(M2)(v)

    result[0] = M1(v) - a1*t - b1
    result[1] = M2(v) - a1*t - b1

    gradient[0,1:] = g1.flatten()
    gradient[1,1:] = g2.flatten()


def straight_waveguide():
    """Computes the DFT fields from the mode source in a straight waveguide
       for use as normalization of the reflectance measurement during the
       optimization.

    Returns:
      1d array of DFT fields and DFT fields object returned by `get_flux_data`.
    """
    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen,fwidth=df),
            size=mp.Vector3(0,sy,0),
            center=src_pt,
            eig_band=1,
            eig_parity=eig_parity,
        )
    ]

    geometry = [
        mp.Block(
            size=mp.Vector3(mp.inf,w,mp.inf),
            center=mp.Vector3(),
            material=Si,
        )
    ]

    sim = mp.Simulation(
        resolution=resolution,
        default_material=SiO2,
        cell_size=cell_size,
        sources=sources,
        geometry=geometry,
        boundary_layers=pml_layers,
        k_point=mp.Vector3(),
    )

    refl_mon = sim.add_mode_monitor(
        frqs,
        mp.ModeRegion(center=refl_pt,
                      size=mp.Vector3(0,sy,0)),
        yee_grid=True,
    )

    sim.run(until_after_sources=stop_cond)

    res = sim.get_eigenmode_coefficients(
        refl_mon,
        [1],
        eig_parity=eig_parity,
    )

    coeffs = res.alpha
    input_flux = np.abs(coeffs[0,:,0])**2
    input_flux_data = sim.get_flux_data(refl_mon)

    return input_flux, input_flux_data


def converter_optimization(input_flux, input_flux_data):
    """Sets up the adjoint optimization of the waveguide mode converter.

    Args:
      input_flux: 1darray of DFT fields from normalization run.
      input_flux_data: DFT fields object returned by `get_flux_data`.

    Returns:
      A `meep.adjoint.OptimizationProblem` class object.
    """
    matgrid = mp.MaterialGrid(
        mp.Vector3(Nx,Ny,0),
        SiO2,
        Si,
        weights=np.ones((Nx,Ny)),
    )

    matgrid_region = mpa.DesignRegion(
        matgrid,
        volume=mp.Volume(
            center=mp.Vector3(),
            size=mp.Vector3(
                design_region_size.x,
                design_region_size.y,
                mp.inf
            ),
        )
    )

    matgrid_geometry = [
        mp.Block(
            center=matgrid_region.center,
            size=matgrid_region.size,
            material=matgrid,
        )
    ]

    geometry = [
        mp.Block(
            size=mp.Vector3(mp.inf,w,mp.inf),
            center=mp.Vector3(),
            material=Si
        )
    ]

    geometry += matgrid_geometry

    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen,fwidth=df),
            size=mp.Vector3(0,sy,0),
            center=src_pt,
            eig_band=1,
            eig_parity=eig_parity),
    ]

    sim = mp.Simulation(
        resolution=resolution,
        default_material=SiO2,
        cell_size=cell_size,
        sources=sources,
        geometry=geometry,
        boundary_layers=pml_layers,
        k_point=mp.Vector3(),
    )

    obj_list = [
        mpa.EigenmodeCoefficient(
            sim,
            mp.Volume(
                center=refl_pt,
                size=mp.Vector3(0,sy,0),
            ),
            1,
            forward=False,
            eig_parity=eig_parity,
            subtracted_dft_fields=input_flux_data,
        ),
        mpa.EigenmodeCoefficient(
            sim,
            mp.Volume(
                center=tran_pt,
                size=mp.Vector3(0,sy,0),
            ),
            2,
            eig_parity=eig_parity,
        )
    ]

    def J(refl_mon,tran_mon):
        return (npa.power(npa.abs(refl_mon), 2) / input_flux) + (
            1 - npa.power(npa.abs(tran_mon), 2) / input_flux)

    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=J,
        objective_arguments=obj_list,
        design_regions=[matgrid_region],
        frequencies=frqs,
    )

    return opt


if __name__ == '__main__':
    input_flux, input_flux_data = straight_waveguide()

    opt = converter_optimization(input_flux, input_flux_data)

    algorithm = nlopt.LD_MMA

    # number of design parameters
    n = Nx * Ny

    # initial guess for design parameters
    x = np.ones((n,)) * 0.5
    x[Si_mask.flatten()] = 1    # set the edges of waveguides to silicon
    x[SiO2_mask.flatten()] = 0  # set the other edges to SiO2

    # lower and upper bounds for design weights
    lb = np.zeros((n,))
    lb[Si_mask.flatten()] = 1
    ub = np.ones((n,))
    ub[SiO2_mask.flatten()] = 0

    # insert epigraph variable and bounds as first element of 1d arrays
    x = np.insert(x, 0, 1.2)   # initial guess for the worst error
    lb = np.insert(lb, 0, 0.)  # lower bound: cannot be less than 0.
    ub = np.insert(ub, 0, 2.)  # upper bound: cannot be more than 2.

    evaluation_history = []
    cur_iter = [0]

    betas = [8, 16, 32]
    max_eval = 80
    tol = np.array([1e-6] * opt.nf)
    for beta in betas:
        solver = nlopt.opt(algorithm, n + 1)
        solver.set_lower_bounds(lb)
        solver.set_upper_bounds(ub)
        solver.set_min_objective(f)
        solver.set_maxeval(max_eval)
        solver.add_inequality_mconstraint(
            lambda r, x, g: c(r, x, g, eta_i, beta),
            tol,
        )
        # apply the minimum linewidth constraint
        # only in the final "epoch"
        if beta == betas[-1]:
            solver.add_inequality_mconstraint(
                lambda r, x, g: glc(r, x, g, beta),
                [1e-8] * opt.nf,
            )
        x[:] = solver.optimize(x)

        optimal_design_weights = mapping(
            x[1:],
            eta_i,
            beta,
        ).reshape(Nx,Ny)

        if mp.am_master():
            # save a bitmap image of the design at the end of each epoch
            plt.figure()
            plt.imshow(
                optimal_design_weights,
                cmap='binary',
                interpolation='none',
            )
            plt.axis('off')
            plt.savefig(
                f'optimal_design_beta{beta}.png',
                dpi=150,
                bbox_inches='tight',
            )

    # save the optimal design as a 2d array in CSV format
    np.savetxt(
        'optimal_design.csv',
        optimal_design_weights,
        fmt='%4.2f',
        delimiter=','
    )

    # save all the important optimization parameters and data
    with open("optimal_design.npz","wb") as fl:
        np.savez(
            fl,
            Nx=Nx,
            Ny=Ny,
            design_region_size=(dx,dy),
            design_region_resolution=design_region_resolution,
            betas=betas,
            max_eval=max_eval,
            evaluation_history=evaluation_history,
            t=x[0],
            unmapped_design_weights=x[1:],
            minimum_length=minimum_length,
            optimal_design_weights=optimal_design_weights,
        )

