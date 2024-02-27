import sys
import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
import nlopt
from matplotlib import pyplot as plt


geom_b = 0.005
all_scale = [0.01, 0.02,0.03,0.04,0.05, 0.06, 0.07,0.08,0.09, 0.1]
my_task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])
wavelength = 1.55
medianum=1+int(my_task_id)#
minimum_length = all_scale[my_task_id]
print(medianum)
print(minimum_length)
savepath = 'ldosnew/media'+str(medianum)
mp.verbosity(0)
D_diag =0.001*2*np.pi/wavelength
Si = mp.Medium(index=3.48), D_conductivity_diag= mp.Vector3(D_diag, D_diag, D_diag))
Air = mp.Medium(epsilon=1), D_conductivity_diag= mp.Vector3(D_diag, D_diag, D_diag))
design_region_resolution = 200
eta_i = 0.5 # blueprint (or intermediate) design field thresholding point (between 0 and 1)                                                           
eta_e = 0.65 # erosion design field thresholding point (between 0 and 1)
eta_d = 1-eta_e # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length,eta_e)
pml_size = 1.0
pml_layers = [mp.PML(pml_size)]
pad=0.225
pad2 = 2*pad
resolution = 200
design_region_width0, design_region_height0 = wavelength, wavelength
design_region_width, design_region_height = wavelength+pad2, wavelength+pad2
Nx, Ny = round(design_region_resolution*design_region_width), round(design_region_resolution*design_region_height)


mask = pad
masking = round(pad*design_region_resolution)
x_g = np.linspace(-design_region_width / 2, design_region_width / 2, Nx)
y_g = np.linspace(-design_region_height / 2, design_region_height / 2, Ny)
X_g, Y_g = np.meshgrid(x_g, y_g, sparse=True, indexing="ij")

left_mask = (X_g <= -(design_region_width/2-mask)) & (np.abs(Y_g) <= design_region_height / 2)
right_mask = (X_g >= design_region_width/2-mask) & (np.abs(Y_g) <= design_region_height / 2)

top_mask = (Y_g >= design_region_height/2-mask) & (np.abs(X_g) <= design_region_width / 2)
bottom_mask = (Y_g <= -(design_region_height/2-mask)) & (np.abs(X_g) <= design_region_width / 2)

Air_mask = left_mask | top_mask | right_mask | bottom_mask



Sx = 2*pml_size + design_region_width
Sy = 2*pml_size + design_region_height
cell_size = mp.Vector3(Sx,Sy)
nf = 1
frequencies = np.array([1/wavelength])

fcen = 1/wavelength
width = 0.2
fwidth = width * fcen
source_center  = [0,0,0]
source_size    = mp.Vector3(0,0,0)
src = mp.GaussianSource(frequency=fcen,fwidth=fwidth)
source = [mp.Source(src, component = mp.Ex, size = source_size, center=source_center)]

design_variables = mp.MaterialGrid(mp.Vector3(Nx,Ny),Air,Si,do_averaging=False,beta=0, damping=0.5*6.28/wavelength) #0.1, 0.2, 0.5
design_region = mpa.DesignRegion(design_variables,volume=mp.Volume(center=mp.Vector3(0,0), size=mp.Vector3(design_region_width, design_region_height, 0)))

def mapping(x,eta,beta):
    x = npa.reshape(x,(Nx, Ny))
    x = npa.where(Air_mask, 0, x)
    # filter
    filtered_field = mpa.conic_filter(x,filter_radius,design_region_width,design_region_height, design_region_resolution)

    return filtered_field.flatten()

geometry = [mp.Block(center=design_region.center, size=design_region.size, material=design_variables)]
sim = mp.Simulation(cell_size=cell_size,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=source,
                    default_material=Air,
                    resolution=resolution)

ldos = mpa.LDOS(sim)
ob_list = [ldos]
def J(l):
    return 1/(npa.abs(l[0]))
opt = mpa.OptimizationProblem(
    simulation = sim,
    objective_functions = [J],
    objective_arguments = ob_list,
    design_regions = [design_region],
    frequencies=frequencies,
    maximum_run_time = 500
)

geom_vhistory = []
geom_history = []
evaluation_history = []
cur_iter = [0]
def f(x, grad, eta_i, beta):
    print("Current iteration: {}; current beta: {}".format(cur_iter[0],beta))
    f0, dJ_du = opt([mapping(x,eta_i, beta)])
    grad[:] = tensor_jacobian_product(mapping,0)(x,eta_i,beta,dJ_du)
    evaluation_history.append(np.real(f0))

    print("cur f0", f0)
    cur_iter[0] += 1

    if cur_iter[0] > 145 and mp.am_really_master():
        plt.figure()
        opt.update_design([mapping(x,eta_i, cur_beta)])
        opt.plot2D(True)
        plt.savefig(savepath+'/struct'+str(cur_iter[0])+'.png')
        plt.close()
        np.save(savepath+"/lens"+str(cur_iter[0])+".npy",x)

    return f0

def geom(x,gradient,eta_i, cur_beta):
    g_beta = min(64, cur_beta)

    x = npa.where(Air_mask.flatten(), 0 ,x)

    
    threshf = lambda v: mpa.tanh_projection(v,g_beta,eta_i)
    filterf = lambda v: mpa.conic_filter(v.reshape(Nx, Ny),filter_radius,design_region_width,design_region_height,design_region_resolution)

    geom_c = (filter_radius/design_region_resolution)**2
    g_s = mpa.constraint_solid(x,geom_c,eta_e,filterf,threshf,1) # constraint
    g_s_grad = grad(mpa.constraint_solid,0)(x,geom_c,eta_e,filterf,threshf,1) # gradient
    my_grad =  np.asarray(g_s_grad)
    my_grad = npa.where(Air_mask.flatten(), 0, my_grad)
    gradient[:] = my_grad
    geom_history.append(g_s)
    print("geom",g_s)
    return float(g_s)-geom_b

def geom_v(x,gradient,eta_i, cur_beta):
    g_beta = min(64, cur_beta)

    x = npa.where(Air_mask.flatten(), 0 ,x)


    threshf = lambda v: mpa.tanh_projection(v,g_beta,eta_i)
    filterf = lambda v: mpa.conic_filter(v.reshape(Nx, Ny),filter_radius,design_region_width,design_region_height,design_region_resolution)

    geom_c = (filter_radius/design_region_resolution)**2
    g_v = mpa.constraint_void(x,geom_c,eta_d,filterf,threshf,1) # constraint                                                                                             
    g_v_grad = grad(mpa.constraint_void,0)(x,geom_c,eta_d,filterf,threshf,1) # gradient                                                                                  
    my_grad =  np.asarray(g_v_grad)
    my_grad = npa.where(Air_mask.flatten(), 0, my_grad)
    gradient[:] = my_grad
    geom_vhistory.append(g_v)
    print("geom_v",g_v)
    return float(g_v)-geom_b



algorithm = nlopt.LD_MMA
n = Nx * Ny # number of parameters

x = np.ones((n,)) * 0.5

# lower and upper bounds
lb = np.zeros((Nx*Ny,))
ub = np.ones((Nx*Ny,))

cur_beta = 24
beta_scale = 1.2
num_betas = 15
update_factor = 10
for iters in range(num_betas):
    design_variables.beta = cur_beta
    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(lb)
    solver.set_upper_bounds(ub)
    solver.set_min_objective(lambda x, g: f(x,g,eta_i,cur_beta))
    solver.set_maxeval(update_factor)
    if cur_beta >= 128:#80:
        opt.design_regions[0].design_parameters.do_averaging = True
    if my_task_id % 2 == 1 and cur_beta > 3500:
        solver.add_inequality_constraint(lambda x,g: geom(x,g,eta_i,cur_beta), 1e-4)
    solver.set_param("dual_ftol_rel",1e-8)
    x[:] = solver.optimize(x)
    if mp.am_really_master():
        plt.figure()
        opt.update_design([mapping(x,eta_i, cur_beta)])
        opt.plot2D(True)
        plt.savefig(savepath+'/struct'+str(cur_beta)+'.png')
        plt.close()

    cur_beta = cur_beta*beta_scale
    
update_factor=25
design_variables.beta = np.inf
loss_scales = [0]
for lsidx in range(len(loss_scales)):
    loss_scale = loss_scales[lsidx]
    np.save(savepath+"/lens"+str(lsidx)+".npy",x)
    
    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(lb)
    solver.set_upper_bounds(ub)
    solver.set_min_objective(lambda x, g: f(x,g,eta_i,cur_beta))
    solver.set_maxeval(update_factor)
    solver.add_inequality_constraint(lambda x,g: geom(x,g,eta_i, cur_beta), 1e-4)
    solver.add_inequality_constraint(lambda x,g: geom_v(x,g,eta_i, cur_beta), 1e-4)
    solver.set_param("dual_ftol_rel",1e-8)
    x[:] = solver.optimize(x)
    if mp.am_really_master():
        plt.figure()
        opt.update_design([mapping(x,eta_i, cur_beta)])
        opt.plot2D(True)
        plt.savefig(savepath+'/struct_inf'+str(loss_scale)+'.png')
        plt.close()

savev = x[:]

num_iters = lb.size
#print(max(lb),max(ub),max(mean))                                                                                                                                     
plt.figure()
plt.plot(evaluation_history,'o-')
plt.yscale('log')
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('1/FOM ')
plt.savefig(savepath+'/FOM.png')
plt.figure()
plt.plot(geom_history)
plt.savefig(savepath+'/geom.png')
plt.figure()
plt.plot(geom_vhistory)
plt.savefig(savepath+'/geom_v.png')
                                                                                                                                          
eps_binary=np.round(np.sign(mapping(x,eta_i,cur_beta)-0.5)/2+0.5)
if mp.am_really_master():
    eps_bi_copy = eps_binary.copy()
    eps_bi_copy = eps_bi_copy.reshape(-1,Ny)
    plt.figure()
    plt.imshow(eps_bi_copy,interpolation='spline36',
        cmap='binary',
        alpha=1.0)

    plt.savefig(savepath+'/binaryx.png')
    plt.close()

binary, _ = opt([eps_binary], need_gradient=False)
print("binary",binary)

opt.update_design([eps_binary])
np.save(savepath+"/binary_lens.npy",eps_binary)

if mp.am_really_master():
    plt.figure()
    opt.update_design([eps_binary])
    opt.plot2D(True)
    plt.savefig(savepath+'/struct.png')
    plt.close()
