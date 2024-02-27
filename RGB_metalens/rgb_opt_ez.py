import sys
import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
import nlopt
from matplotlib import pyplot as plt

all_scale = [0.08, 0.12,0.16, 0.2,0.24]
my_task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])

medianum=1+int(my_task_id)
minimum_length = all_scale[my_task_id]
print(medianum, minimum_length)
pad_dict={0.01:0.02, 0.06:0.1, 0.12:0.2, 0.19:0.3,0.25:0.4}
savepath = 'Ez/media'+str(medianum)
mp.verbosity(0)
Si = mp.Medium(index=2.4)
SiO2 = mp.Medium(index=1.0)

design_region_resolution = 50
eta_i = 0.5 # blueprint (or intermediate) design field thresholding point (between 0 and 1)                                                           
eta_e = 0.75 # erosion design field thresholding point (between 0 and 1)
eta_d = 1-eta_e # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length,eta_e)#*design_region_resolution



resolution = 100
pad = 0.5
pad2 = 2*pad
design_region_width, design_region_height = 10+pad2,1+pad2
design_region_width0, design_region_height0 = 10,1
Nx, Ny =round(design_region_resolution*design_region_width),round(design_region_resolution*design_region_height)
pml_size = 1.0

Sx = 2*pml_size + design_region_width0 + 4
Sy = 2*pml_size + design_region_height0 + 4
cell_size = mp.Vector3(Sx,Sy)
nf = 3
frequencies = np.array([1/0.45 ,1/0.55, 1/0.65])

pml_layers = [mp.PML(pml_size)]

fcen = 1/0.55
width = 0.2
fwidth = width * fcen
source_center  = [0,-2.5,0]
source_size    = mp.Vector3(Sx,0,0)
src1 = mp.GaussianSource(frequency=1/0.45,fwidth=fwidth,is_integrated=True)
src2 = mp.GaussianSource(frequency=fcen,fwidth=fwidth,is_integrated=True)
src3 = mp.GaussianSource(frequency=1/0.65,fwidth=fwidth,is_integrated=True)

source = [mp.Source(src1,
                    component = mp.Ez,
                    size = source_size,
                    center=source_center),mp.Source(src2,
                    component = mp.Ez,
                    size = source_size,
                    center=source_center),mp.Source(src3,
                    component = mp.Ez,
                    size = source_size,
                    center=source_center)]


design_variables = mp.MaterialGrid(mp.Vector3(Nx,Ny),SiO2,Si, beta=0, damping=0.5*6.28/0.55, do_averaging=False)
design_region = mpa.DesignRegion(design_variables,volume=mp.Volume(center=mp.Vector3(0,-1), size=mp.Vector3(design_region_width, design_region_height, 0)))

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

def mapping(x,eta,beta):
    xc = npa.where(Air_mask.flatten(),0,x)
    xc = npa.reshape(xc, (Nx, Ny))

    # filter
    filtered_field = mpa.conic_filter(xc,filter_radius,design_region_width,design_region_height, design_region_resolution)

    # interpolate to actual materials
    return filtered_field.flatten()


geometry = [
mp.Block(center=design_region.center, size=design_region.size, material=design_variables)]
sim = mp.Simulation(cell_size=cell_size,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=source,
                    default_material=SiO2,
                    resolution=resolution)

far_x = [mp.Vector3(0,1.92)]
FourierEz = mpa.FourierFields(sim, mp.Volume(center=mp.Vector3(0,1.92),size=mp.Vector3(0.1,0,0)),mp.Ez)
ob_list = [FourierEz]


def J6(Ez):
    return -npa.abs(Ez[:,2])**2

opt = mpa.OptimizationProblem(
    simulation = sim,
    objective_functions = [J6],
    objective_arguments = ob_list,
    design_regions = [design_region],
    frequencies=frequencies,
    maximum_run_time = 200
)
opt.plot2D(True)
plt.savefig(savepath+'/struct.png')
geom_history = []
geom_v_history = []
evaluation_history = []
cur_iter = [0]
def f(x, grad):
    t = x[0] # "dummy" parameter
    v = x[1:] # design parameters
    if grad.size > 0:
        grad[0] = 1
        grad[1:] = 0
    print("cur t",t)
    return t

def c(result,x,gradient,eta,beta):
    print("Current iteration: {}; current eta: {}, current beta: {}".format(cur_iter[0],eta,beta))

    t = x[0] # dummy parameter
    v = x[1:] # design parameters

    f0, dJ_du = opt([mapping(v,eta, beta)])

    # Backprop the gradients through our mapping function
    my_grad = np.zeros(dJ_du.shape)
    
    for k in range(opt.nf):
        my_grad[:,k] = tensor_jacobian_product(mapping,0)(v,eta,beta,dJ_du[:,k])#np.asarray(mapping_vjp(dJ_du[:,k])[0])

    # Assign gradients
    if gradient.size > 0:
        gradient[:,0] = -1 # gradient w.r.t. "t"
        gradient[:,1:] = my_grad.T # gradient w.r.t. each frequency objective

    result[:] = np.real(f0) - t
    print("cur f0,",f0)

    # store results
    evaluation_history.append(np.real(f0))

    cur_iter[0] = cur_iter[0] + 1

def geom(x,gradient,eta,beta):
    g_b = beta
    threshf = lambda v: mpa.tanh_projection(v,g_b,eta)
    filterf = lambda v: mpa.conic_filter(v.reshape(Nx, Ny),filter_radius,design_region_width+pad/design_region_resolution,design_region_height+pad/design_region_resolution,design_region_resolution)
    t = x[0]
    v = x[1:]
    
    v = npa.where(bottom_mask.flatten(),1,npa.where(Air_mask.flatten(),0,v))

    s_c=(filter_radius/resolution)#**4
    g_s = mpa.constraint_solid(v,s_c,eta_e,filterf,threshf,1) # constraint
    g_s_grad = grad(mpa.constraint_solid,0)(v,s_c,eta_e,filterf,threshf,1)# gradient
    gradient[0]=0
    my_grad = g_s_grad.reshape(Nx,Ny)
    my_grad[-masking:, :] = 0
    my_grad[:masking, :] = 0
    my_grad[:, -masking:] = 0
    my_grad[:, :masking] = 0
    gradient[1:] = my_grad.flatten()
    geom_history.append(g_s)
    return float(g_s)-1e-3

def geom_v(x,gradient,eta,beta):
    
    g_b = beta
    threshf = lambda v: mpa.tanh_projection(v,g_b,eta)
    filterf = lambda v: mpa.conic_filter(v.reshape(Nx, Ny),filter_radius,design_region_width+pad/design_region_resolution,design_region_height+pad/design_region_resolution,design_region_resolution)
    t=x[0]
    v = x[1:]
    
    v = npa.where(bottom_mask.flatten(),1,npa.where(Air_mask.flatten(),0,v))
    
    v_c=(filter_radius/resolution)
    g_v = mpa.constraint_void(v,v_c,eta_d,filterf,threshf,1) # constraint                                                      
    g_v_grad = grad(mpa.constraint_void,0)(v,v_c,eta_d,filterf,threshf,1) # gradient                                           
    gradient[0]=0
    my_grad = g_v_grad.reshape(Nx,Ny)
    my_grad[-masking:, :] = 0
    my_grad[:masking, :] = 0
    my_grad[:, -masking:] = 0
    my_grad[:, :masking] = 0
    gradient[1:] = my_grad.flatten()
    geom_v_history.append(g_v)
    return float(g_v)-1e-3

#opt.update_design([npa.where(Si_mask.flatten(),1,npa.where(SiO2_mask.flatten(),0,np.zeros(Nx*Ny)))])
opt.update_design([np.zeros(Nx*Ny)])
f0, _ = opt()
print(f0)

algorithm = nlopt.LD_CCSAQ
n = Nx * Ny # number of parameters

x = np.ones((n,))* 0.5

# lower and upper bounds
lb = np.zeros((Nx*Ny,))
#lb[Si_mask.flatten()] = 1
ub = np.ones((Nx*Ny,))
#ub[SiO2_mask.flatten()] = 0

# insert dummy parameter bounds and variable
x = np.insert(x,0,-0.13) # our initial guess for the worst error
lb = np.insert(lb,0,-np.inf)
ub = np.insert(ub,0,0)

cur_beta = 8
beta_scale = 2
num_betas = 6
update_factor = 50
for iters in range(num_betas):
    design_variables.beta = cur_beta
    solver = nlopt.opt(algorithm, n+1)
    solver.set_lower_bounds(lb)
    solver.set_upper_bounds(ub)
    solver.set_min_objective(f)
    solver.set_maxeval(update_factor)
    solver.add_inequality_mconstraint(lambda r,x,g: c(r,x,g,eta_i,cur_beta), np.array([1e-3]*nf))
    if cur_beta >= 256:
        opt.design_regions[0].design_parameters.do_averaging = True
        solver.add_inequality_constraint(lambda x,g: geom_v(x,g,eta_i,cur_beta), 1e-6)
        solver.add_inequality_constraint(lambda x,g: geom(x,g,eta_i,cur_beta), 1e-6)
    solver.set_param("dual_ftol_rel",1e-8)

    x[:] = solver.optimize(x)
    opt.plot2D(True)
    plt.savefig(savepath+'/struct'+str(cur_beta)+'.png')
    plt.close()
    cur_beta = cur_beta*beta_scale


if mp.am_really_master():
    opt.plot2D(True)
    plt.savefig(savepath+'/end.png')
    plt.close()

savev = x[1:]
if mp.am_really_master():
    np.save(savepath+'/lens.npy', savev)
    print(savev)
    savevcopy = np.load(savepath+"/lens.npy")
    projx = mpa.tanh_projection(mapping(savev,eta_i,np.inf),np.inf,eta_i)
    projx=projx.reshape(-1,Ny)
    projx = np.round(projx)
    np.save(savepath+'/projx.npy',projx)
    plt.imshow(projx,interpolation='spline36',
    cmap='binary',
        alpha=1.0)

    plt.savefig(savepath+'/projx.png')
    plt.close()

lb = -np.min(evaluation_history,axis=1)
ub = -np.max(evaluation_history,axis=1)
mean = -np.mean(evaluation_history,axis=1)

num_iters = lb.size
#print(max(lb),max(ub),max(mean))
plt.figure()
plt.fill_between(np.arange(num_iters),ub,lb,alpha=0.3)
plt.plot(mean,'o-')
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('FOM')
plt.savefig(savepath+'/FOM.png')
plt.figure()
plt.plot(geom_v_history)
plt.savefig(savepath+'/geom_v.png')
plt.figure()
plt.plot(geom_history)
plt.savefig(savepath+'/geom.png')
design_variables.beta=0
eps_binary=np.round(mapping(savev,eta_i,cur_beta))
if mp.am_really_master():
    eps_bi_copy = eps_binary.copy()
    eps_bi_copy = eps_bi_copy.reshape(-1,Ny)
    plt.imshow(eps_bi_copy,interpolation='spline36',
        cmap='binary',
        alpha=1.0)

    plt.savefig(savepath+'/binaryx.png')
    plt.close()
    
binary, _ = opt([eps_binary], need_gradient=False)
print("binary",binary)


opt.update_design([eps_binary])
np.save(savepath+"/binary_lens.npy",eps_binary)

sim4 = mp.Simulation(cell_size=mp.Vector3(Sx,Sy),
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=source,
                    default_material=SiO2,
                    resolution=resolution)
src = mp.ContinuousSource(frequency=1/0.45,fwidth=fwidth)
source = [mp.Source(src,
                    component = mp.Ez,
                    size = source_size,
                    center=source_center)]
sim4.change_sources(source)

sim4.run(until=400)
plt.figure(figsize=(20,10))
sim4.plot2D(fields=mp.Ez)
plt.savefig(savepath+'/sim4.png')

sim5 = mp.Simulation(cell_size=mp.Vector3(Sx,Sy),
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=source,
                    default_material=SiO2,
                    resolution=resolution)
src = mp.ContinuousSource(frequency=1/0.55,fwidth=fwidth)
source = [mp.Source(src,
                    component = mp.Ez,
                    size = source_size,
                    center=source_center)]
sim5.change_sources(source)

sim5.run(until=400)
plt.figure(figsize=(20,10))
sim5.plot2D(fields=mp.Ez)
plt.savefig(savepath+'/sim5.png')

sim6 = mp.Simulation(cell_size=mp.Vector3(Sx,Sy),
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=source,
                    default_material=SiO2,
                    resolution=resolution)
src = mp.ContinuousSource(frequency=1/0.65,fwidth=fwidth)
source = [mp.Source(src,
                    component = mp.Ez,
                    size = source_size,
                    center=source_center)]
sim6.change_sources(source)

sim6.run(until=400)
plt.figure(figsize=(20,10))
sim6.plot2D(fields=mp.Ez)
plt.savefig(savepath+'/sim6.png')
