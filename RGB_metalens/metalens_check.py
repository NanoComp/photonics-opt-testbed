import sys
import ruler
import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filepath", type=str, help='path to lens file')
parser.add_argument("--resolution", type=int, default=50, help="simulation resolution")
args = parser.parse_args()
filepath, resolution = args.filepath, args.resolution
lens = np.genfromtxt(filepath)
polar=mp.Ex
mp.verbosity(0)
medium = mp.Medium(index=2.4)

Nx, Ny = lens.shape
design_region_width, design_region_height = 10, 1

pml_size = 1.0
pml_layers = [mp.PML(pml_size)]

Sx = 2*pml_size + design_region_width + 4
Sy = 2*pml_size + design_region_height + 4
cell_size = mp.Vector3(Sx,Sy)

nf = 3
frequencies = np.array([1/0.45 ,1/0.55, 1/0.65])
fcen = 1/0.55
width = 0.2
fwidth = width * fcen
source_center  = [0,-2.5,0]
source_size    = mp.Vector3(Sx,0,0)
src1 = mp.GaussianSource(frequency=1/0.45,fwidth=fwidth,is_integrated=True)
src2 = mp.GaussianSource(frequency=fcen,fwidth=fwidth,is_integrated=True)
src3 = mp.GaussianSource(frequency=1/0.65,fwidth=fwidth,is_integrated=True)
source = [mp.Source(src1,
                    component = polar,
                    size = source_size,
                    center=source_center),mp.Source(src2,
                    component = polar,
                    size = source_size,
                    center=source_center),mp.Source(src3,
                    component = polar,
                    size = source_size,
                    center=source_center)]

design_variables = mp.MaterialGrid(mp.Vector3(Nx,Ny),mp.air,medium,beta=8,damping=0.0*6.28/0.55)
design_region = mpa.DesignRegion(design_variables,volume=mp.Volume(center=mp.Vector3(0,-1), size=mp.Vector3(design_region_width, design_region_height, 0)))
geometry = [
    mp.Block(center=mp.Vector3(0, -2.5), size=mp.Vector3(Sx, 2), material=medium),
    mp.Block(center=design_region.center, size=design_region.size, material=design_variables)]
kpoint = mp.Vector3()
sim = mp.Simulation(cell_size=cell_size,
                    boundary_layers=pml_layers,
                    k_point=kpoint,
                    geometry=geometry,
                    sources=source,
                    resolution=resolution)

FourierEx = mpa.FourierFields(sim, mp.Volume(center=mp.Vector3(0,1.92),size=mp.Vector3(0.1,0,0)),mp.Ex)
ob_list = [FourierEx]
def J(Ex):
    return -npa.abs(Ex[:,2])**2
opt = mpa.OptimizationProblem(
    simulation = sim,
    objective_functions = [J],
    objective_arguments = ob_list,
    design_regions = [design_region],
    frequencies=frequencies
)
#ref, _ = opt([np.zeros(Nx*Ny)], need_gradient=False)
#print(ref) # should be [-0.16528992 -0.16579352 -0.16187553]
ref = np.array([-0.16528992, -0.16579352, -0.16187553])
fom, _ = opt([lens.flatten()], need_gradient=False)
print("FOM = ",np.mean(fom/ref))


Lx,Ly = design_region_width, design_region_height # size of the 2d design, correspond to the row and column of the 2d array
obj = ruler.morph([Lx,Ly])
print("minimum_length = ",obj.minimum_length(lens))
    
