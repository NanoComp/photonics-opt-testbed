# Computes the transmittance of the m=+1 order
# for a 3D metagrating with the 2D design
# imported from the file device1.csv

import meep as mp
import math
import numpy as np


def metagrating(P_pol: bool):
    resolution = 100  # pixels/μm

    nSi = 3.45
    Si = mp.Medium(index=nSi)
    nSiO2 = 1.45
    SiO2 = mp.Medium(index=nSiO2)

    theta_d = math.radians(50.0)  # deflection angle

    wvl = 1.05  # wavelength
    fcen = 1/wvl

    px = wvl/math.sin(theta_d)  # period in x
    py = 0.5*wvl  # period in y

    dpml = 1.0  # PML thickness
    gh = 0.325  # grating height
    dsub = 5.0  # substrate thickness
    dair = 5.0  # air padding

    sz = dpml+dsub+gh+dair+dpml

    cell_size = mp.Vector3(px,py,sz)

    boundary_layers = [mp.PML(thickness=dpml,direction=mp.Z)]

    # periodic boundary conditions
    k_point = mp.Vector3()

    # plane of incidence is XZ
    src_cmpt = mp.Ex if P_pol else mp.Ey
    src_pt = mp.Vector3(0,0,-0.5*sz+dpml+0.5*dsub)
    sources = [mp.Source(src=mp.GaussianSource(fcen,fwidth=0.1*fcen),
                         size=mp.Vector3(px,py,0),
                         center=src_pt,
                         component=src_cmpt)]

    sim = mp.Simulation(resolution=resolution,
                        cell_size=cell_size,
                        sources=sources,
                        default_material=SiO2,
                        boundary_layers=boundary_layers,
                        k_point=k_point)

    flux = sim.add_mode_monitor(fcen,
                                0,
                                1,
                                mp.ModeRegion(center=mp.Vector3(0,0,0.5*sz-dpml),
                                              size=mp.Vector3(px,py,0)))

    stop_cond = mp.stop_when_fields_decayed(10,src_cmpt,src_pt,1e-7)
    sim.run(until_after_sources=stop_cond)

    input_flux = mp.get_fluxes(flux)

    sim.reset_meep()

    # image resolution is ~340 pixels/μm and thus
    # Meep resolution should not be larger than ~half this value
    weights = np.genfromtxt('device1.csv',delimiter=',')
    Nx, Ny = weights.shape

    geometry = [mp.Block(size=mp.Vector3(mp.inf,mp.inf,dpml+dsub),
                         center=mp.Vector3(0,0,-0.5*sz+0.5*(dpml+dsub)),
                         material=SiO2),
                mp.Block(size=mp.Vector3(px,py,gh),
                         center=mp.Vector3(0,0,-0.5*sz+dpml+dsub+0.5*gh),
                         material=mp.MaterialGrid(grid_size=mp.Vector3(Nx,Ny,1),
                                                  medium1=mp.air,
                                                  medium2=Si,
                                                  weights=weights))]

    sim = mp.Simulation(resolution=resolution,
                        cell_size=cell_size,
                        sources=sources,
                        geometry=geometry,
                        boundary_layers=boundary_layers,
                        k_point=k_point,
                        eps_averaging=False)

    flux = sim.add_mode_monitor(fcen,
                                0,
                                1,
                                mp.ModeRegion(center=mp.Vector3(0,0,0.5*sz-dpml),
                                              size=mp.Vector3(px,py,0)))

    sim.run(until_after_sources=stop_cond)

    res = sim.get_eigenmode_coefficients(flux,
                                         mp.DiffractedPlanewave((1,0,0),
                                                                mp.Vector3(1,0,0),
                                                                0 if P_pol else 1,
                                                                1 if P_pol else 0))

    coeffs = res.alpha
    tran = abs(coeffs[0,0,0])**2 / input_flux[0]
    print("tran:, {}, {:.6f}".format('P' if P_pol else 'S',tran))

    # for debugging:
    # visualize three orthogonal cross sections of the 3D cell
    # to ensure that structure matches expected design
    if 0:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        output_plane = mp.Volume(center=mp.Vector3(0,0,0.5*sz-dpml-dair-0.5*gh),
                                 size=mp.Vector3(px,py,0))
        plt.figure()
        sim.plot2D(output_plane=output_plane,
                   eps_parameters={'resolution':100})
        plt.savefig('cell_xy.png',dpi=150,bbox_inches='tight')

        output_plane = mp.Volume(center=mp.Vector3(0,0,0),
                                 size=mp.Vector3(0,py,sz))
        plt.figure()
        sim.plot2D(output_plane=output_plane,
                   eps_parameters={'resolution':100})
        plt.savefig('cell_yz.png',dpi=150,bbox_inches='tight')

        output_plane = mp.Volume(center=mp.Vector3(0,0,0),
                                 size=mp.Vector3(px,0,sz))
        plt.figure()
        sim.plot2D(output_plane=output_plane,
                   eps_parameters={'resolution':100})
        plt.savefig('cell_xz.png',dpi=150,bbox_inches='tight')


if __name__ == '__main__':
    metagrating(False)
    metagrating(True)
