# compute the transmittance of the m=+1 order
# for a 3d metagrating with the 2d design
# imported from the file device1.csv
# and compare the result with the expected value
# from RCWA (Reticolo)

import meep as mp
import math
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def metagrating(P_pol: bool):
    resolution = 50  # pixels/μm

    nSi = 3.45
    Si = mp.Medium(index=nSi)
    nSiO2 = 1.45
    SiO2 = mp.Medium(index=nSiO2)

    theta_d = math.radians(50.0)  # deflection angle

    wvl = 1.05  # wavelength
    fcen = 1/wvl

    px = wvl/math.sin(theta_d)  # period in x
    py = 0.5*wvl  # period in y

    dpml = wvl  # PML thickness
    gh = 0.325  # grating height
    dsub = 3.0  # substrate thickness
    dair = 3.0  # air padding

    sz = dpml+dsub+gh+dair+dpml

    cell_size = mp.Vector3(px,py,sz)

    boundary_layers = [mp.PML(thickness=dpml,direction=mp.Z)]

    # periodic boundary conditions
    k_point = mp.Vector3()

    ## disabled due to https://github.com/NanoComp/meep/issues/132
    # symmetries = [mp.Mirror(direction=mp.Y)]

    # plane of incidence is XZ
    # P/TM polarization: Ex, S/TE polarization: Ey
    src_cmpt = mp.Ex if P_pol else mp.Ey
    sources = [mp.Source(src=mp.GaussianSource(fcen,fwidth=0.2*fcen),
                         size=mp.Vector3(px,py,0),
                         center=mp.Vector3(0,0,-0.5*sz+dpml),
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

    sim.run(until_after_sources=mp.stop_when_dft_decayed())

    input_flux = mp.get_fluxes(flux)

    sim.reset_meep()

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
                        k_point=k_point)

    flux = sim.add_mode_monitor(fcen,
                                0,
                                1,
                                mp.ModeRegion(center=mp.Vector3(0,0,0.5*sz-dpml),
                                              size=mp.Vector3(px,py,0)))

    sim.run(until_after_sources=mp.stop_when_dft_decayed())

    res = sim.get_eigenmode_coefficients(flux,
                                         mp.DiffractedPlanewave((1,0,0),
                                                                mp.Vector3(0,0,1),
                                                                0 if P_pol else 1,
                                                                1 if P_pol else 0))

    coeffs = res.alpha
    Tmeep = abs(coeffs[0,0,0])**2 / input_flux[0]

    n1 = 1.0
    n2 = nSiO2
    Tfresnel = 1 - (n1-n2)**2 / (n1+n2)**2
    diff_eff_P = 0.9114 # P-pol. diffraction efficiency computed using Reticolo (RCWA)
    diff_eff_S = 0.9514 # S-pol. diffraction efficiency computed using Reticolo (RCWA)
    # convert diffraction efficiency into transmittance in the Z direction
    Treticolo = (diff_eff_P if P_pol else diff_eff_S) * Tfresnel * math.cos(theta_d)
    err = abs(Tmeep - Treticolo) / Treticolo
    print("err:, {}, {:.6f} (Meep), {:.6f} (reticolo), {:.6f} (error),".format('P' if P_pol else 'S',Tmeep,Treticolo,err))

    # for debugging:
    # visualize cross sections of the computational cell
    # to ensure that metagrating matches expected design
    if 0:
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

    return err

if __name__ == '__main__':
    metagrating(False)
    metagrating(True)
