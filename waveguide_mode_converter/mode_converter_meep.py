"""Computes the worst-case reflectance and transmittance of a mode converter.

This script uses Meep to compute the reflectance and transmittance over a
number of sampled wavelengths of a mode converter imported as an image.
The results are printed in CSV format including the worst-case values.
"""

import numpy as np
import meep as mp
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def mode_converter(design_fname: str):
    resolution = 100 # pixels/μm

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

    # wavelengths to monitor
    wvls = (1.265, 1.270, 1.275, 1.285, 1.290, 1.295)
    frqs = [1/wvl for wvl in wvls]

    # pulsed source center frequency and bandwidth
    wvl_min = 1.26
    wvl_max = 1.30
    frq_min = 1/wvl_max
    frq_max = 1/wvl_min
    fcen = 0.5*(frq_min+frq_max)
    df = frq_max-frq_min

    eig_parity = mp.ODD_Z
    src_pt = mp.Vector3(-0.5*sx+dpml,0)
    sources = [mp.EigenModeSource(src=mp.GaussianSource(fcen,fwidth=df),
                                  size=mp.Vector3(0,sy),
                                  center=src_pt,
                                  eig_band=1,
                                  eig_parity=eig_parity)]

    nSiO2 = 1.5
    SiO2 = mp.Medium(index=nSiO2)
    nSi = 3.5
    Si = mp.Medium(index=nSi)

    geometry = [mp.Block(size=mp.Vector3(mp.inf,w,mp.inf),
                         center=mp.Vector3(),
                         material=Si)]

    sim = mp.Simulation(resolution=resolution,
                        default_material=SiO2,
                        cell_size=cell_size,
                        sources=sources,
                        geometry=geometry,
                        boundary_layers=pml_layers,
                        k_point=mp.Vector3())

    refl_pt = mp.Vector3(-0.5*sx+dpml+0.5*l)
    refl_mon = sim.add_mode_monitor(frqs,
                                    mp.ModeRegion(center=refl_pt,
                                                  size=mp.Vector3(0,sy,0)))

    stop_cond = mp.stop_when_fields_decayed(50, mp.Ez, refl_pt, 1e-7)
    sim.run(until_after_sources=stop_cond)

    res = sim.get_eigenmode_coefficients(refl_mon,
                                         [1],
                                         eig_parity=eig_parity)

    coeffs = res.alpha
    input_flux = np.abs(coeffs[0,:,0])**2
    input_flux_data = sim.get_flux_data(refl_mon)

    plt.figure()
    sim.plot2D()
    if mp.am_master():
        plt.savefig('waveguide_sim_layout.png',dpi=150,bbox_inches='tight')

    sim.reset_meep()

    # image resolution is 100 pixels/μm and thus
    # Meep resolution should not be larger than ~half this value
    weights = np.genfromtxt('designs/' + design_fname,delimiter=',')
    Nx, Ny = weights.shape

    geometry = [mp.Block(size=mp.Vector3(mp.inf,w,mp.inf),
                         center=mp.Vector3(),
                         material=Si),
                mp.Block(size=mp.Vector3(dx,dy,mp.inf),
                         center=mp.Vector3(),
                         material=mp.MaterialGrid(grid_size=mp.Vector3(Nx,Ny),
                                                  medium1=SiO2,
                                                  medium2=Si,
                                                  weights=weights))]

    sim = mp.Simulation(resolution=resolution,
                        default_material=SiO2,
                        cell_size=cell_size,
                        sources=sources,
                        geometry=geometry,
                        boundary_layers=pml_layers,
                        k_point=mp.Vector3())

    refl_mon = sim.add_mode_monitor(frqs,
                                    mp.ModeRegion(center=refl_pt,
                                                  size=mp.Vector3(0,sy,0)))
    sim.load_minus_flux_data(refl_mon, input_flux_data)

    tran_pt = mp.Vector3(0.5*sx-dpml-0.5*l)
    tran_mon = sim.add_mode_monitor(frqs,
                                    mp.ModeRegion(center=tran_pt,
                                                  size=mp.Vector3(0,sy,0)))

    sim.run(until_after_sources=stop_cond)

    res = sim.get_eigenmode_coefficients(refl_mon,
                                         [1],
                                         eig_parity=eig_parity)

    coeffs = res.alpha
    refl = np.abs(coeffs[0,:,1])**2 / input_flux
    refl_flux = -1 * np.array(mp.get_fluxes(refl_mon))

    res = sim.get_eigenmode_coefficients(tran_mon,
                                         [2],
                                         eig_parity=eig_parity)

    coeffs = res.alpha
    tran = np.abs(coeffs[0,:,0])**2 / input_flux
    tran_flux = np.array(mp.get_fluxes(tran_mon))

    sparam_to_dB = lambda s: 20 * np.log10(s)

    for idx,wvl in enumerate(wvls):
        print("refl:, {:.3f}, {:.6f}, {:.6f}".format(wvl,
                                                     refl[idx],
                                                     sparam_to_dB(refl[idx])))

    print("worst-case reflectance (dB):, {:.6f}".format(sparam_to_dB(np.amax(refl))))

    for idx,wvl in enumerate(wvls):
        print("tran:, {:.3f}, {:.6f}, {:.6f}".format(wvl,
                                                     tran[idx],
                                                     sparam_to_dB(tran[idx])))

    print("worst-case transmittance (dB):, {:.6f}".format(sparam_to_dB(np.amin(tran))))

    # compute the total reflectance (R) and transmittance (T) and their sum
    # the scattered power is thus 1-R-T
    R = refl_flux / input_flux
    T = tran_flux / input_flux
    for idx,wvl in enumerate(wvls):
        print("flux:, {:.3f}, {:.6f}, {:.6f}, {:.6f}".format(wvl,
                                                             R[idx],
                                                             T[idx],
                                                             R[idx]+T[idx]))

    plt.figure()
    sim.plot2D()
    if mp.am_master():
        plt.savefig('mode_converter_sim_layout.png',dpi=150,bbox_inches='tight')


if __name__ == '__main__':
    mode_converter('converter_schubert_notched_x33491673_w183_s159.csv')
    mode_converter('converter_schubert_circle_x33491673_w307_s134.csv')
