"""Runs the mode converter test problem on a specified list of designs.

This script runs the mode converter test problem on specified list of
design variables, printing CSV-formatted performance metrics. Worst
case reflection and transmission over the sampled wavelengths are
reported for each design.
"""

import argparse
import os
import numpy as np
import ceviche_challenges
import ceviche_challenges.units as u
import sys

sys.path.append('../')
import ruler


INPUT_PORT_AXIS = 1
INPUT_PORT_IDX = 0
OUTPUT_PORT_IDX = 1

parser = argparse.ArgumentParser()
parser.add_argument(
    'file',
    type=str,
    nargs='+',
    help='Paths to CSV files containing the design variable arrays.',
)
args = parser.parse_args()

design_size = (1600., 1600.)
wavelengths_nm = (1265., 1270., 1275., 1285., 1290., 1295.)

params = ceviche_challenges.mode_converter.prefabs.mode_converter_sim_params(
    wavelengths=u.Array(wavelengths_nm, u.nm),
    resolution=10 * u.nm,
)
spec = ceviche_challenges.mode_converter.prefabs.mode_converter_spec_12(
    variable_region_size=(design_size[0] * u.nm, design_size[1] * u.nm),
    cladding_permittivity=2.25,
    slab_permittivity=12.25,
    left_wg_width=400 * u.nm,
    right_wg_width=400 * u.nm,
)
model = ceviche_challenges.mode_converter.model.ModeConverterModel(
    params,
    spec,
)

print('# Design file, Length scale (nm), Worst-case reflection (dB), Worst-case transmission (dB)')
for file in args.file:
    design = np.loadtxt(file, delimiter=',')

    s_params, _ = model.simulate(design)

    s_params_dB = 20 * np.log10(np.abs(s_params.squeeze(axis=INPUT_PORT_AXIS)))
    reflection_dB_worst = np.max(s_params_dB[:, INPUT_PORT_IDX])
    transmission_dB_worst = np.min(s_params_dB[:, OUTPUT_PORT_IDX])

    length_scale = ruler.minimum_length(
        model.density(design),
        [v * params.resolution.to_value(u.nm) for v in model.shape]
    )

    print(f'{os.path.basename(file)}, {length_scale}, {reflection_dB_worst:.2f}, {transmission_dB_worst:.2f}')
