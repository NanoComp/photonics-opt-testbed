# Waveguide Mode Converter

This is a waveguide mode converter test problem. The design objective is to maximize the conversion of power from the fundamental waveguide mode of the input waveguide to the second-order mode of the output waveguide for the out-of-plane (Ez) polarization. Specifically, the worst-cast (maximum) reflection is minimized and the worst case transmission (minimum) is maximized. The geometry of the device is two-dimensional.

## Schematic

![Waveguide mode converter schematic](mode_converter_schematic.png)

## Parameters

| Parameter                            | Value           |
|--------------------------------------|-----------------|
| Design region width, dx              | 1.6 um          |
| Design region height, dy             | 1.6 um          |
| Waveguide width                      | 400 nm          |
| Solid material relative permittivity | 12.25 (silicon) |
| Void material relative permittivity  | 2.25 (oxide)    |
| Operating wavelength range           | 1260 - 1300 nm  |
| Simulation resolution                | 10 nm           |
| Design pixel size                    | 10 nm           |
| Expected design array shape          | 160 x 160       |

These parameter values correspond to the mode converter demonstrated in
[Inverse Design of Photonic Devices with Strict Foundry Fabrication Constraints](https://doi.org/10.1021/acsphotonics.2c00313).

## Designs

Mode converter designs are under the `designs/` subfolder.

| File                                               | Description                                                                                             |
|----------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| converter_schubert_circle_x33491673_w307_s134.csv  | Circular brush design from [Schubert et al. (2022)](https://doi.org/10.1021/acsphotonics.2c00313)       |
| converter_schubert_notched_x33491673_w183_s159.csv | Notched-square brush design from [Schubert et al. (2022)](https://doi.org/10.1021/acsphotonics.2c00313) |

## Usage

The mode converter test problem can be run on a list of specified designs via
the `run.sh` script. For example,

```
python3 ./run.py designs/converter_schubert_circle_x33491673_w307_s134.csv designs/converter_schubert_notched_x33491673_w183_s159.csv
```

The script will print out CSV-formatted performance metrics. For example,

```
# Worst-case reflection (dB), Worst-case transmission (dB)
-34.11, -0.19
-30.67, -0.26
```

For comparison, the following are the worst-case reflectance and
transmittance spectra for the topology-optimized designs using Meep
with various minimum lengthscales.

The data consists of four columns: (1) reflectance (`refl`) or
transmittance (`tran`), (2) wavelength (μm), (3)
reflectance/transmittance (linear scale in [0,1]), and (4)
reflectance/transmittance (log scale in dB).

1. Minimum lengthscale constraint: 50 nm.

measured lengthscale: 63 nm.

file: converter_meep_min_linewidth_50nm.csv

```
refl:, 1.295, 0.010312, -19.866366
tran:, 1.265, 0.920025, -0.362003
```

2. Minimum lengthscale constraint: 60 nm.

measured lengthscale: 81 nm.

file: converter_meep_min_linewidth_60nm.csv

```
refl:, 1.295, 0.000588, -32.305168
tran:, 1.295, 0.959421, -0.179906
```

3. Minimum lengthscale constraint: 70 nm.

measured lengthscale: 81 nm.

file: converter_meep_min_linewidth_70nm.csv

```
refl:, 1.295, 0.001660, -27.799095
tran:, 1.265, 0.901512, -0.450283
```

4. Minimum lengthscale constraint: 80 nm.

measured lengthscale: 106 nm.

file: converter_meep_min_linewidth_80nm.csv

```
refl:, 1.295, 0.001188, -29.253338
tran:, 1.275, 0.920259, -0.360899
```

5. Minimum lengthscale constraint: 90 nm.

measured lengthscale: 113 nm.

file: converter_meep_min_linewidth_90nm.csv

```
refl:, 1.265, 0.007773, -21.094135
tran:, 1.265, 0.619815, -2.077378
```

6. Minimum lengthscale constraint: 100 nm.

measured lengthscale: 88 nm.

file: converter_meep_min_linewidth_100nm.csv

```
refl:, 1.295, 0.044475, -13.518838
tran:, 1.265, 0.608069, -2.160469
```

7. Minimum lengthscale constraint: 125 nm.

```
refl:, 1.295, 0.014873, -18.275893
tran:, 1.295, 0.427343, -3.692235
```

8. Minimum lengthscale constraint: 150 nm.

measured lengthscale: 163 nm.

file: converter_meep_min_linewidth_150nm.csv

```
refl:, 1.295, 0.016925, -17.714820
tran:, 1.265, 0.617705, -2.092192
```

9. Minimum lengthscale constraint: 175 nm.

measured lengthscale: 175 nm.

file: converter_meep_min_linewidth_175nm.csv

```
refl:, 1.265, 0.003253, -24.877215
tran:, 1.265, 0.518961, -2.848656
```

10. Minimum lengthscale constraint: 200 nm.

measured lengthscale: 325 nm.

file: converter_meep_min_linewidth_200nm.csv

```
refl:, 1.295, 0.001232, -29.093873
tran:, 1.265, 0.816701, -0.879367
```

11. Minimum lengthscale constraint: 225 nm.

measured lengthscale: 275 nm.

file: converter_meep_min_linewidth_275nm.csv

```
refl:, 1.265, 0.015390, -18.127650
tran:, 1.290, 0.273556, -5.629535
```
