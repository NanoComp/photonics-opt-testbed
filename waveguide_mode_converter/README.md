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

For comparison, the following are the reflectance and transmittance spectra for the topology-optimized designs using Meep with various minimum lengthscales.

1. Minimum lengthscale constraint: 50 nm.

measured lengthscale: 63 nm.

file: converter_meep_min_linewidth_50nm.csv

**reflectance into mode 1: wavelength (μm), reflectance, reflectance (dB)**
```
refl:, 1.265, 0.009628, -20.164705
refl:, 1.270, 0.009643, -20.158021
refl:, 1.275, 0.009707, -20.128966
refl:, 1.285, 0.009950, -20.021659
refl:, 1.290, 0.010118, -19.949185
refl:, 1.295, 0.010312, -19.866366
worst-case reflectance (dB):, -19.866366
```

**transmittance into mode 2: wavelength (μm), transmittance, transmittance (dB)**
```
tran:, 1.265, 0.920025, -0.362003
tran:, 1.270, 0.922174, -0.351873
tran:, 1.275, 0.924045, -0.343067
tran:, 1.285, 0.927015, -0.329132
tran:, 1.290, 0.928125, -0.323937
tran:, 1.295, 0.928992, -0.319879
worst-case transmittance (dB):, -0.362003
```

2. Minimum lengthscale constraint: 60 nm.

measured lengthscale: 81 nm.

file: converter_meep_min_linewidth_60nm.csv

**reflectance into mode 1: wavelength (μm), reflectance, reflectance (dB)**
```
refl:, 1.265, 0.000374, -34.275727
refl:, 1.270, 0.000286, -35.443571
refl:, 1.275, 0.000265, -35.774808
refl:, 1.285, 0.000372, -34.299325
refl:, 1.290, 0.000473, -33.255701
refl:, 1.295, 0.000588, -32.305168
worst-case reflectance (dB):, -32.305168
```

**transmittance into mode 2: wavelength (μm), transmittance, transmittance (dB)**
```
tran:, 1.265, 0.963868, -0.159824
tran:, 1.270, 0.963965, -0.159388
tran:, 1.275, 0.963685, -0.160648
tran:, 1.285, 0.962119, -0.167712
tran:, 1.290, 0.960892, -0.173256
tran:, 1.295, 0.959421, -0.179906
worst-case transmittance (dB):, -0.179906
```

3. Minimum lengthscale constraint: 70 nm.

measured lengthscale: 100 nm.

file: converter_meep_min_linewidth_70nm.csv

**reflectance into mode 1: wavelength (μm), reflectance, reflectance (dB)**
```
refl:, 1.265, 0.019693, -17.056968
refl:, 1.270, 0.017361, -17.604295
refl:, 1.275, 0.015615, -18.064702
refl:, 1.285, 0.013498, -18.697453
refl:, 1.290, 0.012916, -18.888789
refl:, 1.295, 0.012493, -19.033184
worst-case reflectance (dB):, -17.056968
```

**transmittance into mode 2: wavelength (μm), transmittance, transmittance (dB)**
```
tran:, 1.265, 0.667054, -1.758389
tran:, 1.270, 0.675296, -1.705058
tran:, 1.275, 0.683075, -1.655316
tran:, 1.285, 0.696382, -1.571525
tran:, 1.290, 0.700852, -1.543736
tran:, 1.295, 0.702586, -1.533007
worst-case transmittance (dB):, -1.758389
```

4. Minimum lengthscale constraint: 80 nm.

measured lengthscale: 75 nm.

file: converter_meep_min_linewidth_80nm.csv

**reflectance into mode 1: wavelength (μm), reflectance, reflectance (dB)**
```
refl:, 1.265, 0.061416, -12.117155
refl:, 1.270, 0.062535, -12.038793
refl:, 1.275, 0.063447, -11.975903
refl:, 1.285, 0.064677, -11.892475
refl:, 1.290, 0.065058, -11.866985
refl:, 1.295, 0.065391, -11.844820
worst-case reflectance (dB):, -11.844820
```

**transmittance into mode 2: wavelength (μm), transmittance, transmittance (dB)**
```
tran:, 1.265, 0.679133, -1.680453
tran:, 1.270, 0.677549, -1.690590
tran:, 1.275, 0.674796, -1.708273
tran:, 1.285, 0.665450, -1.768848
tran:, 1.290, 0.658723, -1.812969
tran:, 1.295, 0.650600, -1.866862
worst-case transmittance (dB):, -1.866862
```

5. Minimum lengthscale constraint: 90 nm.

measured lengthscale: 113 nm

converter_meep_min_linewidth_90nm.csv

**reflectance into mode 1: wavelength (μm), reflectance, reflectance (dB)**
```
refl:, 1.265, 0.007773, -21.094135
refl:, 1.270, 0.007432, -21.288749
refl:, 1.275, 0.007204, -21.424158
refl:, 1.285, 0.007083, -21.497768
refl:, 1.290, 0.007142, -21.461586
refl:, 1.295, 0.007244, -21.400120
worst-case reflectance (dB):, -21.094135
```

**transmittance into mode 2: wavelength (μm), transmittance, transmittance (dB)**
```
tran:, 1.265, 0.619815, -2.077378
tran:, 1.270, 0.631644, -1.995273
tran:, 1.275, 0.643311, -1.915788
tran:, 1.285, 0.666046, -1.764957
tran:, 1.290, 0.676643, -1.696403
tran:, 1.295, 0.686333, -1.634649
worst-case transmittance (dB):, -2.077378
```
