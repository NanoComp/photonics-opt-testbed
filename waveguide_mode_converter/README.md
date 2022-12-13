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
refl:, 1.265, 0.015235, -18.171593
refl:, 1.270, 0.015599, -18.068902
refl:, 1.275, 0.014972, -18.247159
refl:, 1.285, 0.010893, -19.628648
refl:, 1.290, 0.008355, -20.780394
refl:, 1.295, 0.006412, -21.929742
worst-case reflectance (dB):, -18.068902
```

**transmittance into mode 2: wavelength (μm), transmittance, transmittance (dB)**
```
tran:, 1.265, 0.893876, -0.487228
tran:, 1.270, 0.888506, -0.513397
tran:, 1.275, 0.883389, -0.538478
tran:, 1.285, 0.880024, -0.555053
tran:, 1.290, 0.883479, -0.538036
tran:, 1.295, 0.889466, -0.508708
worst-case transmittance (dB):, -0.555053
```

2. Minimum lengthscale constraint: 60 nm.

meausred lengthscale: 63 nm.

file: converter_meep_min_linewidth_60nm.csv

**reflectance into mode 1: wavelength (μm), reflectance, reflectance (dB)**
```
refl:, 1.265, 0.007321, -21.354079
refl:, 1.270, 0.007807, -21.075048
refl:, 1.275, 0.008329, -20.793956
refl:, 1.285, 0.009256, -20.335547
refl:, 1.290, 0.009564, -20.193607
refl:, 1.295, 0.009713, -20.126394
worst-case reflectance (dB):, -20.126394
```

**transmittance into mode 2: wavelength (μm), transmittance, transmittance (dB)**
```
tran:, 1.265, 0.736645, -1.327418
tran:, 1.270, 0.743068, -1.289713
tran:, 1.275, 0.748724, -1.256785
tran:, 1.285, 0.757658, -1.205266
tran:, 1.290, 0.760769, -1.187470
tran:, 1.295, 0.762757, -1.176135
worst-case transmittance (dB):, -1.327418
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

measured lengthscale: 106 nm

converter_meep_min_linewidth_90nm.csv

**reflectance into mode 1: wavelength (μm), reflectance, reflectance (dB)**
```
refl:, 1.265, 0.044141, -13.551562
refl:, 1.270, 0.045734, -13.397585
refl:, 1.275, 0.047568, -13.226838
refl:, 1.285, 0.051794, -12.857210
refl:, 1.290, 0.054098, -12.668218
refl:, 1.295, 0.056479, -12.481139
worst-case reflectance (dB):, -12.481139
```

**transmittance into mode 2: wavelength (μm), transmittance, transmittance (dB)**
```
tran:, 1.265, 0.665760, -1.766824
tran:, 1.270, 0.665468, -1.768726
tran:, 1.275, 0.664588, -1.774473
tran:, 1.285, 0.660871, -1.798835
tran:, 1.290, 0.657982, -1.817857
tran:, 1.295, 0.654382, -1.841689
worst-case transmittance (dB):, -1.841689
```
