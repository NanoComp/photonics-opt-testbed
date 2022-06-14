# Waveguide Mode Converter

This is a waveguide mode converter test problem. The design objective is to maximize the conversion of power from the fundamental waveguide mode of the input waveguide to the second-order mode of the output waveguide. Specifically, the worst-cast (maximum) reflection is minimized and the worst case transmission (minimum) is maximized. The geometry of the device is two-dimensional.

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

| File                                               | Description                                             |
|----------------------------------------------------|---------------------------------------------------------|
| converter_schubert_circle_x33491673_w307_s134.csv  | Circular brush design from Schubert et al. (2022)       |
| converter_schubert_notched_x33491673_w183_s159.csv | Notched-square brush design from Schubert et al. (2022) |

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
