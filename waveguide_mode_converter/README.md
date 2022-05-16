# Waveguide Mode Converter

This is a waveguide mode converter test problem. The design objective is to maximize the conversion of power from the
fundamental waveguide mode of the input waveguide to the second-order mode of the output waveguide. The geometry is two-dimensional.

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

These parameter values correspond to the mode converter demonstrated in
[Inverse design of photonic devices with strict foundry fabrication constraints](https://arxiv.org/abs/2201.12965).