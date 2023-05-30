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

| File                                                     | Description                                                                                             |
|----------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| converter_schubert_circle_x33491673_w307_s134.csv        | Circular brush design from [Schubert et al. (2022)](https://doi.org/10.1021/acsphotonics.2c00313)       |
| converter_schubert_notched_x33491673_w183_s159.csv       | Notched-square brush design from [Schubert et al. (2022)](https://doi.org/10.1021/acsphotonics.2c00313) |
| converter_generator_circle_{6,8,10,12,14,16,18,20}\*.csv | Circular brush designs of varying minimum feature size optimized via Ceviche                            |
| converter_meep_\*.csv                                    | Designs produced using a filter+projection scheme optimized via Meep.                                   |

## Usage

The mode converter test problem can be run on a list of specified designs via
the `run.py` script.

Running the following:

```
python3 ./run.py designs/converter_*.csv
```

Will produce the output below:
```
# Design file, Length scale (nm), Worst-case reflection (dB), Worst-case transmission (dB)
converter_generator_circle_10_x47530832_w11_s483.csv, 90.2978515625, -22.50, -0.19
converter_generator_circle_10_x47530832_w19_s483.csv, 90.2978515625, -28.30, -0.29
converter_generator_circle_10_x47530832_w27_s681.csv, 101.9775390625, -27.17, -0.19
converter_generator_circle_10_x47530832_w35_s719.csv, 101.9775390625, -34.01, -0.24
converter_generator_circle_10_x47530832_w3_s584.csv, 101.9775390625, -36.30, -0.18
converter_generator_circle_10_x47530832_w43_s590.csv, 101.9775390625, -37.79, -0.12
converter_generator_circle_10_x47530832_w51_s853.csv, 90.2978515625, -37.49, -0.22
converter_generator_circle_10_x47530832_w59_s920.csv, 90.2978515625, -23.84, -0.21
converter_generator_circle_10_x47530832_w67_s350.csv, 90.2978515625, -29.21, -0.19
converter_generator_circle_10_x47530832_w75_s547.csv, 90.2978515625, -32.32, -0.16
converter_generator_circle_12_x47530832_w12_s248.csv, 119.4970703125, -28.49, -0.35
converter_generator_circle_12_x47530832_w20_s640.csv, 119.4970703125, -28.00, -0.29
converter_generator_circle_12_x47530832_w28_s528.csv, 119.4970703125, -32.18, -0.16
converter_generator_circle_12_x47530832_w36_s982.csv, 119.4970703125, -28.46, -0.35
converter_generator_circle_12_x47530832_w44_s653.csv, 119.4970703125, -22.86, -0.34
converter_generator_circle_12_x47530832_w4_s804.csv, 119.4970703125, -21.20, -0.57
converter_generator_circle_12_x47530832_w52_s954.csv, 119.4970703125, -20.46, -0.53
converter_generator_circle_12_x47530832_w60_s857.csv, 119.4970703125, -29.14, -0.24
converter_generator_circle_12_x47530832_w68_s88.csv, 119.4970703125, -33.10, -0.52
converter_generator_circle_12_x47530832_w76_s364.csv, 119.4970703125, -26.20, -0.48
converter_generator_circle_14_x47530832_w13_s793.csv, 142.8564453125, -33.62, -0.19
converter_generator_circle_14_x47530832_w21_s956.csv, 142.8564453125, -35.49, -0.25
converter_generator_circle_14_x47530832_w29_s991.csv, 142.8564453125, -34.11, -0.51
converter_generator_circle_14_x47530832_w37_s975.csv, 142.8564453125, -33.08, -0.30
converter_generator_circle_14_x47530832_w45_s878.csv, 142.8564453125, -29.39, -0.29
converter_generator_circle_14_x47530832_w53_s519.csv, 142.8564453125, -24.07, -0.43
converter_generator_circle_14_x47530832_w5_s139.csv, 142.8564453125, -20.67, -0.58
converter_generator_circle_14_x47530832_w61_s667.csv, 142.8564453125, -28.21, -0.30
converter_generator_circle_14_x47530832_w69_s668.csv, 142.8564453125, -28.21, -0.22
converter_generator_circle_14_x47530832_w77_s765.csv, 142.8564453125, -25.35, -0.25
converter_generator_circle_16_x47530832_w14_s150.csv, 166.2158203125, -21.23, -0.81
converter_generator_circle_16_x47530832_w22_s167.csv, 166.2158203125, -19.71, -1.08
converter_generator_circle_16_x47530832_w30_s624.csv, 166.2158203125, -26.43, -0.41
converter_generator_circle_16_x47530832_w38_s404.csv, 166.2158203125, -17.62, -0.88
converter_generator_circle_16_x47530832_w46_s770.csv, 166.2158203125, -23.29, -0.77
converter_generator_circle_16_x47530832_w54_s109.csv, 166.2158203125, -19.77, -0.77
converter_generator_circle_16_x47530832_w62_s193.csv, 166.2158203125, -29.32, -0.61
converter_generator_circle_16_x47530832_w6_s445.csv, 166.2158203125, -18.98, -0.67
converter_generator_circle_16_x47530832_w70_s892.csv, 166.2158203125, -20.63, -0.81
converter_generator_circle_16_x47530832_w78_s371.csv, 166.2158203125, -27.60, -0.81
converter_generator_circle_18_x47530832_w15_s879.csv, 169.1357421875, -23.66, -0.93
converter_generator_circle_18_x47530832_w23_s228.csv, 169.1357421875, -22.50, -0.96
converter_generator_circle_18_x47530832_w31_s46.csv, 183.7353515625, -23.49, -0.86
converter_generator_circle_18_x47530832_w39_s593.csv, 183.7353515625, -28.45, -0.67
converter_generator_circle_18_x47530832_w47_s439.csv, 183.7353515625, -31.18, -0.60
converter_generator_circle_18_x47530832_w55_s534.csv, 183.7353515625, -27.49, -0.79
converter_generator_circle_18_x47530832_w63_s544.csv, 183.7353515625, -19.35, -0.76
converter_generator_circle_18_x47530832_w71_s711.csv, 183.7353515625, -27.51, -0.74
converter_generator_circle_18_x47530832_w79_s267.csv, 183.7353515625, -26.93, -0.87
converter_generator_circle_18_x47530832_w7_s311.csv, 183.7353515625, -22.70, -0.86
converter_generator_circle_20_x47530832_w16_s416.csv, 204.1748046875, -18.43, -1.23
converter_generator_circle_20_x47530832_w24_s997.csv, 204.1748046875, -20.84, -0.95
converter_generator_circle_20_x47530832_w32_s982.csv, 201.2548828125, -24.16, -0.66
converter_generator_circle_20_x47530832_w40_s988.csv, 207.0947265625, -18.16, -1.34
converter_generator_circle_20_x47530832_w48_s996.csv, 201.2548828125, -19.63, -0.95
converter_generator_circle_20_x47530832_w56_s972.csv, 201.2548828125, -29.46, -0.86
converter_generator_circle_20_x47530832_w64_s997.csv, 207.0947265625, -32.20, -0.85
converter_generator_circle_20_x47530832_w72_s846.csv, 201.2548828125, -22.31, -0.83
converter_generator_circle_20_x47530832_w80_s599.csv, 204.1748046875, -25.08, -1.04
converter_generator_circle_20_x47530832_w8_s938.csv, 201.2548828125, -20.08, -0.90
converter_generator_circle_6_x47530832_w17_s412.csv, 64.0185546875, -41.09, -0.14
converter_generator_circle_6_x47530832_w1_s796.csv, 64.0185546875, -29.96, -0.09
converter_generator_circle_6_x47530832_w25_s986.csv, 64.0185546875, -34.60, -0.07
converter_generator_circle_6_x47530832_w33_s242.csv, 64.0185546875, -30.74, -0.08
converter_generator_circle_6_x47530832_w41_s990.csv, 64.0185546875, -33.51, -0.09
converter_generator_circle_6_x47530832_w49_s770.csv, 64.0185546875, -40.64, -0.07
converter_generator_circle_6_x47530832_w57_s968.csv, 64.0185546875, -34.17, -0.06
converter_generator_circle_6_x47530832_w65_s909.csv, 64.0185546875, -41.95, -0.04
converter_generator_circle_6_x47530832_w73_s975.csv, 64.0185546875, -37.55, -0.05
converter_generator_circle_6_x47530832_w9_s893.csv, 64.0185546875, -33.79, -0.08
converter_generator_circle_8_x47530832_w10_s898.csv, 84.4580078125, -30.73, -0.13
converter_generator_circle_8_x47530832_w18_s655.csv, 84.4580078125, -32.76, -0.13
converter_generator_circle_8_x47530832_w26_s710.csv, 84.4580078125, -33.98, -0.09
converter_generator_circle_8_x47530832_w2_s430.csv, 84.4580078125, -36.23, -0.09
converter_generator_circle_8_x47530832_w34_s965.csv, 84.4580078125, -27.76, -0.11
converter_generator_circle_8_x47530832_w42_s878.csv, 84.4580078125, -27.34, -0.15
converter_generator_circle_8_x47530832_w50_s956.csv, 84.4580078125, -28.26, -0.06
converter_generator_circle_8_x47530832_w58_s969.csv, 84.4580078125, -28.66, -0.15
converter_generator_circle_8_x47530832_w66_s878.csv, 84.4580078125, -29.08, -0.13
converter_generator_circle_8_x47530832_w74_s989.csv, 84.4580078125, -34.79, -0.11
converter_meep_min_linewidth_100nm.csv, 84.4580078125, -13.02, -2.09
converter_meep_min_linewidth_125nm.csv, 119.4970703125, -21.25, -2.51
converter_meep_min_linewidth_150nm.csv, 151.6162109375, -16.68, -2.08
converter_meep_min_linewidth_175nm.csv, 166.2158203125, -24.44, -2.80
converter_meep_min_linewidth_200nm.csv, 306.3720703125, -29.28, -0.85
converter_meep_min_linewidth_225nm.csv, 361.8505859375, -29.72, -3.96
converter_meep_min_linewidth_50nm.csv, 55.2587890625, -33.33, -0.07
converter_meep_min_linewidth_60nm.csv, 72.7783203125, -31.49, -0.09
converter_meep_min_linewidth_70nm.csv, 75.6982421875, -25.88, -0.26
converter_meep_min_linewidth_80nm.csv, 96.1376953125, -28.40, -0.34
converter_meep_min_linewidth_90nm.csv, 84.4580078125, -25.90, -0.62
converter_schubert_circle_x33491673_w307_s134.csv, 99.0576171875, -34.11, -0.19
converter_schubert_notched_x33491673_w183_s159.csv, 104.8974609375, -30.67, -0.26
```
