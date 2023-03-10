import numpy as np
import unittest
import ruler_convolution_formatted_0219 as ruler
from regular_shapes import disc, rounded_square

resolution = 1
phys_size = (200, 200)
message = "The estimated minimum length scale is too far from the declared minimum length scale."


class TestRuler(unittest.TestCase):
    def test_rounded_square(self):
        declared_mls = 50
        delta = 9

        for angle in range(0, 90, 10):
            print("Rotation angle of the rounded square: ", angle)
            pattern = rounded_square(resolution, phys_size, declared_mls,
                                     angle)
            solid_mls = ruler.solid_minimum_length(pattern, phys_size)
            # check if values are almost equal
            self.assertAlmostEqual(solid_mls, declared_mls, None, message,
                                   delta)

    def test_disc(self):
        diameter = 50
        pattern = disc(resolution, phys_size, diameter)
        solid_mls = ruler.solid_minimum_length(pattern, phys_size)
        # check if values are almost equal
        self.assertAlmostEqual(solid_mls, diameter, None, message, delta=5)

    def test_ring(self):
        outer_diameter, inner_diameter = 120, 50
        declared_solid_mls, declared_void_mls = (
            outer_diameter - inner_diameter) / 2, inner_diameter

        solid_disc = disc(resolution, phys_size, diameter=outer_diameter)
        void_disc = disc(resolution, phys_size, diameter=inner_diameter)
        pattern = solid_disc ^ void_disc  # ring

        solid_mls = ruler.solid_minimum_length(pattern, phys_size)
        void_mls = ruler.void_minimum_length(pattern, phys_size)
        dual_mls = ruler.dual_minimum_length(pattern, phys_size)

        delta = 1
        # check if values are almost equal
        self.assertAlmostEqual(solid_mls, declared_solid_mls, None, message,
                               delta)
        self.assertAlmostEqual(void_mls, declared_void_mls, None, message,
                               delta)
        self.assertAlmostEqual(dual_mls,
                               min(declared_solid_mls, declared_void_mls),
                               None, message, delta)


if __name__ == "__main__":
    unittest.main()