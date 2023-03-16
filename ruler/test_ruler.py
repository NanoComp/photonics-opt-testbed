import numpy as np
import unittest
import ruler
from regular_shapes import disc, rounded_square

resolution = 1
phys_size = (200, 200)
message = "The estimated minimum length scale is too far from the declared minimum length scale."


class TestRuler(unittest.TestCase):
    def test_rounded_square(self):
        print("------ Testing ruler on rounded squares ------")
        declared_mls = 50
        print("Declared minimum length scale: ", declared_mls)
        delta = 4

        for angle in range(0, 90, 10):
            pattern = rounded_square(resolution, phys_size, declared_mls,
                                     angle)
            solid_mls = ruler.minimum_length_solid(pattern)
            print("Rotation angle of the rounded square: ", angle)
            print("Estimated minimum length scale: ", solid_mls)
            # check if values are almost equal
            self.assertAlmostEqual(solid_mls, declared_mls, None, message,
                                   delta)

    def test_disc(self):
        print("------ Testing ruler on a disc ------")
        diameter = 50
        print("Declared minimum length scale: ", diameter)
        pattern = disc(resolution, phys_size, diameter)
        solid_mls = ruler.minimum_length_solid(pattern, phys_size)
        print("Estimated minimum length scale: ", solid_mls)
        # check if values are almost equal
        self.assertAlmostEqual(solid_mls, diameter, None, message, delta=1)

    def test_ring(self):
        print("------ Testing ruler on concentric circles ------")
        outer_diameter, inner_diameter = 120, 50
        declared_solid_mls, declared_void_mls = (
            outer_diameter - inner_diameter) / 2, inner_diameter
        print("Declared minimum length scale: ", declared_solid_mls,
              "(solid), ", declared_void_mls, "(void)")

        solid_disc = disc(resolution, phys_size, diameter=outer_diameter)
        void_disc = disc(resolution, phys_size, diameter=inner_diameter)
        pattern = solid_disc ^ void_disc  # ring

        solid_mls = ruler.minimum_length_solid(pattern)
        void_mls = ruler.minimum_length_void(pattern)
        dual_mls = ruler.minimum_length(pattern)
        print("Estimated minimum length scale: ", solid_mls, "(solid), ",
              void_mls, "(void)")

        delta = 1
        # check if values are almost equal
        self.assertAlmostEqual(solid_mls, declared_solid_mls, None, message,
                               delta)
        self.assertAlmostEqual(void_mls, declared_void_mls, None, message,
                               delta)
        self.assertAlmostEqual(dual_mls,
                               min(declared_solid_mls, declared_void_mls),
                               None, message, delta)
        self.assertEqual(dual_mls, min(solid_mls, void_mls))


if __name__ == "__main__":
    unittest.main()