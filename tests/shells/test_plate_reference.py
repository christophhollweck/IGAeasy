"""Independent scaling and truncation checks for the Navier reference."""
import unittest
import numpy as np

from IGAeasy.examples.shells.benchmarks.simply_supported_plate.reference import (
    navier_center, navier_coefficients, navier_solution,
)


class PlateReferenceTests(unittest.TestCase):
    def test_kl_unit_rigidity_center(self):
        # Unit flexural rigidity gives the classical square-plate coefficient.
        value = navier_center(include_shear=False, E=12., nue=0.,
                              thickness=1., pressure=1.)
        self.assertAlmostEqual(value, .00406235266, delta=5e-12)

    def test_load_modulus_and_thickness_scaling(self):
        for shear in (False, True):
            base = navier_center(include_shear=shear)
            self.assertAlmostEqual(navier_center(include_shear=shear, pressure=-2.), 2*base)
            self.assertAlmostEqual(navier_center(include_shear=shear, E=21840.), base/2)

        bending = navier_center(include_shear=False)
        shear_part = navier_center()-bending
        self.assertAlmostEqual(navier_center(thickness=.2), bending/8+shear_part/2)

    def test_series_truncation_at_center_and_interior_points(self):
        # Check away from symmetry axes too; a center-only check can hide errors.
        for shear in (False, True):
            for x, y in ((.5, .5), (.173, .319), (.41, .73)):
                coarse = navier_solution(x, y, shear, n_terms=200)
                fine = navier_solution(x, y, shear, n_terms=400)
                self.assertLess(abs(coarse-fine), 1e-10)
            self.assertAlmostEqual(navier_solution(.5, .5, shear),
                                   navier_center(include_shear=shear), delta=1e-15)

    def test_uniform_load_fourier_amplitude(self):
        modes, coefficients = navier_coefficients(1, False, E=12., nue=0.,
                                                  thickness=1., pressure=1.)
        np.testing.assert_array_equal(modes, [1.])
        # First sine load coefficient 16/pi^2 divided by (2*pi^2)^2.
        self.assertAlmostEqual(coefficients[0, 0], 4/np.pi**6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
