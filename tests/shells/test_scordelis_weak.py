"""Loaded curved-shell validation against strong diaphragms and spatial rotation."""
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
import numpy as np


class ScordelisWeakTests(unittest.TestCase):
    def test_equal_angles_about_different_axes(self):
        from tests.support.scordelis_reference import run_formulation
        for formulation in ('RM6', 'RM5', 'RM5_v2', 'RM5_hier', 'KL'):
            with self.subTest(formulation=formulation):
                with contextlib.redirect_stdout(io.StringIO()):
                    rows = run_formulation(formulation, show=False, refinements=0,
                        make_plots=False, method_names=['nitsche'], rotation_angles=[.6, .6],
                        rotation_axes=[(1., 2., 3.), (-2., 1., .7)])
                self.assertEqual([row['orientation'] for row in rows], [0, 1])
                self.assertLess(rows[1]['rotation_difference'], 1e-7)
                self.assertLess(rows[1]['coefficient_rotation_difference'], 1e-7)

    def test_general_rotation_axes(self):
        from tests.support.scordelis_reference import rotation_matrix
        for axis in ((1., 2., 3.), (-2., 1., .7), (.3, -1., 2.)):
            rotation = rotation_matrix(.8, axis)
            np.testing.assert_allclose(rotation.T@rotation, np.eye(3), atol=5e-16)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.)
            self.assertTrue(np.all(np.abs(rotation) > 1e-3))
        for axis in ((0., 0., 0.), (1., 2.), (np.nan, 1., 2.)):
            with self.assertRaises(ValueError):
                rotation_matrix(.8, axis)

    def test_gradient_roof_trimmed_thb_lr(self):
        from tests.support.scordelis_reference import run_formulation
        for formulation in ('KL','RM5_hier'):
            for family in ('THB','LR'):
                with self.subTest(formulation=formulation, family=family):
                    with contextlib.redirect_stdout(io.StringIO()):
                        rows = run_formulation(formulation, family=family, show=False, hole=True,
                            refinements=1, make_plots=False, rotation_angles=[0.])
                    for method in ('penalty','nitsche'):
                        selected = [r for r in rows if r['method'] == method]
                        self.assertLess(selected[1]['error'], selected[0]['error'])
                        self.assertLess(selected[1]['trace'], selected[0]['trace'])
                        self.assertLess(selected[1]['error'], 1e-3)
                        self.assertTrue(np.isnan(selected[1]['benchmark_error']))

    def test_loaded_roof_comparison_and_rotation(self):
        path = Path(__file__).resolve().parents[1]/'support/scordelis_reference.py'
        spec = importlib.util.spec_from_file_location('scordelis_weak_study', path)
        study = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(study)
        for formulation in ('RM6', 'RM5', 'RM5_v2'):
            with self.subTest(formulation=formulation):
                with contextlib.redirect_stdout(io.StringIO()):
                    rows = study.run_formulation(formulation, show=False, refinements=1, make_plots=False,
                        method_names=['strong', 'penalty', 'nitsche', 'penalty_free'],
                        rotation_angles=[0., .6, 1.1],
                        rotation_axes=[(1., 2., 3.), (1., 2., 3.), (-2., 1., .7)])
                for method in ('penalty', 'nitsche', 'penalty_free'):
                    values = [row for row in rows if row['angle'] == 0. and row['method'] == method]
                    self.assertLess(values[1]['error'], values[0]['error'])
                    self.assertLess(values[1]['trace'], values[0]['trace'])
                    self.assertLess(values[1]['error'], .01 if method == 'penalty_free' else 1e-3)
                for row in rows:
                    self.assertLess(row['rotation_difference'], 1e-7)
                    self.assertLess(row['coefficient_rotation_difference'], 1e-7)
                    self.assertLess(row['residual'], 1e-7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
