"""Regression tests for the THB and LR spline core."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import BSpline as SciPyBasis

from IGAeasy.geometry import get_geometry
from IGAeasy.splines.LRSplines.b_spline import BSpline as LRBasis
from IGAeasy.splines.LRSplines.b_spline import _evaluate_univariate_b_spline as lr_eval
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.splines.LRSplines.meshline import Meshline
from IGAeasy.splines.THBSplines.b_spline import BSpline as THBBasis
from IGAeasy.splines.THBSplines.b_spline import _evaluate_univariate_b_spline as thb_eval
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.univariate import _evaluate_univariate_b_spline as shared_eval


def basis(kind, knots):
    if kind == 'THB':
        return THBBasis(2, 0, 0, knots, [0, 0, 1, 1], [0, 0, 0])
    return LRBasis(2, knots, [0, 0, 1, 1], [0, 0, 0])


def thb_tensor_coefficients(space, field_names=('cp', 'displacements', 'scalar_value')):
    """Independent test expansion of the actual truncated basis functions.

    Expand each active function separately. Truncated nodes follow only
    inactive children; ordinary nodes use full tensor-product prolongation.
    This does not call the production control-point transfer routine.
    """
    finest = space.nlevels - 1
    cache = {}

    def expand(b):
        key = id(b)
        if key in cache:
            return cache[key]
        if b.level == finest:
            terms = {b.loc_id: 1.0}
        else:
            terms = {}
            if b.trunc:
                children = [(child, c) for child, c in zip(b.children, b.coefficient) if not child.active]
            else:
                # Full prolongation of an ordinary function must not use the
                # truncation state of its descendants.
                children = None
                terms = expand_ordinary(b.level, b.loc_id)
            if children is not None:
                for child, coefficient in children:
                    for index, value in expand(child).items():
                        terms[index] = terms.get(index, 0.0) + coefficient * value
        cache[key] = terms
        return terms

    def expand_ordinary(level, index):
        if level == finest:
            return {index: 1.0}
        matrix = space.consecutive_matrix[level]
        terms = {}
        for child, coefficient in zip(matrix.rows[index], matrix.data[index]):
            for target, value in expand_ordinary(level + 1, child).items():
                terms[target] = terms.get(target, 0.0) + coefficient * value
        return terms

    width = sum(np.asarray(getattr(space.basis[0], field)).size for field in field_names)
    result = np.zeros((len(space.spaces[finest].basis), width))
    for b in space.basis:
        fields = np.concatenate([np.asarray(getattr(b, field)).reshape(-1) for field in field_names])
        for index, coefficient in expand(b).items():
            result[index] += coefficient * fields
    return result


def lr_snapshot(space):
    return [(b.knots_u.copy(), b.knots_v.copy(), b.weight,
             np.r_[b.cp, b.displacements, b.scalar_value]) for b in space.basis]


def lr_common_knots(records, axis, degree):
    multiplicities = {}
    for record in records:
        knots, counts = np.unique(record[axis], return_counts=True)
        for knot, count in zip(knots, counts):
            multiplicities[knot] = max(multiplicities.get(knot, 0), int(count))
    return np.array([k for k in sorted(multiplicities) for _ in range(multiplicities[k])])


def lr_tensor_coefficients(records, ku, kv, degree=2):
    """Expand LR functions with SciPy knot insertion, independently of LR splitting."""
    def univariate_coefficients(local, target):
        # Extend by fully repeated exterior knots; the only nonzero coefficient
        # selects exactly the polynomial B-spline with the supplied local knots.
        extent = target[-1] - target[0]
        padded = np.r_[np.repeat(target[0] - extent, degree + 1), local,
                       np.repeat(target[-1] + extent, degree + 1)]
        c = np.zeros(len(padded) - degree - 1)
        c[degree + 1] = 1
        spline = SciPyBasis(padded, c, degree)
        for knot, count in zip(*np.unique(target, return_counts=True)):
            missing = int(count - np.count_nonzero(spline.t == knot))
            if missing > 0:
                spline = spline.insert_knot(knot, missing)
        indices = {tuple(target[i:i + degree + 2]): i for i in range(len(target) - degree - 1)}
        result = np.zeros(len(indices))
        for i in range(len(spline.t) - degree - 1):
            if spline.c[i] != 0:
                result[indices[tuple(spline.t[i:i + degree + 2])]] += spline.c[i]
        return result

    result = np.zeros(((len(ku) - degree - 1) * (len(kv) - degree - 1), 7))
    for u, v, weight, fields in records:
        coefficients = np.kron(univariate_coefficients(v, kv), univariate_coefficients(u, ku))
        result += weight * coefficients[:, None] * fields
    return result


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.previous_size = shared_eval.maxsize
        shared_eval.cache_clear()
        shared_eval.set_cache_maxsize(2)

    def tearDown(self):
        shared_eval.cache_clear()
        shared_eval.set_cache_maxsize(self.previous_size)

    def evaluate(self, x):
        return shared_eval(x, [0, 0, .5, 1], 2)

    def test_lru_evicts_least_recently_used_entry(self):
        self.evaluate(.1)
        self.evaluate(.2)
        self.evaluate(.1)
        self.evaluate(.3)
        self.evaluate(.1)
        info = shared_eval.cache_info()
        self.assertEqual(info['hits'], 2)
        self.assertEqual(info['currsize'], 2)
        self.assertEqual(info['evictions'], 1)
        self.evaluate(.2)
        self.assertEqual(shared_eval.cache_info()['misses'], 4)

    def test_eviction_preserves_results(self):
        for x in np.linspace(.51, .99, 20):
            self.assertAlmostEqual(self.evaluate(float(x)), 2*(1-x)**2)
        self.assertEqual(shared_eval.cache_info()['currsize'], 2)

    def test_resize_disable_and_clear(self):
        self.evaluate(.1)
        self.evaluate(.2)
        shared_eval.set_cache_maxsize(1)
        self.assertEqual(shared_eval.cache_info()['currsize'], 1)
        shared_eval.set_cache_maxsize(0)
        self.evaluate(.2)
        self.assertEqual(shared_eval.cache_info()['currsize'], 0)
        shared_eval.cache_clear()
        self.assertEqual(shared_eval.cache_info()['misses'], 0)
        self.assertEqual(shared_eval.cache_info()['hits'], 0)
        self.assertEqual(shared_eval.cache_info()['evictions'], 0)
        for invalid in (-1, 1.5, None, True):
            with self.assertRaises(ValueError):
                shared_eval.set_cache_maxsize(invalid)


class BasisTests(unittest.TestCase):
    def test_degrees_and_multiplicities_against_scipy(self):
        for p in range(6):
            for multiplicity in range(1, p + 2):
                knots = np.r_[np.zeros(p + 1), np.repeat(.3, multiplicity), .7, np.ones(p + 1)]
                n = len(knots) - p - 1
                reference = SciPyBasis(knots, np.eye(n), p, extrapolate=False)
                for x in (.15, .5, .85, 0, 1):
                    for r in range(min(p, 2) + 1):
                        values = [thb_eval(x, knots[i:i+p+2], p, endpoint=knots[i+p+1] == 1, r=r)
                                  for i in range(n)]
                        with self.subTest(p=p, multiplicity=multiplicity, x=x, r=r):
                            np.testing.assert_allclose(values, reference(x, nu=r), rtol=2e-12, atol=2e-11)

    def test_tensor_derivatives_against_scipy(self):
        for p in range(2, 6):
            ku = np.linspace(0, 1, p + 2)**1.3
            kv = np.linspace(0, 1, p + 2)**1.7
            ru, rv = SciPyBasis.basis_element(ku), SciPyBasis.basis_element(kv)
            for b in (THBBasis(p, 0, 0, ku, kv, [0, 0, 0]), LRBasis(p, ku, kv, [0, 0, 0])):
                u, v = .41, .57
                np.testing.assert_allclose(b.grad(u, v), [ru(u, nu=1)*rv(v), ru(u)*rv(v, nu=1)], atol=2e-11)
                np.testing.assert_allclose(b.grad2(u, v), [ru(u, nu=2)*rv(v), ru(u)*rv(v, nu=2)], atol=2e-11)
                self.assertAlmostEqual(b.grad_mix(u, v), ru(u, nu=1)*rv(v, nu=1), places=11)
    def test_interior_support_end_uses_right_hand_derivative(self):
        for evaluator in (thb_eval, lr_eval):
            self.assertEqual(evaluator(.5, [0, 0, .5], 1, r=1), 0)

    def test_cell_selects_one_sided_derivative(self):
        left = SimpleNamespace(u_min=0, u_max=.5, v_min=0, v_max=1)
        right = SimpleNamespace(u_min=.5, u_max=1, v_min=0, v_max=1)
        functions = (
            THBBasis(1, 0, 0, [0, .5, 1], [0, 0, 1], [0, 0, 0]),
            LRBasis(1, [0, .5, 1], [0, 0, 1], [0, 0, 0]),
        )
        for b in functions:
            self.assertAlmostEqual(b(.5, .25, r1=1), -1.5)
            self.assertAlmostEqual(b.grad(.5, .25, cell=left)[0], 1.5)
            self.assertAlmostEqual(b.grad(.5, .25, cell=right)[0], -1.5)

    def test_patch_end_value_and_left_derivative(self):
        thb = THBBasis(1, 0, 0, [0, 1, 1], [0, 1, 1], [0, 0, 0], end_u=True, end_v=True)
        lr = LRBasis(1, [0, 1, 1], [0, 1, 1], [0, 0, 0])
        for b in (thb, lr):
            self.assertEqual(b(1, 1), 1)
            np.testing.assert_allclose(b.grad(1, 1), [1, 1])
            self.assertEqual(b(1.01, 1), 0)

    def test_repeated_interior_knot(self):
        for evaluator in (thb_eval, lr_eval):
            self.assertAlmostEqual(evaluator(.5, [0, .5, .5, 1], 2), 1)
            self.assertAlmostEqual(evaluator(.5, [0, .5, .5, 1], 2, r=1), -4)

    def test_identity_and_hash_contract(self):
        for kind in ('THB', 'LR'):
            a = basis(kind, [0, 0, .5, 1])
            identical = basis(kind, [0, 0, .5, 1])
            nearby = basis(kind, [0, 0, .500000001, 1])
            identical.cp[:] = 7
            self.assertEqual(a, identical)
            self.assertEqual(hash(a), hash(identical))
            self.assertNotEqual(a, nearby)
            self.assertEqual(len({a, identical, nearby}), 2)
            with self.assertRaises(ValueError):
                a.knots_u[2] = .6

    def test_lr_weight_merge_preserves_weighted_geometry(self):
        a = LRBasis(2, [0, 0, .5, 1], [0, 0, 1, 1], [1, 2, 3], weight=.25)
        b = LRBasis(2, [0, 0, .5, 1], [0, 0, 1, 1], [4, 5, 6], weight=.75)
        before = a.cp * a(.25, .4) + b.cp * b(.25, .4)
        a.update_weights(b)
        self.assertEqual(a.weight, 1)
        np.testing.assert_allclose(a.cp * a(.25, .4), before)

    def test_quadratic_values_and_derivatives(self):
        # On (0.5, 1), this basis is 2*(1-x)**2.
        for evaluator in (thb_eval, lr_eval):
            for r, expected in enumerate((0.125, -1.0, 4.0)):
                with self.subTest(evaluator=evaluator, derivative=r):
                    self.assertAlmostEqual(evaluator(.75, [0, 0, .5, 1], 2, r=r), expected)

    def test_derivative_above_degree_is_zero(self):
        for evaluator in (thb_eval, lr_eval):
            self.assertEqual(evaluator(.25, [0, 0, 1], 1, r=2), 0)

    def test_greville_uses_inner_knots(self):
        for kind in ('THB', 'LR'):
            self.assertEqual(basis(kind, [0, 0, .5, 1]).knot_average, (.25, .5))

    def test_invalid_derivative_order_even_after_cache_hit(self):
        for evaluator in (thb_eval, lr_eval):
            evaluator(.25, [0, 0, 1], 1, r=0)
            for r in (-1, .5):
                with self.subTest(derivative=r), self.assertRaises(ValueError):
                    evaluator(.25, [0, 0, 1], 1, r=r)

    def test_nonuniform_knots_against_independent_reference(self):
        for offset, scale in ((0, 1), (-2, 3)):
            knots = offset + scale * np.array([0, .1, .4, 1])
            reference = SciPyBasis.basis_element(knots, extrapolate=False)
            for evaluator in (thb_eval, lr_eval):
                for r in range(3):
                    for x in offset + scale * np.array([.05, .2, .7]):
                        with self.subTest(offset=offset, scale=scale, r=r, x=x):
                            self.assertAlmostEqual(evaluator(x, knots, 2, r=r), reference(x, nu=r), places=12)


class SpaceTests(unittest.TestCase):
    def test_multilevel_truncated_cell_derivatives_match_tensor_reference(self):
        space = self.make_space('h', levels=4)
        for level in (0, 1, 2):
            space.refine({level: [min(space.afunc_level[level])]})
        coefficients = thb_tensor_coefficients(space)[:, :3]
        fine = space.spaces[-1]
        nu, nv = len(fine.ku)-fine.p-1, len(fine.kv)-fine.p-1
        bu, bv = SciPyBasis(fine.ku, np.eye(nu), fine.p), SciPyBasis(fine.kv, np.eye(nv), fine.p)
        self.assertTrue(any(b.trunc for b in space.basis))
        for cell in space.elements:
            for u, v in ((cell.u_min, cell.v_min), (cell.u_max, cell.v_max), cell.center):
                x, y = np.nextafter(u, cell.center[0]), np.nextafter(v, cell.center[1])
                for r1, r2 in ((0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)):
                    value = sum((b.cp*b(u, v, r1=r1, r2=r2, cell=cell) for b in cell.supported_b_splines))
                    reference = np.kron(bv(y, nu=r2), bu(x, nu=r1)) @ coefficients
                    np.testing.assert_allclose(value, reference, rtol=0, atol=2e-9)
    def test_thb_overlapping_markings_are_order_independent(self):
        for kind in ('h', 'p', 'k'):
            results = []
            for indices in ([0, 1, 6], [6, 1, 0]):
                space = self.make_space(kind)
                space.add_level(3)
                before = thb_tensor_coefficients(space)
                space.refine({0: indices})
                np.testing.assert_allclose(thb_tensor_coefficients(space), before, rtol=0, atol=2e-12)
                self.check_invariants(space)
                self.check_support_lists(space)
                results.append(space)
            self.assertEqual(results[0].afunc_level, results[1].afunc_level)
            self.assertEqual(results[0].aelem_level, results[1].aelem_level)

    def test_thb_multilevel_markings_preserve_fields_in_either_order(self):
        for kind in ('h', 'p', 'k'):
            for nonuniform in (False, True):
                results = []
                for order in ((0, 1), (1, 0)):
                    with self.subTest(kind=kind, nonuniform=nonuniform, order=order):
                        space = self.make_nonuniform_space(kind) if nonuniform else self.make_space(kind)
                        space.refine({0: [0]})
                        space.add_level(3)
                        rng = np.random.default_rng(765)
                        for b in space.basis:
                            b.cp += rng.normal(0, .07, 3)
                            b.displacements = rng.normal(size=3)
                            b.scalar_value = rng.normal()
                        before = thb_tensor_coefficients(space)
                        marked = {level: [min(space.afunc_level[level])] for level in order}
                        space.refine(marked)
                        np.testing.assert_allclose(thb_tensor_coefficients(space), before, rtol=0, atol=2e-12)
                        self.check_support_lists(space)
                        for u, v in ((.05, .05), (.23, .4), (.8, .9)):
                            self.assertAlmostEqual(sum(b(u, v) for b in space.basis), 1, places=12)
                        results.append(space)
                if len(results) == 2:
                    self.assertEqual(results[0].afunc_level, results[1].afunc_level)
                    self.assertEqual(results[0].aelem_level, results[1].aelem_level)

    def test_inconsistent_field_shapes_rejected_before_refinement(self):
        for kind in ('h', 'LR'):
            space = self.make_space(kind)
            space.basis[0].dof_values = np.ones(3)
            space.basis[1].dof_values = np.ones(6)
            before = list(space.basis)
            with self.assertRaises(ValueError):
                if kind == 'LR':
                    space.insert_line(Meshline(0, 1, .23, 0))
                else:
                    space.refine({0: [0]})
            self.assertEqual(space.basis, before)
    def test_truncated_parent_transfers_custom_fields_to_children(self):
        rng = np.random.default_rng(987)
        space = self.make_space('h')
        space.refine({0: [0]})
        candidates = [b for b in space.basis if b.trunc and any(c.active for c in b.children)
                      and any(not c.active for c in b.children)]
        self.assertTrue(candidates, 'Test requires a truncated parent with both active and inactive children.')
        parent = candidates[0]
        space.add_level(3)
        for b in space.basis:
            b.cp += rng.normal(0, .1, 3)
            b.dof_values = rng.normal(size=6)
            b.temperature = rng.normal()
            b.custom_tensor = rng.normal(size=(2, 2))
        names = ('cp', 'dof_values', 'temperature', 'custom_tensor')
        before = thb_tensor_coefficients(space, names)
        old_children = {id(c): c.dof_values.copy() for c in parent.children if c.active}
        space.refine({parent.level: [parent.loc_id]})
        self.assertFalse(parent.active)
        np.testing.assert_allclose(thb_tensor_coefficients(space, names), before, rtol=0, atol=2e-12)
        for child in parent.children:
            if child.active and id(child) in old_children:
                np.testing.assert_array_equal(child.dof_values, old_children[id(child)])
        self.check_support_lists(space)

    def check_support_lists(self, space):
        for b in space.basis:
            self.assertFalse(hasattr(b, 'elements_of_support_after_ref'))
            self.assertEqual(set(b.elements_of_support),
                             {e for e in space.elements if b in e.supported_b_splines})

    def test_lr_transfers_custom_vector_and_tensor_fields(self):
        rng = np.random.default_rng(654)
        space = self.make_space('LR')
        points = rng.uniform(.01, .99, (30, 2))
        for b in space.basis:
            b.dof_values = rng.normal(size=6)
            b.custom_tensor = rng.normal(size=(2, 2))
        def evaluate(field):
            return np.array([sum((getattr(b, field) * b(u, v) for b in space.basis)) for u, v in points])
        before = {field: evaluate(field) for field in ('dof_values', 'custom_tensor')}
        space.refine_by_structured_mesh([space.basis[0]])
        space.get_connectivity()
        for field, values in before.items():
            np.testing.assert_allclose(evaluate(field), values, rtol=0, atol=2e-12)
        self.check_support_lists(space)
    def test_lr_all_tensor_coefficients_survive_deformation_and_refinement(self):
        rng = np.random.default_rng(456)
        for nonuniform in (False, True):
            with self.subTest(nonuniform=nonuniform):
                space = self.make_nonuniform_space('LR') if nonuniform else self.make_space('LR')
                space.refine_by_structured_mesh([space.basis[0]])
                space.get_connectivity()
                for step in range(2):
                    for b in space.basis:
                        b.cp += rng.normal(0, .07, 3)
                        b.displacements = rng.normal(0, .1, 3)
                        b.scalar_value = rng.normal()
                    before = lr_snapshot(space)
                    space.refine_by_structured_mesh([space.basis[len(space.basis) // 2]])
                    space.get_connectivity()
                    after = lr_snapshot(space)
                    ku = lr_common_knots(before + after, 0, 2)
                    kv = lr_common_knots(before + after, 1, 2)
                    c_before = lr_tensor_coefficients(before, ku, kv)
                    c_after = lr_tensor_coefficients(after, ku, kv)
                    np.testing.assert_allclose(c_after, c_before, rtol=0, atol=2e-12)
                    nu, nv = len(ku) - 3, len(kv) - 3
                    bu, bv = SciPyBasis(ku, np.eye(nu), 2), SciPyBasis(kv, np.eye(nv), 2)
                    for u, v in rng.uniform(.01, .99, (8, 2)):
                        np.testing.assert_allclose(np.kron(bv(v), bu(u)) @ c_after[:, :3],
                                                   space(u, v), rtol=0, atol=2e-12)

    def test_thb_all_tensor_coefficients_survive_deformation_and_refinement(self):
        rng = np.random.default_rng(123)
        for kind in ('h', 'p', 'k'):
            for nonuniform in (False, True):
                with self.subTest(kind=kind, nonuniform=nonuniform):
                    space = self.make_nonuniform_space(kind) if nonuniform else self.make_space(kind)
                    space.refine({0: [0]})
                    # Fix one common reference space for both representations.
                    space.add_level(3)
                    for level in (1, 0):
                        for b in space.basis:
                            b.cp += rng.normal(0, .07, 3)
                            b.displacements = rng.normal(0, .1, 3)
                            b.scalar_value = rng.normal()
                        before = thb_tensor_coefficients(space)
                        space.refine({level: [min(space.afunc_level[level])]})
                        after = thb_tensor_coefficients(space)
                        np.testing.assert_allclose(after, before, rtol=0, atol=2e-12)

    def test_tensor_reference_reconstructs_geometry_with_scipy(self):
        rng = np.random.default_rng(321)
        for kind in ('h', 'p', 'k'):
            space = self.make_space(kind)
            space.refine({0: [0]})
            space.refine({1: [min(space.afunc_level[1])]})
            for b in space.basis:
                b.cp += rng.normal(0, .1, 3)
            coefficients = thb_tensor_coefficients(space)
            fine = space.spaces[-1]
            nu, nv = len(fine.ku) - fine.p - 1, len(fine.kv) - fine.p - 1
            bu = SciPyBasis(fine.ku, np.eye(nu), fine.p, extrapolate=False)
            bv = SciPyBasis(fine.kv, np.eye(nv), fine.p, extrapolate=False)
            for u, v in rng.uniform(.01, .99, (12, 2)):
                reference = np.kron(bv(v), bu(u)) @ coefficients[:, :3]
                np.testing.assert_allclose(reference, space(u, v), rtol=0, atol=2e-12)
            # Confirm that the coefficient check detects a deliberate change.
            before = coefficients.copy()
            space.basis[0].cp[2] += .01
            self.assertGreater(np.max(np.abs(thb_tensor_coefficients(space) - before)), 1e-5)
    def test_meshline_identity_does_not_merge_nearby_knots(self):
        a = Meshline(0, 1e-12, 2e-13, 0)
        b = Meshline(0, 1e-12, 3e-13, 0)
        self.assertNotEqual(a, b)
        self.assertFalse(a._similar(b))
        self.assertEqual(a._number_of_knots_contained_helper(np.array([0, 2e-13, 2e-13, 3e-13])), 2)

    def test_lr_scaled_full_refinement(self):
        for scale, offset in ((1e-12, 0), (1, -7), (1e6, 2e6)):
            with self.subTest(scale=scale, offset=offset):
                original = self.make_nonuniform_space('LR')
                space = LRSpline(2, offset + scale * original.ku, offset + scale * original.kv,
                                 np.array([b.cp for b in original.basis]))
                before = len(space.basis)
                space.refine_by_structured_mesh([space.basis[len(space.basis) // 2]])
                space.get_connectivity()
                self.assertGreater(len(space.basis), before)
                for u in np.linspace(0, 1, 7):
                    for v in np.linspace(0, 1, 7):
                        x, y = offset + scale * u, offset + scale * v
                        self.assertAlmostEqual(sum(b(x, y) for b in space.basis), 1, places=12)
                        np.testing.assert_allclose(space(x, y), [u, v, 0], atol=1e-12)

    def test_meshline_index_matches_live_lines_after_replacement(self):
        space = self.make_nonuniform_space('LR')
        for multiplicity in (1, 2):
            space.insert_line(Meshline(0, 1, .23, 0, multiplicity))
        self.assertEqual(len(space.mls_v_rtree), sum(map(len, space.meshlines_v.values())))

    def test_meshline_index_tracks_extended_bounds(self):
        space = self.make_nonuniform_space('LR')
        first = Meshline(0, .4, .23, 0, 1)
        space.merge_meshlines(first)
        space.merge_meshlines(Meshline(.3, .8, .23, 0, 2))
        self.assertEqual(first.stop, .8)
        candidates = set(space.mls_v_rtree.intersection((.23, .7, .23, .7)))
        self.assertIn(id(first), candidates)
        self.assertEqual(len(space.mls_v_rtree), sum(map(len, space.meshlines_v.values())))

    def make_nonuniform_space(self, kind):
        ku = np.array([0, 0, 0, .12, .45, .8, 1, 1, 1])
        kv = np.array([0, 0, 0, .3, .7, 1, 1, 1])
        gu = [np.mean(ku[i + 1:i + 3]) for i in range(len(ku) - 3)]
        gv = [np.mean(kv[i + 1:i + 3]) for i in range(len(kv) - 3)]
        cps = np.array([[u, v, 0] for v in gv for u in gu])
        if kind == 'LR':
            return LRSpline(2, ku, kv, cps)
        return HierarchicalSpace(ku, kv, 2, cps, 3, kind)

    def test_nonuniform_structured_refinement_preserves_geometry(self):
        for kind in ('LR', 'h', 'p', 'k'):
            with self.subTest(kind=kind):
                space = self.make_nonuniform_space(kind)
                self.check_invariants(space)
                if kind == 'LR':
                    for step in range(3):
                        space.refine_by_structured_mesh([space.basis[len(space.basis) // 2]])
                        space.get_connectivity()
                        self.check_invariants(space)
                else:
                    space.refine({0: [8]})
                    self.check_invariants(space)

    def test_lr_full_lines_are_order_independent(self):
        specifications = [(0, .23, 1), (1, .52, 1), (0, .62, 1), (0, .23, 2)]
        spaces = []
        for sequence in (specifications, list(reversed(specifications))):
            space = self.make_nonuniform_space('LR')
            for axis, value, multiplicity in sequence:
                space.insert_line(Meshline(0, 1, value, axis, multiplicity))
                space.get_connectivity()
                self.check_invariants(space)
            spaces.append(space)
        self.assertEqual(set(spaces[0].basis_dic), set(spaces[1].basis_dic))
        for key, b in spaces[0].basis_dic.items():
            other = spaces[1].basis_dic[key]
            self.assertAlmostEqual(b.weight, other.weight, places=12)
            np.testing.assert_allclose(b.cp, other.cp, atol=1e-12)

    def test_lr_spatial_element_index_has_no_removed_entries(self):
        space = self.make_nonuniform_space('LR')
        space.insert_line(Meshline(0, 1, .23, 0))
        space.get_connectivity()
        self.assertEqual(len(space.els_rtree), len(space.elements))

    def test_lr_partial_lines_preserve_geometry_in_both_orders(self):
        specifications = [(0, .23, 0, .7), (1, .52, 0, .8), (0, .62, .3, 1)]
        for sequence in (specifications, list(reversed(specifications))):
            space = self.make_nonuniform_space('LR')
            for axis, value, start, stop in sequence:
                space.insert_line(Meshline(start, stop, value, axis))
                space.get_connectivity()
                self.check_invariants(space)
                self.assertEqual(len(space.els_rtree), len(space.elements))

    def test_candidate_bounds_are_local(self):
        space = self.make_nonuniform_space('LR')
        b = LRBasis(2, [.2, .3, .4, .5], [.3, .4, .5, .6], [0, 0, 0])
        self.assertEqual(space.compute_enclosing_aabb({b.basis_key: b}), (.2, .3, .5, .6))

    def test_structured_refinement_splits_largest_spans(self):
        space = self.make_nonuniform_space('LR')
        b = LRBasis(2, [0, 1, 2, 2.5], [0, 1, 2, 2.5], [0, 0, 0])
        lines = space.get_structured_refinement_meshlines(b)
        for axis in (0, 1):
            self.assertEqual([m.constant_value for m in lines if m.axis == axis], [.5, 1.5])

    def test_structured_selection_is_scale_and_translation_invariant(self):
        space = self.make_nonuniform_space('LR')
        for scale in (1e-9, 1, 1e6):
            for offset in (0, -7 * scale):
                with self.subTest(scale=scale, offset=offset):
                    knots = offset + scale * np.array([0, 1, 2, 2.5])
                    b = LRBasis(2, knots, knots, [0, 0, 0])
                    lines = space.get_structured_refinement_meshlines(b)
                    for axis in (0, 1):
                        selected = [(m.constant_value - offset) / scale for m in lines if m.axis == axis]
                        np.testing.assert_allclose(selected, [.5, 1.5], atol=1e-12)

    def test_structured_selection_tolerates_relative_input_noise(self):
        space = self.make_nonuniform_space('LR')
        for scale in (1e-9, 1, 1e6):
            knots = scale * np.array([0, 1, 2.0000005, 2.5])
            b = LRBasis(2, knots, knots, [0, 0, 0])
            lines = space.get_structured_refinement_meshlines(b)
            with self.subTest(scale=scale):
                self.assertEqual(len(lines), 4)

    def test_structured_selection_with_large_coordinate_offset(self):
        space = self.make_nonuniform_space('LR')
        knots = 1e8 + np.array([0, .1, .2, .25])
        b = LRBasis(2, knots, knots, [0, 0, 0])
        lines = space.get_structured_refinement_meshlines(b)
        self.assertEqual(len(lines), 4)
        np.testing.assert_allclose([m.constant_value - 1e8 for m in lines if m.axis == 0], [.05, .15], atol=2e-8)

    def test_structured_span_tolerance_is_configurable(self):
        space = self.make_nonuniform_space('LR')
        b = LRBasis(2, [0, 1, 2.0000005, 2.5], [0, 1, 2.0000005, 2.5], [0, 0, 0])
        self.assertEqual(len(space.get_structured_refinement_meshlines(b, span_rtol=0)), 2)
        for invalid in (-1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                space.get_structured_refinement_meshlines(b, span_rtol=invalid)

    def test_lr_hash_collision_cannot_merge_distinct_functions(self):
        with patch.object(LRBasis, '__hash__', return_value=1):
            space = self.make_space('LR')
            self.assertEqual(len(space.basis), 20)
            space.refine_by_structured_mesh([space.basis[0]])
            space.get_connectivity()
            self.check_invariants(space)
            self.assertEqual(len(space.bfs_rtree), len(space.basis))

    def test_lr_repeated_refinement_and_spatial_lookup(self):
        space = self.make_space('LR')
        for step in range(3):
            space.refine_by_structured_mesh([space.basis[0]])
            space.get_connectivity()
            self.check_invariants(space)
            self.assertEqual(len(space.bfs_rtree), len(space.basis))
            self.assertEqual(len(space._basis_by_index), len(space.basis))
            found = space.get_funcs_in_rectangle(0, [[[0, 1], [0, 1]]])
            self.assertEqual(set(found), {b for b in space.basis if b.level == 0})

    def make_space(self, kind, levels=3):
        geometry = get_geometry.surface(3, 2, 1, 1, 1, 1, 1, 2)
        ku, kv = geometry.knot_vectors
        if kind == 'LR':
            return LRSpline(2, ku, kv, geometry.control_points)
        return HierarchicalSpace(ku, kv, 2, geometry.control_points, levels, kind)

    def check_invariants(self, space):
        for u in np.linspace(0, 1, 7):
            for v in np.linspace(0, 1, 7):
                values = [b(u, v) for b in space.basis]
                self.assertAlmostEqual(sum(values), 1, places=12)
                self.assertGreaterEqual(min(values), -1e-12)
                np.testing.assert_allclose(space(u, v), [u, v, 0], atol=1e-12)
        for element in space.elements:
            u, v = element.center
            np.testing.assert_allclose(sum((b.grad(u, v) for b in space.basis)), [0, 0], atol=1e-12)
            for b in element.supported_b_splines:
                self.assertIn(element, b.elements_of_support)

    def test_local_refinement_preserves_geometry_and_partition(self):
        for kind in ('h', 'p', 'k', 'LR'):
            with self.subTest(kind=kind):
                space = self.make_space(kind)
                self.check_invariants(space)
                if kind == 'LR':
                    space.refine_by_structured_mesh([space.basis[0]])
                    space.get_connectivity()
                else:
                    space.refine({0: [0]})
                self.check_invariants(space)

    def test_thb_repeated_refinement(self):
        space = self.make_space('h', levels=4)
        for level in (0, 1, 2):
            space.refine({level: [min(space.afunc_level[level])]})
            self.check_invariants(space)

    def test_refine_deform_refine_preserves_current_geometry(self):
        points = [(u, v) for u in np.linspace(0, 1, 17) for v in np.linspace(0, 1, 15)]
        for kind in ('h', 'p', 'k', 'LR'):
            for target in ('fine', 'coarse'):
                with self.subTest(kind=kind, target=target):
                    space = self.make_space(kind, levels=4)
                    if kind == 'LR':
                        space.refine_by_structured_mesh([space.basis[0]])
                        space.get_connectivity()
                    else:
                        space.refine({0: [0]})
                    # Nonlinear control-net deformation, including out-of-plane motion.
                    for b in space.basis:
                        x, y, z = b.cp
                        b.cp = np.array([x + .17*x*y, y + .11*x*x,
                                         z + .23*np.sin(2*x)*np.cos(3*y)])
                    before = np.array([space(u, v) for u, v in points])
                    if kind == 'LR':
                        selected = space.basis[0] if target == 'fine' else space.basis[-1]
                        space.refine_by_structured_mesh([selected])
                        space.get_connectivity()
                    else:
                        level = 1 if target == 'fine' else 0
                        space.refine({level: [min(space.afunc_level[level])]})
                    after = np.array([space(u, v) for u, v in points])
                    np.testing.assert_allclose(after, before, rtol=0, atol=2e-12)

    def test_thb_repeated_deformations_preserve_geometry_and_tangents(self):
        rng = np.random.default_rng(42)
        points = rng.uniform(.01, .99, (45, 2))
        for kind in ('h', 'p', 'k'):
            with self.subTest(kind=kind):
                space = self.make_space(kind, levels=3)
                space.refine({0: [0]})
                for level in (1, 0, 1):
                    for b in space.basis:
                        b.cp = b.cp + rng.normal(0, .04, 3)
                    before = np.array([space(u, v) for u, v in points])
                    tangents = np.array([sum((np.outer(b.cp, b.grad(u, v)) for b in space.basis))
                                         for u, v in points])
                    space.refine({level: [min(space.afunc_level[level])]})
                    np.testing.assert_allclose([space(u, v) for u, v in points], before, rtol=0, atol=2e-12)
                    after_tangents = np.array([sum((np.outer(b.cp, b.grad(u, v)) for b in space.basis))
                                               for u, v in points])
                    np.testing.assert_allclose(after_tangents, tangents, rtol=0, atol=2e-11)

    def test_refinement_preserves_displaced_geometry_and_scalar_field(self):
        from IGAeasy.postprocessing.postprocessing import update_cps
        points = [(u, v) for u in np.linspace(0, 1, 11) for v in np.linspace(0, 1, 9)]
        for kind in ('h', 'p', 'k', 'LR'):
            with self.subTest(kind=kind):
                space = self.make_space(kind)
                for step in range(2):
                    dis = np.array([[.1*b.cp[0]*b.cp[1], .2*b.cp[0]**2,
                                     .15*np.sin(b.cp[0]+b.cp[1])] for b in space.basis]).ravel()
                    update_cps(space, dis, dof_per_cp=3)
                    for b in space.basis:
                        b.scalar_value = np.cos(b.cp[0] - b.cp[1])
                    before = np.array([space(u, v) + space.eval_displacement(u, v) for u, v in points])
                    scalar = np.array([space.eval_scalar_field(u, v) for u, v in points])
                    if kind == 'LR':
                        space.refine_by_structured_mesh([space.basis[0]])
                        space.get_connectivity()
                    else:
                        space.refine({step: [min(space.afunc_level[step])]})
                    after = np.array([space(u, v) + space.eval_displacement(u, v) for u, v in points])
                    np.testing.assert_allclose(after, before, rtol=0, atol=2e-12)
                    np.testing.assert_allclose([space.eval_scalar_field(u, v) for u, v in points], scalar, rtol=0, atol=2e-12)

    def test_thb_levels_grow_on_demand(self):
        for kind in ('h', 'p', 'k'):
            with self.subTest(kind=kind):
                space = self.make_space(kind, levels=3)
                self.assertEqual(len(space.spaces), 1)
                space.refine({0: []})
                self.assertEqual(len(space.spaces), 1)
                for level in (0, 1):
                    space.refine({level: [min(space.afunc_level[level])]})
                    self.assertEqual(len(space.spaces), level + 2)
                    self.assertEqual(space.n_max_level, 3)
                    self.check_invariants(space)

    def test_thb_growth_matches_prebuilt_hierarchy(self):
        for kind in ('h', 'p', 'k'):
            lazy = self.make_space(kind, levels=3)
            prebuilt = self.make_space(kind, levels=3)
            prebuilt.add_level(3)
            for level in (0, 1):
                for space in (lazy, prebuilt):
                    space.refine({level: [min(space.afunc_level[level])]})
            self.assertEqual(lazy.afunc_level, prebuilt.afunc_level)
            for a, b in zip(lazy.basis, prebuilt.basis):
                self.assertEqual(a.basis_key, b.basis_key)
                np.testing.assert_allclose(a.cp, b.cp, atol=1e-12)

    def test_thb_connections_are_not_duplicated(self):
        space = self.make_space('h', levels=3)
        space.add_level(2)
        counts = [len(b.children) for b in space.spaces[0].basis]
        space.add_level(2)
        space.connect_levels()
        space.add_level(3)
        self.assertEqual(counts, [len(b.children) for b in space.spaces[0].basis])

    def test_thb_last_level_rejected_before_mutation(self):
        space = self.make_space('h', levels=2)
        space.refine({0: [0]})
        active_before = {level: ids.copy() for level, ids in space.afunc_level.items()}
        with self.assertRaises(ValueError):
            space.refine({0: [1], 1: [min(space.afunc_level[1])]})
        self.assertEqual(space.afunc_level, active_before)
        self.check_invariants(space)

    def test_thb_counts_follow_active_space(self):
        space = self.make_space('h')
        space.refine({0: [0]})
        self.assertEqual(space.nfuncs, len(space.basis))
        self.assertEqual(space.nel, len(space.elements))


if __name__ == '__main__':
    unittest.main(verbosity=2)
