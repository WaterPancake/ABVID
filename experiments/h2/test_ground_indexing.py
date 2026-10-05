"""Ground-angle regression tests; analytical filters and tiny artificial sequences only."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import erfcx

HERE = Path(__file__).resolve().parent


def load_package(root, alias):
    """Load the exact requested source under a private name; no installed fallback."""
    spec = importlib.util.spec_from_file_location(alias, root/"__init__.py",
                                                 submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


# Keep the indexing-only regression reproducible as the working renderer evolves.
# In particular, its direct-path parity assertion must not silently include later
# interpolation or propagation corrections. Test the current table separately.
INDEXING_BACKEND = HERE/"backend_revisions/indexing_20261004_v1"
PATCHED = load_package(INDEXING_BACKEND/"pyroadacoustics", "_abvid_ground_patch")
WORKING = load_package(HERE/"backend/pyroadacoustics", "_abvid_ground_working")
REFERENCE = load_package(HERE/"evidence/pyroadacoustics/pyroadacoustics", "_abvid_ground_reference")


def manager(package, material=20000):
    env = package.Environment(fs=8000, temperature=20, pressure=1, rel_humidity=50,
                              road_material=material)
    obj = package.SimulatorManager(env.c, env.fs, env.Z0, env.road_material,
                                   env.air_absorption_coefficients)
    return obj


def analytical_row(obj, angle):
    """Direct evaluation at a physical angle; never indexes a cached coefficient table."""
    theta = math.radians(angle)
    inverse_z = 1/obj.Z
    w = 1 + math.cos(theta)*inverse_z - math.sin(theta)*np.sqrt(1-inverse_z**2)
    z_cos = obj.Z*math.cos(theta)
    reflection = (z_cos-1)/(z_cos+1)
    return w, reflection


def analytical_filter(obj, angle, distance, sound_speed=343.):
    w0, reflection = analytical_row(obj, angle)
    # Default to the historical indexing-only revision's 343 m/s convention.
    # The current renderer uses its environment's c, supplied explicitly below.
    w = np.sqrt((-2j*np.pi/sound_speed)*obj.f_tmp*distance*w0)
    boundary = 1-1j*np.sqrt(np.pi)*w*erfcx(1j*w)
    spectrum = reflection + (1-reflection)*boundary
    return np.fft.irfft(spectrum)[:40]


class GroundIndexingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patched = manager(PATCHED)
        cls.reference = manager(REFERENCE)

    def test_every_angle_keeps_a_complex_frequency_row(self):
        obj = self.patched
        for table in [obj.w_tmp, obj.Rp]:
            self.assertEqual(table.shape, (179, 512))
            self.assertEqual(table.dtype, np.dtype("complex128"))
            self.assertTrue(np.isfinite(table).all())
            self.assertTrue(np.any(table.imag != 0))
        # Normal incidence is a particularly simple independent limiting case.
        np.testing.assert_allclose(obj.w_tmp[89], 1+1/obj.Z, rtol=2e-14, atol=1e-15)
        np.testing.assert_allclose(obj.Rp[89], (obj.Z-1)/(obj.Z+1), rtol=2e-14, atol=1e-15)

    def test_all_rows_match_their_own_angles(self):
        obj = self.patched
        for index, angle in enumerate(obj._theta_vector):
            with self.subTest(angle=int(angle)):
                w, reflection = analytical_row(obj, angle)
                np.testing.assert_allclose(obj.w_tmp[index], w, rtol=2e-14, atol=1e-15)
                np.testing.assert_allclose(obj.Rp[index], reflection, rtol=2e-14, atol=1e-15)

    def test_complete_filter_matches_uncached_angle_equations(self):
        obj = self.patched
        for angle in [-89, -60, -30, 0, 30, 60, 85, 89]:
            for distance in [5., 20., 50., 135.]:
                with self.subTest(angle=angle, distance=distance):
                    result = obj._get_asphalt_reflection_filter(angle, distance)
                    self.assertEqual(result.shape, (40,))
                    self.assertTrue(np.isfinite(result).all())
                    np.testing.assert_allclose(result, analytical_filter(obj, angle, distance),
                                               rtol=1e-11, atol=1e-12)

    def test_angle_rounding_and_grazing_clamping(self):
        obj = self.patched
        for requested, selected in [(0, 0), (30.49, 30), (30.51, 31), (-30.49, -30),
                                    (-30.51, -31), (89, 89), (90, 89), (95, 89),
                                    (-89, -89), (-90, -89), (-95, -89)]:
            with self.subTest(requested=requested):
                # Recomputed complex filters can differ at floating roundoff even
                # for the same selected row; table selection must be numerically equal.
                np.testing.assert_allclose(obj._get_asphalt_reflection_filter(requested, 20),
                                           obj._get_asphalt_reflection_filter(selected, 20),
                                           rtol=1e-13, atol=1e-14)

    def test_unselected_angle_rows_cannot_affect_selected_filter(self):
        original = self.patched
        # A light fixture shares only read-only attributes; mutate copies of both tables.
        obj = PATCHED.SimulatorManager.__new__(PATCHED.SimulatorManager)
        obj.__dict__ = original.__dict__.copy()
        obj.w_tmp = original.w_tmp.copy(); obj.Rp = original.Rp.copy()
        expected = obj._get_asphalt_reflection_filter(30, 20).copy()
        mask = obj._theta_vector != 30
        obj.w_tmp[mask] = np.nan + 1j*np.nan
        obj.Rp[mask] = np.nan + 1j*np.nan
        np.testing.assert_allclose(obj._get_asphalt_reflection_filter(30, 20), expected,
                                   rtol=1e-13, atol=1e-14)

    def test_selected_frequency_bins_retain_their_values(self):
        original = self.patched
        obj = PATCHED.SimulatorManager.__new__(PATCHED.SimulatorManager)
        obj.__dict__ = original.__dict__.copy()
        obj.w_tmp = original.w_tmp.copy(); obj.Rp = original.Rp.copy()
        before = obj._get_asphalt_reflection_filter(30, 20).copy()
        # Change a frequency well away from the angle-index integer (119).
        obj.Rp[119, 400] += .25 + .1j
        after = obj._get_asphalt_reflection_filter(30, 20)
        self.assertGreater(float(np.max(np.abs(after-before))), 1e-6)

    def test_original_defect_is_detected(self):
        obj = self.reference
        self.assertEqual(obj.w_tmp.shape, (512,))
        self.assertEqual(obj.Rp.shape, (512,))
        # The old loop retains the final angle and fails the normal-incidence law.
        np.testing.assert_allclose(obj.Rp, analytical_row(obj, 89)[1], rtol=2e-14, atol=1e-15)
        self.assertGreater(float(np.max(np.abs(obj.Rp-analytical_row(obj, 0)[1]))), .1)
        old = obj._get_asphalt_reflection_filter(30, 20)
        correct = analytical_filter(obj, 30, 20)
        self.assertGreater(float(np.max(np.abs(old-correct))), 1e-3)

    def test_material_object_branch_unchanged(self):
        new = manager(PATCHED, PATCHED.Material("average_asphalt"))
        old = manager(REFERENCE, REFERENCE.Material("average_asphalt"))
        for angle in [-89, -30, 0, 30, 85, 89, 90]:
            with self.subTest(angle=angle):
                np.testing.assert_array_equal(new._get_asphalt_reflection_filter(angle, 20),
                                               old._get_asphalt_reflection_filter(angle, 20))

    def test_direct_only_output_unchanged(self):
        n = np.arange(512)
        trajectory = np.column_stack([np.full(512, 5.), n/8000*10, np.full(512, .5)])
        mic = np.array([0., 0., 1.2])
        artificial = .1*np.sin(2*np.pi*500*n/8000)
        outputs = []
        for package in [REFERENCE, PATCHED]:
            obj = manager(package)
            obj.simulation_params = {"interp_method": "Sinc", "include_reflected_path": False,
                                     "include_air_absorption": False}
            obj.initialize(trajectory, mic, 0., "omnidirectional", 0., "omnidirectional")
            outputs.append(np.array([obj.update(p, mic, x) for p, x in zip(trajectory, artificial)]))
        self.assertGreater(float(np.max(np.abs(outputs[0]))), 0)
        np.testing.assert_array_equal(outputs[0], outputs[1])


class WorkingGroundIndexingTests(unittest.TestCase):
    """Retained indexing contract in the unfinished propagation working copy."""

    @classmethod
    def setUpClass(cls):
        cls.working = manager(WORKING)

    def test_current_tables_preserve_all_angle_frequency_pairs(self):
        obj = self.working
        for table in (obj.w_tmp, obj.Rp):
            self.assertEqual(table.shape, (179, 512))
            self.assertEqual(table.dtype, np.dtype("complex128"))
            self.assertTrue(np.isfinite(table).all())
        for index, angle in enumerate(obj._theta_vector):
            with self.subTest(angle=int(angle)):
                w, reflection = analytical_row(obj, angle)
                np.testing.assert_allclose(obj.w_tmp[index], w, rtol=2e-14, atol=1e-15)
                np.testing.assert_allclose(obj.Rp[index], reflection, rtol=2e-14, atol=1e-15)

    def test_current_filters_use_matching_rows_and_environment_sound_speed(self):
        obj = self.working
        for angle in [-89, -60, -30, 0, 30, 60, 85, 89]:
            for distance in [5., 20., 50., 135.]:
                with self.subTest(angle=angle, distance=distance):
                    np.testing.assert_allclose(
                        obj._get_asphalt_reflection_filter(angle, distance),
                        analytical_filter(obj, angle, distance, sound_speed=obj.c),
                        rtol=1e-11, atol=1e-12)

    def test_current_grazing_filter_does_not_read_other_rows(self):
        original = self.working
        obj = WORKING.SimulatorManager.__new__(WORKING.SimulatorManager)
        obj.__dict__ = original.__dict__.copy()
        obj.w_tmp = original.w_tmp.copy(); obj.Rp = original.Rp.copy()
        expected = analytical_filter(obj, 89, 50., sound_speed=obj.c)
        obj.w_tmp[:-1] = np.nan + 1j*np.nan
        obj.Rp[:-1] = np.nan + 1j*np.nan
        for angle in [89, 90, 95]:
            with self.subTest(angle=angle):
                np.testing.assert_allclose(obj._get_asphalt_reflection_filter(angle, 50.),
                                           expected, rtol=1e-11, atol=1e-12)


if __name__ == "__main__": unittest.main()
