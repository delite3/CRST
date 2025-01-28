"""
Build script for C++ and Python bindings. Requires a
C++14 compiler. Remember to initialize the submodules.

"""
# Build
import sysconfig
import subprocess

lib_name = "radar_model"
subdir = "src"
mod_name = f"{subdir}.{lib_name}"
lib_suffix = sysconfig.get_config_var("EXT_SUFFIX")
lib_code = ["radarModel.cpp", "radarModel_pybind.cpp"]
include_paths = []
include_paths.append(sysconfig.get_path("include"))
include_paths.append("extern/pybind11/include")
include_paths.append("extern/eigen")

include_args = ["-I" + path for path in include_paths]
cmd = [
    "c++",
    "-O3",
    "-Wall",
    "-shared",
    "-std=c++14",
    "-fPIC",
    *include_args,
    *lib_code,
    f"-o{lib_name}{lib_suffix}",
]
print(cmd)
subprocess.run(cmd, cwd=subdir, check=True)

# Test
from src.radar_model import BoschRadar
from radar_model import RadarModel
import numpy as np
import unittest


class TestModelMethods(unittest.TestCase):
    def setUp(self) -> None:
        snr_50_db = 3.768
        rcs_50 = 0.854
        r_50 = 70
        pf = 0.01
        scale = 0.2292
        rcs_min = -30

        self.model_cpp = BoschRadar(snr_50_db, rcs_50, r_50, pf, scale, rcs_min)
        self.model_py = RadarModel(snr_50_db, rcs_50, r_50, pf, scale, rcs_min)

        self.r_range = [0.0, 100.0]
        self.phi_range = [-np.deg2rad(45), np.deg2rad(45)]
        self.rcs_range = [0.1, 50.0]
        self.size = (300, 400)

    def test_gaussian_gain(self) -> None:
        phi_float = np.random.uniform(*self.phi_range)
        self.assertAlmostEqual(
            self.model_cpp.gaussian_gain(phi_float),
            self.model_py.gaussian_gain(phi_float),
        )

        phi_array = np.random.uniform(*self.phi_range, self.size)
        np.testing.assert_allclose(
            self.model_cpp.gaussian_gain(phi_array),
            self.model_py.gaussian_gain(phi_array),
        )
        np.testing.assert_allclose(
            self.model_cpp.gaussian_gain(phi_array.T.copy()),
            self.model_py.gaussian_gain(phi_array.T.copy()),
        )

    def test_compute_pd(self) -> None:
        pd_float = np.random.rand()
        self.assertAlmostEqual(
            self.model_cpp.compute_pd(pd_float),
            self.model_py.compute_pd(pd_float),
        )

        pd_array = np.random.rand(3, 4)
        np.testing.assert_allclose(
            self.model_cpp.compute_pd(pd_array),
            self.model_py.compute_pd(pd_array),
        )
        np.testing.assert_allclose(
            self.model_cpp.compute_pd(pd_array.T.copy()),
            self.model_py.compute_pd(pd_array.T.copy()),
        )

    def test_compute_snr(self) -> None:
        r_float = np.random.uniform(*self.r_range)
        phi_float = np.random.uniform(*self.phi_range)
        rcs_float = np.random.uniform(*self.rcs_range)
        self.assertAlmostEqual(
            self.model_cpp.compute_snr(r_float, phi_float, rcs_float),
            self.model_py.compute_snr(r_float, phi_float, rcs_float),
        )

        r_array = np.random.uniform(*self.r_range, self.size)
        phi_array = np.random.uniform(*self.phi_range, self.size)
        np.testing.assert_allclose(
            self.model_cpp.compute_snr(r_array, phi_array, rcs_float),
            self.model_py.compute_snr(r_array, phi_array, rcs_float),
        )
        np.testing.assert_allclose(
            self.model_cpp.compute_snr(r_array.T.copy(), phi_array.T.copy(), rcs_float),
            self.model_py.compute_snr(r_array.T.copy(), phi_array.T.copy(), rcs_float),
        )

    def test_sim_rcs(self) -> None:
        rcs_float = np.random.uniform(*self.rcs_range)
        r_float = rcs_float
        self.assertAlmostEqual(
            self.model_cpp.sim_rcs(r_float, rcs_float, add_noise=False),
            self.model_py.sim_rcs(r_float, rcs_float, add_noise=False),
        )

        r_float = rcs_float * 2 + 1
        self.assertAlmostEqual(
            self.model_cpp.sim_rcs(r_float, rcs_float, add_noise=False),
            self.model_py.sim_rcs(r_float, rcs_float, add_noise=False),
        )

        r_float = np.random.uniform(*self.r_range)
        delta_float = r_float * 1.5 / 10.0
        self.assertAlmostEqual(
            self.model_cpp.sim_rcs(r_float, rcs_float, add_noise=False),
            self.model_py.sim_rcs(r_float, rcs_float, add_noise=False),
            delta=delta_float,
        )

        r_array = np.full(self.size, rcs_float)
        np.testing.assert_allclose(
            self.model_cpp.sim_rcs(r_array, rcs_float, add_noise=False),
            self.model_py.sim_rcs(r_array, rcs_float, add_noise=False),
        )

        r_array = np.full(self.size, rcs_float * 2 + 1)
        np.testing.assert_allclose(
            self.model_cpp.sim_rcs(r_array, rcs_float, add_noise=False),
            self.model_py.sim_rcs(r_array, rcs_float, add_noise=False),
        )

        r_array = np.random.uniform(*self.r_range, self.size)
        delta_array = r_array * 1.5 / 10.0
        np.testing.assert_array_less(
            np.abs(
                self.model_cpp.sim_rcs(r_array, rcs_float, add_noise=True)
                - self.model_py.sim_rcs(r_array, rcs_float, add_noise=True)
            ),
            delta_array,
        )


unittest.main(exit=False)

# Stubs
from mypy import stubgen
import re

# Generate stubs
stubgen.main(["-m", mod_name, "-o", "."])
stub_path = f"./{mod_name.replace('.', '/')}.pyi"

# Fix numpy issues
with open(stub_path, "r") as stub_file:
    lines = stub_file.readlines()
    pattern = re.compile(r"(?<=),flags\..*?(?=])")
    has_numpy = False
    for i, line in enumerate(lines):
        # Pythons regex implementation doesn't support variable-length lookbehind
        # so we must test this condition with the usual string methods
        if "numpy" in lines[i]:
            has_numpy = True
            lines[i] = lines[i].replace("[m,n]", "")
            lines[i] = lines[i].replace("numpy.ndarray", "numpy.typing.NDArray")
            lines[i] = pattern.sub("", lines[i])

    if has_numpy:
        lines.insert(lines.index("import numpy\n") + 1, "import numpy.typing\n")
        lines.remove("import flags\n")

with open(stub_path, "w") as stub_file:
    stub_file.writelines(lines)

print(f"Fixed {stub_path}")
