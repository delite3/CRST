#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "radarModel.h"

namespace py = pybind11;

PYBIND11_MODULE(radar_model, m)
{
    py::class_<BoschRadar>(m, "BoschRadar")
        .def(py::init<const Floating, const Floating, const Floating, const Floating, const Floating, const Floating>(), py::arg("snr50db"), py::arg("rcs50"), py::arg("r50"), py::arg("pf"), py::arg("scale"), py::arg("rcs_min"))
        .def_static("db_to_linear", &BoschRadar::dbToLinear, py::arg("db"))
        .def("compute_snr", py::overload_cast<const Floating, const Floating, const Floating>(&BoschRadar::computeSnr), py::arg("range"), py::arg("phi"), py::arg("rcs"))
        .def("compute_snr", py::overload_cast<const Eigen::Ref<const ArrayXF> &, const Eigen::Ref<const ArrayXF> &, const Floating>(&BoschRadar::computeSnr), py::arg("range").noconvert(), py::arg("phi").noconvert(), py::arg("rcs"))
        .def("compute_pd", py::overload_cast<const Floating>(&BoschRadar::computePd), py::arg("snr"))
        .def("compute_pd", py::overload_cast<const Eigen::Ref<const ArrayXF> &>(&BoschRadar::computePd), py::arg("snr").noconvert())
        .def("gaussian_gain", py::overload_cast<const Floating>(&BoschRadar::gaussianGain), py::arg("phi"))
        .def("gaussian_gain", py::overload_cast<const Eigen::Ref<const ArrayXF> &>(&BoschRadar::gaussianGain), py::arg("phi").noconvert())
        .def("sim_rcs", py::overload_cast<const Floating, const Floating, const bool>(&BoschRadar::simRcs), py::arg("r"), py::arg("rcs"), py::arg("add_noise"))
        .def("sim_rcs", py::overload_cast<const Eigen::Ref<const ArrayXF> &, const Floating, const bool>(&BoschRadar::simRcs), py::arg("r").noconvert(), py::arg("rcs"), py::arg("add_noise"));
}