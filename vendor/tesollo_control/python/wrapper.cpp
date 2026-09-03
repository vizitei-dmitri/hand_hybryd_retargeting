#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "../dg_control/control.hpp"
#include "../poses/poses.hpp"

using namespace handcontrol;

namespace py = pybind11;

py::array_t<float> to_numpy_copy(const float* data, size_t size)
{
    py::array_t<float> arr(size);
    std::memcpy(arr.mutable_data(), data, size * sizeof(float));
    return arr;
}

PYBIND11_MODULE(dg5f_python, m)
{
    py::class_<DGControl>(m, "DGApi")
        .def_static(
            "instance",
            &DGControl::getInstance,
            py::return_value_policy::reference
        )
        .def("start", &DGControl::start)
        .def("stop", &DGControl::stop)
        .def("set_target_position",
            [](handcontrol::DGControl& self,
            py::array_t<float, py::array::c_style | py::array::forcecast> arr)
            {
                auto buf = arr.request();

                if (buf.ndim != 1)
                    throw std::runtime_error("Expected 1D array");

                if (buf.size != MAX_JOINT_COUNT)
                    throw std::runtime_error(
                        "Expected array of size " + std::to_string(MAX_JOINT_COUNT));

                const float* ptr = static_cast<const float*>(buf.ptr);

                return self.setTragetPosition(ptr);
            }
        )
        .def("get_current_position",
            [](handcontrol::DGControl& self)
            {
                py::array_t<float> arr({MAX_JOINT_COUNT});

                auto buf = arr.request();
                float* ptr = static_cast<float*>(buf.ptr);

                bool ok = self.getCurrentPosition(ptr);

                return py::make_tuple(ok, arr);
            }
        )                                                                                                                                                                                                                                                                                                                
        .def("get_current_current",
            [](handcontrol::DGControl& self)
            {
                py::array_t<float> arr({MAX_JOINT_COUNT});

                auto buf = arr.request();
                float* ptr = static_cast<float*>(buf.ptr);

                bool ok = self.getCurrentCurrent(ptr);

                return py::make_tuple(ok, arr);
            }
        ) 
        .def("get_current_velocity",
            [](handcontrol::DGControl& self)
            {
                py::array_t<float> arr({MAX_JOINT_COUNT});

                auto buf = arr.request();
                float* ptr = static_cast<float*>(buf.ptr);

                bool ok = self.getCurrentVelocity(ptr);

                return py::make_tuple(ok, arr);
            }
        ) 
        .def("get_current_temp",
            [](handcontrol::DGControl& self)
            {
                py::array_t<float> arr({MAX_JOINT_COUNT});

                auto buf = arr.request();
                float* ptr = static_cast<float*>(buf.ptr);

                bool ok = self.getCurrentTemperature(ptr);

                return py::make_tuple(ok, arr);
            }
        );

    auto poses = m.def_submodule("poses");

    auto grasps = poses.def_submodule("grasps");
    auto gestures = poses.def_submodule("gestures");
    auto numbers = poses.def_submodule("numbers");

    // ===== GRASPS =====
    grasps.attr("OPEN") =
        to_numpy_copy(poses::grasps::OPEN, MAX_JOINT_COUNT);

    grasps.attr("OPEN_BIG") =
        to_numpy_copy(poses::grasps::OPEN_BIG, MAX_JOINT_COUNT);

    grasps.attr("OPEN_VERY_BIG") =
        to_numpy_copy(poses::grasps::OPEN_VERY_BIG, MAX_JOINT_COUNT);

    grasps.attr("FIVE_FINGER_TO_POINT") =
        to_numpy_copy(poses::grasps::FIVE_FINGER_TO_POINT, MAX_JOINT_COUNT);

    grasps.attr("THREE_FINGER_TO_POINT") =
        to_numpy_copy(poses::grasps::THREE_FINGER_TO_POINT, MAX_JOINT_COUNT);

    grasps.attr("THUMB_TO_INDEX") =
        to_numpy_copy(poses::grasps::THUMB_TO_INDEX, MAX_JOINT_COUNT);

    grasps.attr("THUMB_TO_MIDDLE") =
        to_numpy_copy(poses::grasps::THUMB_TO_MIDDLE, MAX_JOINT_COUNT);

    grasps.attr("FOUR_FINGER_TO_RECT") =
        to_numpy_copy(poses::grasps::FOUR_FINGER_TO_RECT, MAX_JOINT_COUNT);

    // ===== GESTURES =====
    gestures.attr("START") =
        to_numpy_copy(poses::gestures::START, MAX_JOINT_COUNT);

    gestures.attr("PEACE") =
        to_numpy_copy(poses::gestures::PEACE, MAX_JOINT_COUNT);

    gestures.attr("GOAT") =
        to_numpy_copy(poses::gestures::GOAT, MAX_JOINT_COUNT);

    gestures.attr("FIXERS") =
        to_numpy_copy(poses::gestures::FIXERS, MAX_JOINT_COUNT);

    gestures.attr("MIDDLE_FINGER") =
        to_numpy_copy(poses::gestures::MIDDLE_FINGER, MAX_JOINT_COUNT);

    gestures.attr("GOAT") =
        to_numpy_copy(poses::gestures::OKEY, MAX_JOINT_COUNT);

    gestures.attr("FIST") =
        to_numpy_copy(poses::gestures::FIST, MAX_JOINT_COUNT);

    gestures.attr("INDEX") =
        to_numpy_copy(poses::gestures::INDEX, MAX_JOINT_COUNT);

    gestures.attr("HOOK") =
        to_numpy_copy(poses::gestures::HOOK, MAX_JOINT_COUNT);

    gestures.attr("HOOK_INDEX_1") =
        to_numpy_copy(poses::gestures::HOOK_INDEX_1, MAX_JOINT_COUNT);

    gestures.attr("HOOK_INDEX_2") =
        to_numpy_copy(poses::gestures::HOOK_INDEX_2, MAX_JOINT_COUNT);

    gestures.attr("OPEN_HOLDER_1") =
        to_numpy_copy(poses::gestures::OPEN_HOLDER_1, MAX_JOINT_COUNT);

    gestures.attr("OPEN_HOLDER_2") =
        to_numpy_copy(poses::gestures::OPEN_HOLDER_2, MAX_JOINT_COUNT);

    // ===== NUMBERS =====
    numbers.attr("ZERO") =
        to_numpy_copy(poses::numbers::ZERO, MAX_JOINT_COUNT);

    numbers.attr("ONE") =
        to_numpy_copy(poses::numbers::ONE, MAX_JOINT_COUNT);

    numbers.attr("TWO") =
        to_numpy_copy(poses::numbers::TWO, MAX_JOINT_COUNT);

    numbers.attr("THREE") =
        to_numpy_copy(poses::numbers::THREE, MAX_JOINT_COUNT);

    numbers.attr("FOUR") =
        to_numpy_copy(poses::numbers::FOUR, MAX_JOINT_COUNT);

    numbers.attr("FIVE") =
        to_numpy_copy(poses::numbers::FIVE, MAX_JOINT_COUNT);
}