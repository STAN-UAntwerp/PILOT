#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>
#include <carma>
#include "tree.h"

namespace py = pybind11;

arma::vec numpy_to_vec(const py::array_t<double>& array) {
    py::buffer_info info = array.request();
    double* data = static_cast<double*>(info.ptr);
    return arma::vec(data, info.size, false, true);
}

arma::uvec numpy_to_uvec(const py::array_t<unsigned int>& array) {
    py::buffer_info info = array.request();
    unsigned int* data = static_cast<unsigned int*>(info.ptr);
    arma::uvec vec(info.size);
    for (size_t i = 0; i < info.size; ++i) {
        vec[i] = static_cast<arma::uword>(data[i]);
    }
    return vec;
}

py::array_t<double> vec_to_numpy(const arma::vec& vec) {
    return py::array_t<double>(vec.n_elem, vec.memptr());
}

class PyPILOT {
public:
    PyPILOT(const py::array_t<double>& dfs,
            unsigned int min_sample_leaf,
            unsigned int min_sample_alpha,
            unsigned int min_sample_fit,
            unsigned int maxDepth,
            unsigned int maxModelDepth,
            unsigned int maxFeatures,
            unsigned int approx,
            double rel_tolerance,
            double precScale)
        : pilot(numpy_to_vec(dfs),
                min_sample_leaf,
                min_sample_alpha,
                min_sample_fit,
                maxDepth,
                maxModelDepth,
                maxFeatures,
                approx,
                rel_tolerance,
                precScale) {}

    void train(const py::array_t<double>& X,
               const py::array_t<double>& y,
               const py::array_t<unsigned int>& catIds) {
        pilot.train(carma::arr_to_mat(X),
                    numpy_to_vec(y),
                    numpy_to_uvec(catIds));
    }

    py::array_t<double> predict(const py::array_t<double>& X) const {
        arma::vec predictions = pilot.predict(carma::arr_to_mat(X));
        return vec_to_numpy(predictions);
    }

    py::array_t<double> print() const {
        arma::mat tree = pilot.print();
        return carma::mat_to_arr(tree);
    }

private:
    PILOT pilot;
};

PYBIND11_MODULE(cpilot, m) {
    py::class_<PyPILOT>(m, "PILOT")
        .def(py::init<const py::list&, //dfs
                      unsigned int, //min_sample_leaf
                      unsigned int, //min_sample_alpha
                      unsigned int, //min_sample_fit
                      unsigned int, //maxDepth
                      unsigned int, //maxModelDepth
                      unsigned int, //maxFeatures
                      unsigned int, //approx
                      double, //rel_tolerance
                      double>()) //precScale
        .def("train", &PyPILOT::train)
        .def("predict", &PyPILOT::predict)
        .def("print", &PyPILOT::print);
}