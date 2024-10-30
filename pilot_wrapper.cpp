#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>
#include <carma>
#include "tree.h"

namespace py = pybind11;

arma::ivec list_to_ivec(const py::list& list) {
    arma::ivec vec(list.size());
    for (size_t i = 0; i < list.size(); ++i) {
        vec[i] = list[i].cast<int>();
    }
    return vec;
}

arma::mat numpy_to_mat(const py::array_t<double>& array) {
    py::buffer_info info = array.request();
    double* data = static_cast<double*>(info.ptr);
    arma::mat mat(info.shape[0], info.shape[1]); // Create an empty matrix with the correct shape

    // Manually copy the data from the NumPy array to the Armadillo matrix
    for (size_t i = 0; i < info.shape[0]; ++i) {
        for (size_t j = 0; j < info.shape[1]; ++j) {
            mat(i, j) = data[i * info.shape[1] + j];
        }
    }

    return mat;
}

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
    PyPILOT(const py::list& dfs,
            unsigned int min_sample_leaf,
            unsigned int min_sample_alpha,
            unsigned int min_sample_fit,
            unsigned int maxDepth,
            unsigned int maxModelDepth,
            double precScale)
        : pilot(list_to_ivec(dfs),
                min_sample_leaf,
                min_sample_alpha,
                min_sample_fit,
                maxDepth,
                maxModelDepth,
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

private:
    PILOT pilot;
};

PYBIND11_MODULE(cpilot, m) {
    py::class_<PyPILOT>(m, "PILOT")
        .def(py::init<const py::list&,
                      unsigned int,
                      unsigned int,
                      unsigned int,
                      unsigned int,
                      unsigned int,
                      double>())
        .def("train", &PyPILOT::train)
        .def("predict", &PyPILOT::predict);
}