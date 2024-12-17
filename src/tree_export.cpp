// Include Rcpp system header file (e.g. <>)
#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

#include "tree.h"
#include <memory>

// Expose (some of) the PILOT class
RCPP_MODULE(RcppPILOT){
  Rcpp::class_<PILOT>("PILOTcpp")
  .constructor<arma::vec,arma::uvec,double,double>()
  .nonconst_method("train", &PILOT::train)
  .const_method("print", &PILOT::print)
  .const_method("getResiduals", &PILOT::getResiduals)
  .const_method("predict", &PILOT::predict)
  .const_method("toJson", &PILOT::toJson)
  .const_method("fromJson", &PILOT::toJson);
}

