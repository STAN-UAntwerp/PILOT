// Include Rcpp system header file (e.g. <>)
#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

#include "forest.h"
#include <memory>

// Expose (some of) the PILOT class
RCPP_MODULE(RcppRAFFLE){
  Rcpp::class_<RAFFLE>("RAFFLEcpp")
  .constructor()
  .constructor<arma::uword,arma::vec,arma::uvec,double,double>()
  .nonconst_method("train", &RAFFLE::train)
  .const_method("print", &RAFFLE::print)
  .const_method("getResiduals", &RAFFLE::getResiduals)
  .const_method("predict", &RAFFLE::predict)
  .const_method("toJson", &RAFFLE::toJson)
  .nonconst_method("fromJson", &RAFFLE::fromJson);
}

