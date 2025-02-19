
#pragma once
#ifndef UTILS_H
#define UTILS_H

#include "RcppArmadillo.h"


std::string trim(const std::string &str);
  
arma::vec parse_json_array(const std::string &json_array);

template <typename VecType>
std::string arma_vec_to_json(const VecType &vec)
{ // convert armadillo vector (vec, uvec, ivec) to json array
  std::ostringstream oss;
  oss << "[";
  
  for (size_t i = 0; i < vec.n_elem; ++i)
  {
    oss << vec(i);
    if (i != vec.n_elem - 1)
    {
      oss << ", ";
    }
  }
  
  oss << "]";
  return oss.str();
}


std::string json_safe_number(double value);
std::string parse_json_block(const std::string &first_line,
                             std::istream &input_stream);
  
#endif