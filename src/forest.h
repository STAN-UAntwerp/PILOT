
#pragma once
#ifndef FOREST_H
#define FOREST_H

#include "tree.h"

class RAFFLE {
public:
  // constructors
  RAFFLE(){
    fleet.clear();
    };
  RAFFLE(const arma::uword& nTrees, 
         const arma::vec& dfs,
         const arma::uvec& modelParams,
         const double &rel_tolerance,
         const double& precScale);
  
  // public methods
  void train(const arma::mat& X,
             const arma::vec& y,
             const arma::uvec& catIds);
  arma::colvec predict(const arma::mat& X,
                       arma::uword upToDepth) const;
  arma::cube print() const;
  arma::vec getResiduals(const arma::mat& X,
                         const arma::vec& y,
                         arma::uword upToDepth) const;

  std::string toJson(arma::uword treeNb = 0) const;
  void fromJson(const std::string& json_text);
  // 
  void load_from_file(const std::string& filename);
  void write_to_file(const std::string& filename) const {
    std::ofstream file(filename);
    if (file.is_open()) {
      file << toJson(0);
      file.close();
    } else {
      std::cerr << "Error opening file: " << filename << "\n";
    }
  }
  
  
protected:
  
  std::vector<PILOT> fleet;
  arma::uword nTrees;
  arma::uword n;
  
};

#endif