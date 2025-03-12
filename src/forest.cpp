#include "forest.h"
#include "utils.h"

// Some notes:
// ranking is still done locally, which harms computation time
// 


// constructors 

RAFFLE::RAFFLE(const arma::uword& nTrees, 
               const arma::vec& dfs,
               const arma::uvec& modelParams,
               const double &rel_tolerance,
               const double& precScale) : nTrees(nTrees) {
  
  
  if (nTrees <= 0)
  {
    throw std::range_error("nTrees should be a positive integer.");
  }
  
  // fill the fleet with empty pilots
  for (arma::uword i=0; i < nTrees; i++) {
    fleet.emplace_back(dfs, modelParams, rel_tolerance, precScale);  // Efficient in-place construction
  }
  
}



// methods

void RAFFLE::train(const arma::mat &X,
                   const arma::vec &y,
                   const arma::uvec &catIds)
{ // d-dimensional vector indicating the categorical variables with 1
  // can perform input checks here
  n = X.n_rows;
  for (auto& pilot : fleet) { 
    arma::uvec bs = arma::randi<arma::uvec>(n, arma::distr_param(0, n - 1));// bootstrap Ids
    arma::mat X_bs = X.rows(bs);
    arma::vec y_bs = y.elem(bs);
    pilot.train(X_bs, y_bs, catIds);
  }
}


arma::colvec RAFFLE::predict(const arma::mat& X,
                             arma::uword upToDepth) const {
  
  if (fleet.empty()) {
    throw std::logic_error("Fleet is empty, cannot check training status.");
  }
  arma::vec predictions(X.n_rows, arma::fill::zeros);
  
  for (auto& pilot : fleet) { // iterate through trees and add predictions
    predictions += pilot.predict(X, upToDepth, 0).col(0);
  }
  predictions /= static_cast<double>(nTrees);
  return(predictions);
}

arma::cube RAFFLE::print() const {
  if (fleet.empty()) {
    throw std::logic_error("Fleet is empty, cannot check training status.");
  }
  
  arma::cube output(1, 10 , nTrees);
  output.fill(arma::datum::nan);
  
  for (arma::uword i = 0; i < nTrees; i++) {
    arma::mat newpilot = fleet[i].print();
    
    if (newpilot.n_rows > output.n_rows) {
      output.resize(newpilot.n_rows, 10, nTrees);
    }
    output.slice(i).head_rows(newpilot.n_rows) = newpilot;
  }
  return(output);
}

arma::vec RAFFLE::getResiduals(const arma::mat& X,
                               const arma::vec& y,
                               arma::uword upToDepth) const {
  // We calculate the predictions for the original dataset.
  // we cannot extract them from the individual trees, since
  // they have been trained on bootstrapped samples.
  
  if (fleet.empty()) {
    throw std::logic_error("Fleet is empty, cannot check training status.");
  }
  
  arma::vec predictions(X.n_rows, arma::fill::zeros);
  
  for (auto& pilot : fleet) {
    predictions += pilot.predict(X, upToDepth, 0).col(0);
  }
  
  arma::vec residuals = y - predictions / static_cast<double>(nTrees);
  return(residuals);
}


// json parsing



std::string RAFFLE::toJson() const
{
  
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"nTrees\": " << nTrees << ",\n";
  oss << "  \"pilots\": [\n";
  
  for (size_t i = 0; i < fleet.size(); ++i) {
    oss << fleet[i].toJson();
    if (i < fleet.size() - 1) oss << ",\n";
  }
  oss << "] \n}";
  
  return (oss.str());
}


void RAFFLE::fromJson(const std::string &json_text)
{
  std::istringstream ss(json_text);
  std::string line;
  bool readingTrees = false;
  std::string treeJson;
  size_t treeNB = 0;
  
  while (std::getline(ss, line))
  {
    line = trim(line);
    
    if (line.find("\"nTrees\":") != std::string::npos)
    {
      nTrees = std::stoul(trim(line.substr(line.find(":") + 1)));
    }
    if (line.find("\"pilots\": [") != std::string::npos) {
      readingTrees = true;
      continue;
    }
    
    if (readingTrees) {
      treeJson = parse_json_block(line, ss);
      fleet[treeNB].fromJson(treeJson);
      treeNB++;
      if (treeNB == nTrees) {
        readingTrees = false;
      }
    }
  }
}

