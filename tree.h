#pragma once
#ifndef TREE_H
#define TREE_H

#define ARMA_NO_DEBUG
#include <armadillo> 
#include <memory>


struct bestSplitOut {
  arma::uword best_feature; // The feature/predictor id at which the dataset is best split.
  double best_splitVal; // value at which split should occur
  arma::uword best_type; // node type 0/1/2/3/4/5 for con/lin/pcon/blin/plin/pconc
  double best_intL;
  double best_slopeL;
  double best_intR;
  double best_slopeR;
  double best_rss; // best residual sums of squares
  double best_bic; // best BIC criterion
  double best_rangeL; // The left bound of the range of the training data on this node
  double best_rangeR; // The right bound of the range of the training data on this node
  arma::vec best_pivot_c; // An array of the levels belong to the left node, sorted increasingly. Used if the chosen feature/predictor is categorical.
  arma::uvec ind_local_c; //  An array of the indices of the best categorical variable x with levels in pivot_c. Used if the chosen feature/predictor is categorical.
};

class node {
public:
  // parameters set at construction
  arma::uword nodeId; //depth of node
  arma::uword depth; //depth of node
  arma::uword modelDepth; //depth of node, also incremented for lin models
  arma::uvec  obsIds; // IDs of observations in this node
  
  //parameters set after evaluating best node type
  arma::uword predId; // feature id used for splitting or model
  arma::uword type; // node type 0/1/2/3/4/5 for con/lin/pcon/blin/plin/pconc
  double rss;
  double splitVal; // value of that feature at splitpoint
  double intL;
  double slopeL;
  double intR;
  double slopeR;
  double rangeL;
  double rangeR;
  arma::vec pivot_c; // An array of the levels belong to the left node sorted in increasing order. Used if the chosen feature/predictor is categorical.
  
  std::unique_ptr<node> left;
  std::unique_ptr<node> right;
};


class PILOT {
  
public:
  // constructors
  // PILOT();
  PILOT(const arma::vec& dfs,
        const arma::uword& min_sample_leaf,
        const arma::uword& min_sample_alpha,
        const arma::uword& min_sample_fit,
        const arma::uword& maxDepth,
        const arma::uword& maxModelDepth,
        const arma::uword& maxFeatures,
        const arma::uword& approx,
        const double &rel_tolerance,
        const double& precScale);
  // for forests, should still include:
  // const arma::uword& Id, const arma::uword mtry
  // etc.
  
  
  // public methods
  void train(const arma::mat& X,
             const arma::vec& y,
             const arma::uvec& catIds);
  arma::colvec predict(const arma::mat& X) const;
  arma::mat print() const;
  arma::vec getResiduals() const;
  
protected:
  // protected methods
  void growTree(node* nd,
                const arma::vec & y,
                const arma::mat & X,
                const arma::umat& Xrank,
                const arma::uvec & catIds);
  bestSplitOut findBestSplit(arma::uvec & obsIds, // observations currently in this node
                             const arma::mat & X, // matrix of predictors
                             const arma::umat& Xrank,
                             const arma::uvec & catIds);
  void printNode(node* nd, arma::mat& tr) const;
  bool stopGrowing(node* nd) const; // const as it will not change anything to the PILOT object
  
protected:
  // protected fields
  //int _id;
  arma::vec dfs;// the degrees of freedom for 'con/lin/pcon'/'blin'/'plin/pconc'. negative values means they are not considered
  arma::uword min_sample_leaf; // the minimal number of samples required to be a leaf node
  arma::uword min_sample_alpha; // the minimal number of samples required to fit a model that splits the node.
  arma::uword min_sample_fit; // minimal number of samples required to fit any model
  arma::uword maxDepth; // max depth excluding lin nodes
  arma::uword maxModelDepth; // max depth counting lin nodes as well
  arma::uword maxFeatures;
  arma::uword approx;
  double rel_tolerance;
  double precScale;
  double lowerBound;
  double upperBound;
 
  std::unique_ptr<node> root;
  arma::vec res; // contains the residuals 
  arma::uvec nbNodesPerModelDepth; // count number of existing nodes per model depth
};

#endif