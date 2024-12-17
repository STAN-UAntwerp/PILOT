#include "tree.h"

// constructors 

PILOT::PILOT(const arma::vec &dfs,
             const arma::uvec &modelParams,
             /*const arma::uword &min_sample_leaf,
             const arma::uword &min_sample_alpha,
             const arma::uword &min_sample_fit,
             const arma::uword &maxDepth,
             const arma::uword &maxModelDepth,
             const arma::uword &maxFeatures,
             const arma::uword &approx,*/
             const double &rel_tolerance,
             const double &precScale) : dfs(dfs),
             min_sample_leaf(modelParams(0)),
             min_sample_alpha(modelParams(1)),
             min_sample_fit(modelParams(2)),
             maxDepth(modelParams(3)),
             maxModelDepth(modelParams(4)),
             maxFeatures(modelParams(5)),
             approx(modelParams(6)),
             rel_tolerance(rel_tolerance),
             precScale(precScale)
{
  // if we want, we can do an additional input check here
  
  if (dfs(0) < 0)
  {
    throw std::range_error("The con node should have non-negative degrees of freedom.");
  }
  
  if ((approx > 0) && (dfs(3) >= 0))
  {
    throw std::range_error("Approximate cannot be used (yet) in conjunction with the blin model.");
  }
  
  root = nullptr;
}

// methods

void PILOT::train(const arma::mat &X,
                  const arma::vec &y,
                  const arma::uvec &catIds)
{ // d-dimensional vector indicating the categorical variables with 1
  // can perform input checks here
  if (maxFeatures > X.n_cols)
  {
    throw std::range_error("maxFeatures should not be larger than the number of features");
  }
  
  // calculate Xrank
  // note that Xrank gives ties the same rank!s
  arma::umat Xrank(X.n_rows, X.n_cols, arma::fill::zeros);
  arma::uvec xrank(X.n_rows);
  arma::vec x(X.n_rows);
  arma::uvec xxorder(X.n_rows);
  maxNbCats = 0;
  for (arma::uword j = 0; j < X.n_cols; j++)
  { // O(d n\log(n))
    x = X.col(j);
    xxorder = arma::sort_index(x);
    xrank.zeros();
    
    if (catIds(j) == 0)
    { // for continuous variables, ties are getting contiguous ranks
      xrank(xxorder) = arma::regspace<arma::uvec>(0, X.n_rows - 1);
    }
    else if (catIds(j) == 1)
    { // for categorical variables, ties are getting the same ranks
      
      xrank(xxorder(0)) = 0;
      arma::uword counter = 0;
      for (arma::uword i = 1; i < X.n_rows; i++)
      {
        if (x(xxorder(i)) - x(xxorder(i - 1)) > precScale)
        { // ties get the same order
          counter++;
        }
        xrank(xxorder(i)) = counter;
      }
      
      if (maxNbCats < x(xxorder(xxorder.n_elem - 1)))
      {
        maxNbCats = (arma::uword)x(xxorder(xxorder.n_elem - 1));
      }
    }
    Xrank.col(j) = xrank;
  }
  maxNbCats++;
  
  sumsy.set_size(maxNbCats);
  sumsy2.set_size(maxNbCats);
  IDmask.set_size(maxNbCats);
  
  //Rcpp::Rcout << "Features ranked. Start growing tree" << std::endl;
  // build root node
  root = std::make_unique<node>();
  root->startID = 0;
  root->endID = X.n_rows - 1;
  root->depth = 0;
  root->modelDepth = 0;
  root->nodeId = 0;
  
  IDvec = arma::regspace<arma::uvec>(0, 1, X.n_rows - 1);
  // build tree
  res = y; // initialize res
  double maxy = arma::max(y);
  double miny = arma::min(y);
  lowerBound = miny; // 5 * miny / 4 - maxy / 4; // min(y) - range(y) / 4
  upperBound = maxy;  // 5 * maxy / 4 - miny / 4; // max(y) + range(y) / 4
  nbNodesPerModelDepth = arma::zeros<arma::uvec>(maxModelDepth + 1); // initialize nodes per model depth
  nbNodesPerModelDepth(0) = 1;                                       // 1 root node at depth 0
  
  PILOT::growTree(root.get(), y, X, Xrank, catIds);
  
  // reduce memory usage by resetting temporary variables.
  sumsy.reset();
  sumsy2.reset();
  IDmask.reset();
}

arma::vec PILOT::getResiduals() const
{
  return (res);
}

arma::mat PILOT::print() const
{
  // print tree recursively starting from a root
  // matrix has
  // 1st column: depth
  // 2nd column: modelDepth
  // 3nd column: nodeId
  // 4nd column: nodetype
  // 5rd column: feature index
  // 6th column: split value
  // 7th column: int left
  // 8th column: slope left
  // 9th column: int right
  // 10th column: slope right
  // first the left node row is added, then right node
  // could also add modelID which increments for lin nodes as well.
  
  arma::mat tr(0, 10);
  if (root != nullptr)
  { // check if tree has been constructed
    node *nd = root.get();
    printNode(nd, tr);
  }
  return tr;
}

void PILOT::printNode(node *nd, arma::mat &tr) const
{
  
  tr.insert_rows(tr.n_rows, 1);
  arma::rowvec vec = {(double)nd->depth,
                      (double)nd->modelDepth,
                      (double)nd->nodeId,
                      (double)nd->type,
                      (double)nd->predId,
                      (double)nd->splitVal,
                      (double)nd->intL,
                      (double)nd->slopeL,
                      (double)nd->intR,
                      (double)nd->slopeR};
  if (nd->type == 0)
  {
    vec(4) = arma::datum::nan;
  }
  tr.row(tr.n_rows - 1) = vec;
  if (nd->type == 1)
  { // lin node
    printNode(nd->left.get(), tr);
  }
  else if (nd->type > 1)
  { // pcon/blin/plin/pconc --> split
    printNode(nd->left.get(), tr);
    printNode(nd->right.get(), tr);
  }
}

bool PILOT::stopGrowing(node *nd) const
{
  // check that depth is less than maxDepth
  if (nd->depth >= maxDepth)
  {
    return true;
  }
  // check that depth is less than maxDepth
  if (nd->modelDepth >= maxModelDepth)
  {
    return true;
  }
  // at least min_sample_fit number of points for continuing growth
  if (nd->endID - nd->startID - 1 <= min_sample_fit)
  {
    return true;
  }
  return false;
}

void PILOT::growTree(node *nd,
                     const arma::vec &y,
                     const arma::mat &X,
                     const arma::umat &Xrank,
                     const arma::uvec &catIds)
{
  
  if (stopGrowing(nd))
  {
    // update res and rss
    nd->intL = arma::mean(res(IDvec.subvec(nd->startID, nd->endID)));
    nd->slopeL = arma::datum::nan;
    nd->splitVal = arma::datum::nan;
    nd->predId = arma::datum::nan;
    nd->intR = arma::datum::nan;
    nd->slopeR = arma::datum::nan;
    nd->rangeL = arma::datum::nan;
    nd->rangeR = arma::datum::nan;
    res(IDvec.subvec(nd->startID, nd->endID)) -= nd->intL; // subtract mean
    nd->rss = arma::sum(arma::square(res(IDvec.subvec(nd->startID, nd->endID))));
    // set type to con, and no new call to growtree
    nd->type = 0;
  }
  else
  {
    bestSplitOut newSplit = findBestSplit(nd->startID,
                                          nd->endID,
                                          X,
                                          Xrank,
                                          catIds);
    
    nd->rss = newSplit.best_rss;
    nd->splitVal = newSplit.best_splitVal; // value of that feature at splitpoint
    nd->intL = newSplit.best_intL;
    nd->slopeL = newSplit.best_slopeL;
    nd->intR = newSplit.best_intR;
    nd->slopeR = newSplit.best_slopeR;
    nd->predId = newSplit.best_feature;
    nd->type = newSplit.best_type;
    nd->rangeL = newSplit.best_rangeL;
    nd->rangeR = newSplit.best_rangeR;
    nd->pivot_c = newSplit.best_pivot_c;
    
    /// update residuals and depth + continue growing
    if (newSplit.best_type == 0)
    {                                                                  // con is best split
      // update res and rss
      res(IDvec.subvec(nd->startID, nd->endID)) -= newSplit.best_intL; // subtract mean
      
      // no new call to growtree
    }
    else if (newSplit.best_type == 1)
    { // lin is best split
      
      arma::uvec obsIds = IDvec.subvec(nd->startID, nd->endID);
      arma::vec x = X.col(newSplit.best_feature);
      x = x(obsIds);
      // update res
      double rss_old = arma::sum(arma::square(res(obsIds)));
      res(obsIds) += (-newSplit.best_intL - newSplit.best_slopeL * x);
      // truncate prediction
      // we clip the overall predictions (=y - current residuals) between lowerBound and upperBound
      
      res(obsIds) = y(obsIds) - arma::clamp(y(obsIds) - res(obsIds), lowerBound, upperBound);
      
      double rss_new = arma::sum(arma::square(res(obsIds)));
      
      if ((rss_old - rss_new) / rss_old < rel_tolerance)
      {
        nd->intL = arma::mean(res(obsIds));
        nd->slopeL = arma::datum::nan;
        nd->splitVal = arma::datum::nan;
        nd->predId = arma::datum::nan;
        nd->intR = arma::datum::nan;
        nd->slopeR = arma::datum::nan;
        nd->rangeL = arma::datum::nan;
        nd->rangeR = arma::datum::nan;
        res(obsIds) -= nd->intL; // subtract mean
        nd->rss = arma::sum(arma::square(res(obsIds)));
        // set type to con, and no new call to growtree
        nd->type = 0;
      }
      else
      {
        // construct a new node (left node only here)
        nd->left = std::make_unique<node>();
        
        nd->left->startID = nd->startID;
        nd->left->endID = nd->endID;
        nd->left->depth = nd->depth;
        nd->left->modelDepth = nd->modelDepth + 1;
        //
        
        nd->left->nodeId = nbNodesPerModelDepth(nd->left->modelDepth);
        nbNodesPerModelDepth(nd->left->modelDepth)++;
        
        // continue growing the tree
        growTree(nd->left.get(),
                 y,
                 X,
                 Xrank,
                 catIds);
      }
    }
    else if (newSplit.best_type == 5)
    {                        // split on categorical variable (pconc)
      arma::uword nLeft = 0; // Counter for the number of elements in the left partition
      
      // Iterate over IDvec and partition it based on values of x, while also updating residuals
      // all in a single pass.
      
      for (arma::uword i = nd->startID; i <= nd->endID; ++i)
      {
        arma::uword id = IDvec(i); // Get the actual index for x, y, and res
        
        if (std::binary_search(newSplit.best_pivot_c.begin(), newSplit.best_pivot_c.end(), X(id, newSplit.best_feature)))
        {
          // Left partition: update residuals
          res(id) -= newSplit.best_intL;
          // Move this id to the left partition by swapping it with the "next" right partition element
          std::swap(IDvec(i), IDvec(nd->startID + nLeft)); // Swap the element to the left partition zone
          nLeft++;                                         // Increment left partition count
        }
        else
        {
          // Right partition: update residuals
          res(id) -= newSplit.best_intR;
        }
      }
      
      // construct new nodes
      nd->left = std::make_unique<node>();
      nd->right = std::make_unique<node>();
      
      nd->left->startID = nd->startID;
      nd->left->endID = nd->startID + nLeft - 1;
      nd->right->startID = nd->startID + nLeft;
      nd->right->endID = nd->endID;
      
      nd->left->depth = nd->depth + 1;
      nd->right->depth = nd->depth + 1;
      
      nd->left->modelDepth = nd->modelDepth + 1;
      nd->left->nodeId = nbNodesPerModelDepth(nd->left->modelDepth);
      nbNodesPerModelDepth(nd->left->modelDepth)++;
      
      nd->right->modelDepth = nd->modelDepth + 1;
      nd->right->nodeId = nbNodesPerModelDepth(nd->right->modelDepth);
      nbNodesPerModelDepth(nd->right->modelDepth)++;
      
      // continue growing the tree
      growTree(nd->left.get(),
               y,
               X,
               Xrank,
               catIds);
      growTree(nd->right.get(),
               y,
               X,
               Xrank,
               catIds);
    }
    else
    { // if best split is not lin, con or pconc
      
      arma::uword nLeft = 0; // Counter for the number of elements in the left partition
      
      // Iterate over IDvec and partition it based on values of x, while also updating residuals
      // all in a single pass.
      for (arma::uword i = nd->startID; i <= nd->endID; ++i)
      {
        arma::uword id = IDvec(i); // Get the actual index for x, y, and res
        
        if (X(id, newSplit.best_feature) <= newSplit.best_splitVal)
        {
          // Left partition: update residuals
          res(id) -= newSplit.best_intL + newSplit.best_slopeL * X(id, newSplit.best_feature);
          res(id) = y(id) - std::clamp(y(id) - res(id), lowerBound, upperBound);
          
          // Move this id to the left partition by swapping it with the "next" right partition element
          std::swap(IDvec(i), IDvec(nd->startID + nLeft)); // Swap the element to the left partition zone
          nLeft++;                                         // Increment left partition count
        }
        else
        {
          // Right partition: update residuals
          res(id) -= newSplit.best_intR + newSplit.best_slopeR * X(id, newSplit.best_feature);
          res(id) = y(id) - std::clamp(y(id) - res(id), lowerBound, upperBound);
        }
      }
      // construct new nodes
      nd->left = std::make_unique<node>();
      nd->right = std::make_unique<node>();
      
      nd->left->startID = nd->startID;
      nd->left->endID = nd->startID + nLeft - 1;
      nd->right->startID = nd->startID + nLeft;
      nd->right->endID = nd->endID;
      nd->left->depth = nd->depth + 1;
      nd->right->depth = nd->depth + 1;
      
      nd->left->modelDepth = nd->modelDepth + 1;
      nd->left->nodeId = nbNodesPerModelDepth(nd->left->modelDepth);
      nbNodesPerModelDepth(nd->left->modelDepth)++;
      
      nd->right->modelDepth = nd->modelDepth + 1;
      nd->right->nodeId = nbNodesPerModelDepth(nd->right->modelDepth);
      nbNodesPerModelDepth(nd->right->modelDepth)++;
      
      // continue growing the tree
      growTree(nd->left.get(),
               y,
               X,
               Xrank,
               catIds);
      growTree(nd->right.get(),
               y,
               X,
               Xrank,
               catIds);
    }
  }
}

bestSplitOut PILOT::findBestSplit(arma::uword startID,
                                  arma::uword endID,
                                  const arma::mat &X, // matrix of predictors
                                  const arma::umat &Xrank,
                                  const arma::uvec &catIds)
{
  
  //   Remarks:
  // > If the input data is not allowed to split, the function will return default values.
  // > categorical variables need to take values in 0, ... , k-1 and the Xorder need to give the ordering of these
  // > cat IDs indicates the categircal variables.
  //
  
  // intialize return values
  
  arma::uword best_feature = arma::datum::nan; // The feature/predictor id at which the dataset is best split.
  double best_splitVal = arma::datum::nan;     // value at which split should occur
  arma::uword best_type = 0;                   // node type 0/1/2/3/4/5 for con/lin/pcon/blin/plin/pconc
  double best_intL = 0.0;
  double best_slopeL = arma::datum::nan;
  double best_intR = arma::datum::nan;
  double best_slopeR = arma::datum::nan;
  double best_rss = arma::datum::inf; // best residual sums of squares
  double best_bic = arma::datum::inf; // best BIC criterion
  double best_rangeL = 0.0;           // The range of the training data on this node
  double best_rangeR = 0.0;           // The range of the training data on this node
  arma::vec best_pivot_c;             // An array of the levels belong to the left node. Used if the chosen feature/predictor is categorical.
  best_pivot_c.reset();
  
  const arma::uword d = X.n_cols;
  const arma::uword nObs = endID - startID + 1;
  const double n = (double)nObs; // double since we use it in many double calculations
  const double logn = std::log(n);
  // first compute con as benchmark
  
  arma::uvec obsIds = IDvec.subvec(startID, endID);
  double sumy = arma::sum(res(obsIds));
  double sumy2 = arma::sum(arma::square(res(obsIds)));
  
  // get BIC of constant model
  double intercept = sumy / n;
  double rss = sumy2 + intercept * intercept * n - 2 * intercept * sumy;
  double bic = n * (std::log(rss) - logn) + logn * dfs(0);
  
  best_intL = intercept;
  best_bic = bic;
  best_rss = rss;
  
  // now check the features whether model building is worth it.
  // declare variables
  arma::uvec xorder(obsIds.n_elem);
  arma::vec xs(obsIds.n_elem);
  arma::vec ys(obsIds.n_elem);
  arma::vec xunique;
  arma::vec::fixed<3> u;
  arma::vec sums;
  arma::vec sums2;
  arma::uvec counts;
  arma::vec pivot_c;
  arma::vec means;
  arma::uvec means_order;
  arma::vec sums_s;
  arma::vec sums2_s;
  arma::uvec counts_s;
  arma::uvec cumcounts;
  
  double sumx, sumx2, sumxy, nL, nR, sumxL, sumx2L, sumyL, sumy2L, sumxyL, sumxR, sumx2R, sumyR, sumy2R, sumxyR, slopeL, intL, slopeR, intR, varL, varR, splitVal_old, splitVal, delta, xdiff, x2diff, ydiff, y2diff, xydiff;
  ;
  arma::mat::fixed<3, 3> XtX;
  arma::vec::fixed<3> XtY;
  
  arma::uvec splitCandidates(obsIds.n_elem);
  arma::uword splitID, nsteps, mini, nSplitCandidates, k, nbUnique, ii, stepSize;
  
  bool sortLocally = false;
  bool approxed;
  
  if (n * std::log(n) < 0.5 * (double)X.n_rows)
  {
    sortLocally = true;
  }
  else
  {
    xorder.set_size(X.n_rows);
  }
  
  arma::uvec featuresToConsider = arma::regspace<arma::uvec>(0, d - 1);
  if (maxFeatures < d)
  {// sample features considered for splitting
    featuresToConsider = arma::randperm(d, maxFeatures); 
  }
  
  for (arma::uword j : featuresToConsider)
  { // iterate over features
    
    if (catIds(j) == 0)
    { // check whether numerical feature
      
      // get sorted variables.
      
      if (sortLocally)
      {
        // if n is small, it is faster to sort locally.
        xorder = arma::sort_index(Xrank.submat(obsIds, arma::uvec{j})); // local order. can also do  arma::sort_index(X.submat(obsIds, arma::uvec{j}));
        xs = X.submat(obsIds(xorder), arma::uvec{j});
        ys = res(obsIds(xorder));
        
        // compute moments
        sumx = arma::sum(xs);
        sumx2 = arma::sum(arma::square(xs));
        sumxy = arma::sum(xs % ys);
      }
      else
      { // This O(n), but can be slow if obsIds.n_elem is small and X.n_rows is large, due to copying large vectors
        
        xorder.fill(X.n_rows); // fill with an out of bound value
        xorder(Xrank.submat(obsIds, arma::uvec{j})) = arma::regspace<arma::uvec>(0, obsIds.n_elem - 1);
        
        sumx = 0;
        sumx2 = 0;
        sumxy = 0;
        arma::uword counter = 0;
        for (arma::uword i = 0; i < X.n_rows; i++)
        {
          if (xorder(i) == X.n_rows)
          {
            continue;
          }
          arma::uword localID = obsIds(xorder(i));
          double newx = X(localID, j);
          double newy = res(localID);
          xs(counter) = newx;
          ys(counter) = newy;
          sumx += newx;
          sumx2 += newx * newx;
          sumxy += newx * newy;
          counter++;
        }
      }
      
      // lin model
      if (dfs(1) >= 0)
      { // check if lin model is allowed
        
        nbUnique = 1; // Start counting from the first element as a unique value
        
        // Iterate through the sorted vector
        for (arma::uword i = 1; i < xs.n_elem; ++i)
        {
          if (xs(i) != xs(i - 1))
          {
            ++nbUnique;
            // Early exit if we have found at least 5 unique elements
            if (nbUnique >= 5)
            {
              break;
            }
          }
        }
        
        if (nbUnique >= 5)
        {                                 // check whether at least 5 unique predictor values
          varL = n * sumx2 - sumx * sumx; // var * n^2
          slopeL = (n * sumxy - sumx * sumy) / varL;
          intL = (sumy - slopeL * sumx) / n;
          rss = sumy2 + n * intL * intL +
            (2 * slopeL * intL * sumx) + (slopeL * slopeL * sumx2) -
            2 * intL * sumy - 2 * slopeL * sumxy;
          bic = n * (std::log(rss) - logn) + logn * dfs(1);
          
          // update if better
          if (bic < best_bic)
          {
            best_feature = j;
            best_splitVal = arma::datum::nan; // value at which split should occur
            best_type = 1;                    // node type 1 for lin
            best_intL = intL;
            best_slopeL = slopeL;
            best_rangeL = xs.front();
            best_rangeR = xs.back();
            best_bic = bic;
            best_rss = rss;
          }
        }
      } // end of lin model
      
      if (n < min_sample_alpha)
      { // check if enough samples to fit piecewise models
        continue;
      }
      
      nL = 0.0;
      nR = n;
      
      // initialize left/right moments
      sumxL = 0.0;
      sumx2L = 0.0;
      sumyL = 0.0;
      sumy2L = 0.0;
      sumxyL = 0.0;
      
      sumxR = sumx;
      sumx2R = sumx2;
      sumyR = sumy;
      sumy2R = sumy2;
      sumxyR = sumxy;
      
      slopeL = 0.0;
      intL = 0.0;
      slopeR = 0.0;
      intR = 0.0;
      varL = 1.0;
      varR = 1.0;
      
      splitVal_old = 0.0;
      
      // for blin, we need to keep the following variables
      // maintain a matrix XtX and vector Xty, which contains
      // the X'X matrix for X_{i, .} = [ 1 x_i (x_i-splitval)^+]
      // this is the design matrix for segmented regression
      // with segmentation point equal to splitVal.
      //
      
      if (dfs(3) >= 0)
      {
        XtX(0, 0) = n;
        XtX(0, 1) = sumx;
        XtX(0, 2) = sumx;
        XtX(1, 0) = sumx;
        XtX(1, 1) = sumx2;
        XtX(1, 2) = sumx2;
        XtX(2, 0) = sumx;
        XtX(2, 1) = sumx2;
        XtX(2, 2) = sumx2;
        
        XtY(0) = sumy;
        XtY(1) = sumxy;
        XtY(2) = sumxy;
      }
      
      // now start with evaluating split models
      // first determine the indices of the possible splits:
      
      stepSize = 1;
      approxed = false;
      
      if ((approx > 0) && (n > approx))
      {
        stepSize = std::round(n / (double)approx);
        approxed = true;
      }
      
      nSplitCandidates = 0;
      for (arma::uword i = 0; i < n; i += stepSize)
      { // xs(i) is the new splitcandidate
        ii = 0;
        while ((i + (ii) < n) && (xs(i + ii) - xs(i) < precScale))
        { // counter the number of occurences of x(i)
          ii++;
        }
        splitCandidates(nSplitCandidates) = i + (ii - 1);
        nSplitCandidates++;
        
        if (ii > 1)
        { // shift the iterator in case of duplicates
          i = i + (ii - 1);
        }
      }
      
      for (arma::uword i = 0; i < nSplitCandidates - 1; i++)
      { // iterate over all candidate split points
        
        splitID = splitCandidates(i);
        splitVal = xs(splitID);
        
        // count number of skipped steps from the previous splitVal
        // This can also be > 1 when not all splitpoints are considered (i.e. approx > 0)
        if (i == 0)
        {
          nsteps = splitID + 1;
        }
        else
        {
          nsteps = splitID - splitCandidates(i - 1);
        }
        
        if (dfs(3) >= 0)
        {
          delta = splitVal - splitVal_old;
          
          // first construct the update vector. This is the one that has to be added to
          // row and column 3 of the data. Note it has to be added only once to cell (3,3)
          u(0) = -delta * nR;
          u(1) = -delta * sumxR;
          u(2) = delta * delta * nR - 2 * delta * XtX(0, 2);
          
          // update XtX and Xty:
          XtX(0, 2) += u(0);
          XtX(1, 2) += u(1);
          XtX(2, 2) += u(2);
          XtX(2, 0) += u(0);
          XtX(2, 1) += u(1);
          
          XtY(2) += -delta * sumyR;
        }
        
        // update sizes
        nL += nsteps;
        nR -= nsteps;
        
        // update moments
        if (nsteps == 1)
        {
          xdiff = splitVal;
          x2diff = xdiff * splitVal;
          ydiff = ys(splitID);
          y2diff = ys(splitID) * ys(splitID);
          xydiff = splitVal * ydiff;
        }
        else
        {
          mini = splitID - nsteps + 1;
          ydiff = arma::sum(ys.subvec(mini, splitID));
          y2diff = arma::sum(arma::square(ys.subvec(mini, splitID)));
          if (approxed)
          {
            xdiff = arma::sum(xs.subvec(mini, splitID));
            x2diff = arma::sum(arma::square(xs.subvec(mini, splitID)));
            xydiff = arma::sum(xs.subvec(mini, splitID) % ys.subvec(mini, splitID));
          }
          else
          { // in this case, nsteps only counts the number of duplicate values, and we can update xdiff and x2diff efficiently.
            xdiff = nsteps * splitVal;
            x2diff = xdiff * splitVal;
            xydiff = splitVal * ydiff;
          }
        }
        
        sumxL += xdiff;
        sumx2L += x2diff;
        sumyL += ydiff;
        sumy2L += y2diff;
        sumxyL += xydiff;
        
        sumxR -= xdiff;
        sumx2R -= x2diff;
        sumyR -= ydiff;
        sumy2R -= y2diff;
        sumxyR -= xydiff;
        
        // check if pcon/blin / plin split is eligible
        // based on the min_sample_leaf criterion
        if (nL < min_sample_leaf)
        {
          splitVal_old = splitVal;
          continue; // skip to next splitpoînt
        }
        else if (nR < min_sample_leaf)
        {
          break; // all splitpoints have been considered for this variable
        }
        
        // pcon model
        if (dfs(2) >= 0)
        {
          intL = sumyL / nL;
          intR = sumyR / nR;
          slopeL = 0.0;
          slopeR = 0.0;
          
          rss = sumy2L + nL * intL * intL -
            2 * intL * sumyL +
            sumy2R + nR * intR * intR -
            2 * intR * sumyR;
          
          bic = n * (std::log(rss) - logn) + logn * dfs(2);
          
          // update if better
          if (bic < best_bic)
          {
            best_feature = j;
            best_splitVal = splitVal; // value at which split should occur
            best_type = 2;            // node type 2 for pcon
            best_intL = intL;
            best_slopeL = slopeL;
            best_intR = intR;
            best_slopeR = slopeR;
            best_rangeL = xs.front();
            best_rangeR = xs.back();
            best_bic = bic;
            best_rss = rss;
          }
        }
        
        // check if blin/plin split is eligible
        // based on the minimum unique values criterion
        // for blin, this is somewhat more strict than the Python implementation
        if ((i < 4) || (nSplitCandidates - i - 1 < 5))
        { // at least 5 unique values needed left
          splitVal_old = splitVal;
          continue; // skip to next splitpoînt
        }
        
        // blin model
        if (dfs(3) >= 0)
        {
          if (XtX.is_sympd())
          {
            arma::vec coefs = solve(XtX, XtY, arma::solve_opts::likely_sympd);
            slopeL = coefs(1);
            intL = coefs(0);
            slopeR = coefs(1) + coefs(2);
            intR = coefs(0) - coefs(2) * splitVal;
            
            rss = sumy2L + nL * intL * intL +
              (2 * slopeL * intL * sumxL) + (slopeL * slopeL * sumx2L) -
              2 * intL * sumyL - 2 * slopeL * sumxyL +
              sumy2R + nR * intR * intR +
              (2 * slopeR * intR * sumxR) + (slopeR * slopeR * sumx2R) -
              2 * intR * sumyR - 2 * slopeR * sumxyR;
            
            bic = n * (std::log(rss) - logn) + logn * dfs(3);
            
            // update if better
            if (bic < best_bic)
            {
              best_feature = j;
              best_splitVal = splitVal; // value at which split should occur
              best_type = 3;            // node type 3 for blin
              best_intL = intL;
              best_slopeL = slopeL;
              best_intR = intR;
              best_slopeR = slopeR;
              best_rangeL = xs.front();
              best_rangeR = xs.back();
              best_bic = bic;
              best_rss = rss;
            }
          }
        }
        
        // plin model
        if (dfs(4) >= 0)
        {
          
          varL = nL * sumx2L - sumxL * sumxL; // var * n^2
          varR = nR * sumx2R - sumxR * sumxR; // var * n^2
          
          if ((varL > precScale * nL * nL) && (varR > precScale * nR * nR))
          {
            slopeL = (nL * sumxyL - sumxL * sumyL) / varL;
            intL = (sumyL - slopeL * sumxL) / nL;
            
            slopeR = (nR * sumxyR - sumxR * sumyR) / varR;
            intR = (sumyR - slopeR * sumxR) / nR;
            
            rss = sumy2L + nL * intL * intL +
              (2 * slopeL * intL * sumxL) + (slopeL * slopeL * sumx2L) -
              2 * intL * sumyL - 2 * slopeL * sumxyL +
              sumy2R + nR * intR * intR +
              (2 * slopeR * intR * sumxR) + (slopeR * slopeR * sumx2R) -
              2 * intR * sumyR - 2 * slopeR * sumxyR;
            
            bic = n * (std::log(rss) - logn) + logn * dfs(4);
            
            // update if better
            if (bic < best_bic)
            {
              best_feature = j;
              best_splitVal = splitVal; // value at which split should occur
              best_type = 4;            // node type 4 for plin
              best_intL = intL;
              best_slopeL = slopeL;
              best_intR = intR;
              best_slopeR = slopeR;
              best_rangeL = xs.front();
              best_rangeR = xs.back();
              best_bic = bic;
              best_rss = rss;
            }
          }
        }
        
        splitVal_old = splitVal;
      }
    }
    else if (catIds(j) == 1)
    { // categorical variables: pconc
      
      k = 1; // maximum number of categories. Highest category takes value k-1
      bool atLeastTwoUnique = false;
      sumsy.zeros();
      sumsy2.zeros();
      IDmask.zeros();
      
      for (arma::uword i = 0; i < nObs; ++i)
      {
        arma::uword overallRank = Xrank(IDvec(startID + i), j);
        IDmask(overallRank)++; // counts the number of times this duplicate value occurs. Note that Xrank has to give the same rank to ties!
        sumsy(overallRank) += res(IDvec(startID + i));
        sumsy2(overallRank) += res(IDvec(startID + i)) * res(IDvec(startID + i));
        if (overallRank + 1 > k)
        {
          k = overallRank + 1;
        }
        if (!atLeastTwoUnique && (Xrank(IDvec(startID), j) != overallRank))
        {
          atLeastTwoUnique = true;
        }
      }
      
      if (atLeastTwoUnique)
      {
        //  at least two unique predictor variables needed for pcon
        // k = arma::conv_to<arma::uword>::from(xunique.tail(1)) + 1;
        sums.zeros(k);
        sums2.zeros(k);
        counts.zeros(k);
        means.zeros(k);
        pivot_c.set_size(k);
        pivot_c.fill((double)k);
        
        // xunique contains the unique elements in ascending order
        // we want the counts per unique element in the same order
        
        arma::uword counter = 0;
        arma::uvec xunique(k);
        for (arma::uword i = 0; i < k; i++)
        {
          if (IDmask(i) > 0)
          {
            sums(counter) = sumsy(i);
            sums2(counter) = sumsy2(i);
            counts(counter) = IDmask(i);
            means(counter) = sumsy(i) / ((double)IDmask(i));
            xunique(counter) = i;
            counter++;
          }
        }
        
        sums = sums.head(counter);
        sums2 = sums2.head(counter);
        counts = counts.head(counter);
        means = means.head(counter);
        xunique = xunique.head(counter);
        // # sort unique values w.r.t. the mean of the responses
        
        means_order = arma::sort_index(means);
        sums_s = sums(means_order);
        sums2_s = sums2(means_order);
        counts_s = counts(means_order);
        cumcounts = arma::cumsum(counts);
        // loop over the sorted possible_p and find the best partition
        
        sumyL = 0;
        sumyR = sumy;
        sumy2L = 0;
        sumy2R = sumy2;
        nL = 0;
        nR = nObs;
        for (arma::uword i = 0; i < xunique.n_elem - 1; i++)
        {
          nL += counts_s(i);
          nR -= counts_s(i);
          
          sumyL += sums_s(i);
          sumyR -= sums_s(i);
          
          sumy2L += sums2_s(i);
          sumy2R -= sums2_s(i);
          
          // now need to insert this so that pivot_c is sorted in the end
          
          pivot_c(xunique(means_order(i))) = xunique(means_order(i));
          
          // here should check that nL and nR are at least min_leaf etc.
          if (nL < min_sample_leaf)
          {
            continue; // skip to next splitpoînt
          }
          else if (nR < min_sample_leaf)
          {
            break; // all splitpoints have been considered for this variable
          }
          intL = sumyL / nL;
          intR = sumyR / nR;
          rss = sumy2L + nL * intL * intL -
            2 * intL * sumyL +
            sumy2R + nR * intR * intR -
            2 * intR * sumyR;
          bic = n * (std::log(rss) - logn) + logn * dfs(5);
          
          // update if better
          if (bic < best_bic)
          {
            best_feature = j;
            best_type = 5; // node type 5 for pconc
            best_rangeL = xunique(0);
            best_rangeR = xunique(xunique.n_elem - 1);
            best_intL = intL;
            best_intR = intR;
            best_bic = bic;
            best_rss = rss;
            best_pivot_c = pivot_c(arma::find(pivot_c < k));
          }
        }
      }
    }
  } // end of loop over features
  
  // construct ouput and return
  bestSplitOut result;
  
  result.best_feature = best_feature;
  result.best_splitVal = best_splitVal; // value at which split should occur
  result.best_type = best_type;         // node type 1 for lin
  result.best_intL = best_intL;
  result.best_slopeL = best_slopeL;
  result.best_intR = best_intR;
  result.best_slopeR = best_slopeR;
  result.best_rangeL = best_rangeL;
  result.best_rangeR = best_rangeR;
  result.best_bic = best_bic;
  result.best_rss = best_rss;
  result.best_pivot_c = best_pivot_c;
  
  return (result);
}

arma::vec PILOT::predict(const arma::mat &X) const
{
  
  arma::vec yhat(X.n_rows, arma::fill::zeros);
  
  for (arma::uword i = 0; i < X.n_rows; i++)
  {
    node *nd = root.get();
    double yhati = 0.0;
    
    while (nd->type != 0)
    {
      double x = std::clamp(X(i, nd->predId), nd->rangeL, nd->rangeR);
      if (nd->type == 1)
      { // lin
        yhati += (nd->intL + (nd->slopeL) * x);
        nd = nd->left.get();
      }
      else if (nd->type == 5)
      { // pconc
        bool isLeft = std::binary_search(nd->pivot_c.begin(), nd->pivot_c.end(), x);
        if (isLeft)
        {
          yhati += nd->intL;
          nd = nd->left.get();
        }
        else
        {
          yhati += nd->intR;
          nd = nd->right.get();
        }
      }
      else
      { // pcon/blin/plin
        if (x <= nd->splitVal)
        {
          yhati += (nd->intL + (nd->slopeL) * x);
          nd = nd->left.get();
        }
        else
        {
          yhati += (nd->intR + (nd->slopeR) * x);
          nd = nd->right.get();
        }
      }
    }
    // now at con node: still need to subtract intercept
    
    yhati += (nd->intL);
    yhat(i) = yhati;
  }
  
  return (yhat);
}



// JSON parsing

// Utility to strip leading and trailing spaces
std::string trim(const std::string &str)
{
  size_t start = str.find_first_not_of(" \t\n\r");
  size_t end = str.find_last_not_of(" \t\n\r");
  return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
}

// Utility to parse a JSON array into an Armadillo vector
arma::vec parse_json_array(const std::string &json_array)
{
  arma::vec result;
  std::stringstream ss(json_array.substr(1, json_array.size() - 2)); // Remove square brackets
  std::string item;
  
  while (std::getline(ss, item, ','))
  {
    result.insert_rows(result.n_elem, 1);              // Expand the vector
    result(result.n_elem - 1) = std::stod(trim(item)); // Convert to double
  }
  return result;
}

template <typename VecType>
std::string arma_vec_to_json(const VecType &vec)
{
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

std::string json_safe_number(double value)
{
  if (std::isnan(value))
  {
    return "null"; // JSON-friendly representation for NaN
  }
  else if (std::abs(value) < 1e-12)
  {
    return ("0");
  }
  else
  {
    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return oss.str();
    //    return std::to_string(value);
  }
}

std::string parse_json_block(const std::string &first_line, std::istream &input_stream, const std::string &key)
{
  
  std::string line = first_line;         // Start with the first line
  std::string block = first_line + "\n"; // Include the first line in the block
  int brace_depth = 0;
  
  if (line.find("null") != std::string::npos && line.find("{") == std::string::npos)
  {
    return ""; // Return empty string
  }
  
  // Validate the first line and initialize brace depth
  if (line.find("{") != std::string::npos)
  {
    brace_depth++;
  }
  else
  {
    throw std::invalid_argument("block should contain opening braces in first line");
  }
  
  while (std::getline(input_stream, line))
  {
    line = trim(line); // Trim spaces or newline characters
    
    // Update brace depth
    if (line.find("{") != std::string::npos)
    {
      brace_depth++; // Increase brace depth
    }
    if (line.find("}") != std::string::npos)
    {
      brace_depth--; // Decrease brace depth
    }
    
    block += line + "\n"; // Append the current line
    
    // Stop collecting when all braces are balanced
    if (brace_depth == 0)
    {
      break;
    }
  }
  
  return block;
}

void node::fromJson(const std::string &json_text)
{
  std::istringstream ss(json_text);
  std::string line;
  
  std::getline(ss, line); // skip first line which contains the key (="root/left/right" of the node)
  
  while (std::getline(ss, line))
  {
    line = trim(line);
    
    if (line.find("\"nodeId\":") != std::string::npos)
    {
      nodeId = std::stoul(trim(line.substr(line.find(":") + 1)));
    }
    else if (line.find("\"depth\":") != std::string::npos)
    {
      depth = std::stoul(trim(line.substr(line.find(":") + 1)));
    }
    else if (line.find("\"modelDepth\":") != std::string::npos)
    {
      modelDepth = std::stoul(trim(line.substr(line.find(":") + 1)));
    }
    else if (line.find("\"startID\":") != std::string::npos)
    {
      startID = std::stoul(trim(line.substr(line.find(":") + 1)));
    }
    else if (line.find("\"endID\":") != std::string::npos)
    {
      endID = std::stoul(trim(line.substr(line.find(":") + 1)));
    }
    else if (line.find("\"predId\":") != std::string::npos)
    {
      predId = std::stoul(trim(line.substr(line.find(":") + 1)));
    }
    else if (line.find("\"type\":") != std::string::npos)
    {
      type = std::stoul(trim(line.substr(line.find(":") + 1)));
    }
    else if (line.find("\"rss\":") != std::string::npos)
    {
      rss = std::stod(trim(line.substr(line.find(":") + 1)));
    }
    else if (line.find("\"splitVal\":") != std::string::npos)
    {
      if (line.find("null") != std::string::npos)
      {
        splitVal = arma::datum::nan;
      }
      else
      {
        splitVal = std::stod(trim(line.substr(line.find(":") + 1)));
      }
    }
    else if (line.find("\"intL\":") != std::string::npos)
    {
      if (line.find("null") != std::string::npos)
      {
        intL = arma::datum::nan;
      }
      else
      {
        intL = std::stod(trim(line.substr(line.find(":") + 1)));
      }
    }
    else if (line.find("\"slopeL\":") != std::string::npos)
    {
      if (line.find("null") != std::string::npos)
      {
        slopeL = arma::datum::nan;
      }
      else
      {
        slopeL = std::stod(trim(line.substr(line.find(":") + 1)));
      }
    }
    else if (line.find("\"intR\":") != std::string::npos)
    {
      if (line.find("null") != std::string::npos)
      {
        intR = arma::datum::nan;
      }
      else
      {
        intR = std::stod(trim(line.substr(line.find(":") + 1)));
      }
    }
    else if (line.find("\"slopeR\":") != std::string::npos)
    {
      if (line.find("null") != std::string::npos)
      {
        slopeR = arma::datum::nan;
      }
      else
      {
        slopeR = std::stod(trim(line.substr(line.find(":") + 1)));
      }
    }
    else if (line.find("\"rangeL\":") != std::string::npos)
    {
      if (line.find("null") != std::string::npos)
      {
        rangeL = arma::datum::nan;
      }
      else
      {
        rangeL = std::stod(trim(line.substr(line.find(":") + 1)));
      }
    }
    else if (line.find("\"rangeR\":") != std::string::npos)
    {
      if (line.find("null") != std::string::npos)
      {
        rangeR = arma::datum::nan;
      }
      else
      {
        rangeR = std::stod(trim(line.substr(line.find(":") + 1)));
      }
    }
    else if (line.find("\"pivot_c\":") != std::string::npos)
    {
      size_t start = line.find("[");
      size_t end = line.find("]");
      pivot_c = parse_json_array(line.substr(start, end - start + 1));
    }
    
    if (line.find("\"left\":") != std::string::npos)
    {
      std::string left_block = parse_json_block(line, ss, "\"left\":");
      
      if (!left_block.empty() && left_block != "null")
      {
        left = std::make_unique<node>();
        left->fromJson(left_block); // Recursively parse left node
      }
    }
    
    if (line.find("\"right\":") != std::string::npos)
    {
      std::string right_block = parse_json_block(line, ss, "\"right\":");
      if (!right_block.empty() && right_block != "null")
      {
        right = std::make_unique<node>();
        right->fromJson(right_block); // Recursively parse right node
      }
    }
  }
}

void PILOT::fromJson(const std::string &json_text)
{
  std::istringstream ss(json_text);
  std::string line;
  
  while (std::getline(ss, line))
  {
    line = trim(line);
    
    if (line.find("\"dfs\":") != std::string::npos)
    {
      size_t start = line.find("[");
      size_t end = line.find("]");
      dfs = parse_json_array(line.substr(start, end - start + 1));
    }
    else if (line.find("\"min_sample_leaf\":") != std::string::npos)
    {
      min_sample_leaf = std::stoul(trim(line.substr(line.find(":") + 1)));
    }
    else if (line.find("\"maxDepth\":") != std::string::npos)
    {
      maxDepth = std::stoul(trim(line.substr(line.find(":") + 1)));
    }
    else if (line.find("\"maxModelDepth\":") != std::string::npos)
    {
      maxModelDepth = std::stoul(trim(line.substr(line.find(":") + 1)));
    }
    else if (line.find("\"maxFeatures\":") != std::string::npos)
    {
      maxFeatures = std::stoul(trim(line.substr(line.find(":") + 1)));
    }
    else if (line.find("\"approx\":") != std::string::npos)
    {
      approx = std::stoul(trim(line.substr(line.find(":") + 1)));
    }
    else if (line.find("\"rel_tolerance\":") != std::string::npos)
    {
      rel_tolerance = std::stod(trim(line.substr(line.find(":") + 1)));
    }
    else if (line.find("\"precScale\":") != std::string::npos)
    {
      precScale = std::stod(trim(line.substr(line.find(":") + 1)));
    }
    else if (line.find("\"lowerBound\":") != std::string::npos)
    {
      lowerBound = std::stod(trim(line.substr(line.find(":") + 1)));
    }
    else if (line.find("\"upperBound\":") != std::string::npos)
    {
      upperBound = std::stod(trim(line.substr(line.find(":") + 1)));
    }
    else if (line.find("\"nbNodesPerModelDepth\":") != std::string::npos)
    {
      size_t start = line.find("[");
      size_t end = line.find("]");
      nbNodesPerModelDepth = arma::conv_to<arma::uvec>::from(parse_json_array(line.substr(start, end - start + 1)));
    }
    else if (line.find("\"maxNbCats\":") != std::string::npos)
    {
      maxNbCats = std::stoul(trim(line.substr(line.find(":") + 1)));
    }
    else if (line.find("\"IDvec\":") != std::string::npos)
    {
      size_t start = line.find("[");
      size_t end = line.find("]");
      IDvec = arma::conv_to<arma::uvec>::from(parse_json_array(line.substr(start, end - start + 1)));
    }
    else if (line.find("\"root\":") != std::string::npos)
    {
      std::string root_block = parse_json_block(line, ss, "\"root\":");
      
      if (!root_block.empty() && root_block != "null")
      {
        //std::cout << "start parsing root block " << std::endl;
        root = std::make_unique<node>();
        root->fromJson(root_block); // Recursively parse root node
      }
    }
  }
}

void PILOT::load_from_file(const std::string &filename)
{
  std::ifstream file(filename);
  if (!file.is_open())
  {
    throw std::runtime_error("Could not open file: " + filename);
  }
  
  // Read entire file content into a string
  std::string json_content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
  
  file.close();
  // do PILOT::fromJson
  fromJson(json_content);
  res = {arma::datum::nan};
}

std::string PILOT::toJson() const
{
  
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"dfs\": " << arma_vec_to_json(dfs) << ",\n";
  oss << "  \"min_sample_leaf\": " << min_sample_leaf << ",\n";
  oss << "  \"min_sample_alpha\": " << min_sample_alpha << ",\n";
  oss << "  \"min_sample_fit\": " << min_sample_fit << ",\n";
  oss << "  \"maxDepth\": " << maxDepth << ",\n";
  oss << "  \"maxModelDepth\": " << maxModelDepth << ",\n";
  oss << "  \"maxFeatures\": " << maxFeatures << ",\n";
  oss << "  \"approx\": " << approx << ",\n";
  oss << "  \"rel_tolerance\": " << json_safe_number(rel_tolerance) << ",\n";
  oss << "  \"precScale\": " << json_safe_number(precScale) << ",\n";
  oss << "  \"lowerBound\": " << json_safe_number(lowerBound) << ",\n";
  oss << "  \"upperBound\": " << json_safe_number(upperBound) << ",\n";
  oss << "  \"res\": [],\n";
  oss << "  \"nbNodesPerModelDepth\": " << arma_vec_to_json(nbNodesPerModelDepth) << ",\n";
  oss << "  \"maxNbCats\": " << maxNbCats << ",\n";
  oss << "  \"IDvec\": [],\n";
  //oss << "  \"IDvec\": " << arma_vec_to_json(IDvec) << ",\n";
  
  oss << "  \"root\": ";
  if (root)
  {
    oss << root->toJson(); // Serialize the node if it exists
  }
  else
  {
    oss << "null"; // Represent absence of a node
  }
  oss << "\n}";
  
  return (oss.str());
}

std::string node::toJson() const
{
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"nodeId\": " << std::to_string(nodeId) << ",\n";
  oss << "  \"depth\": " << std::to_string(depth) << ",\n";
  oss << "  \"modelDepth\": " << std::to_string(modelDepth) << ",\n";
  oss << "  \"startID\": " << std::to_string(startID) << ",\n";
  oss << "  \"endID\": " << std::to_string(endID) << ",\n";
  oss << "  \"predId\": " << std::to_string(predId) << ",\n";
  oss << "  \"type\": " << std::to_string(type) << ",\n";
  oss << "  \"rss\": " << std::to_string(rss) << ",\n";
  oss << "  \"splitVal\": " << json_safe_number(splitVal) << ",\n";
  oss << "  \"intL\": " << json_safe_number(intL) << ",\n";
  oss << "  \"slopeL\": " << json_safe_number(slopeL) << ",\n";
  oss << "  \"intR\": " << json_safe_number(intR) << ",\n";
  oss << "  \"slopeR\": " << json_safe_number(slopeR) << ",\n";
  oss << "  \"rangeL\": " << json_safe_number(rangeL) << ",\n";
  oss << "  \"rangeR\": " << json_safe_number(rangeR) << ",\n";
  if (pivot_c.n_elem > 0)
  {
    oss << "  \"pivot_c\": " << arma_vec_to_json(pivot_c) << ",\n";
  }
  else
  {
    oss << "  \"pivot_c\": [],\n";
  }
  
  // Serialize left child
  oss << "  \"left\": ";
  if (left)
  {
    oss << left->toJson();
  }
  else
  {
    oss << "null";
  }
  oss << ",\n";
  
  // Serialize right child
  oss << "  \"right\": ";
  if (right)
  {
    oss << right->toJson();
  }
  else
  {
    oss << "null";
  }
  oss << "\n}";
  
  return oss.str();
}
