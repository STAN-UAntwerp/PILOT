#include "tree.h"
#include <iostream>

// constructors

PILOT::PILOT(const arma::vec &dfs,
             const arma::uword &min_sample_leaf,
             const arma::uword &min_sample_alpha,
             const arma::uword &min_sample_fit,
             const arma::uword &maxDepth,
             const arma::uword &maxModelDepth,
             const arma::uword &maxFeatures,
             const arma::uword &approx,
             const double &rel_tolerance,
             const double &precScale) : dfs(dfs),
                                        min_sample_leaf(min_sample_leaf),
                                        min_sample_alpha(min_sample_alpha),
                                        min_sample_fit(min_sample_fit),
                                        maxDepth(maxDepth),
                                        maxModelDepth(maxModelDepth),
                                        maxFeatures(maxFeatures),
                                        approx(approx),
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
  arma::umat Xrank(X.n_rows, X.n_cols, arma::fill::zeros);
  for (arma::uword i = 0; i < X.n_cols; i++)
  { // O(d n\log(n))
    arma::uvec xorder = arma::sort_index(X.col(i));
    arma::uvec xrank(X.n_rows, arma::fill::zeros);
    xrank(xorder) = arma::regspace<arma::uvec>(0, X.n_rows - 1);
    Xrank.col(i) = xrank;
  }
  // build root node
  root = std::make_unique<node>();
  root->obsIds = arma::regspace<arma::uvec>(0, 1, X.n_rows - 1);
  root->depth = 0;
  root->modelDepth = 0;
  root->nodeId = 0;

  // build tree
  res = y; // initialize res
  double maxy = arma::max(y);
  double miny = arma::min(y);
  lowerBound = miny;                                                 // 5 * miny / 4 - maxy / 4; // min(y) - range(y) / 4
  upperBound = maxy;                                                 // 5 * maxy / 4 - miny / 4; // max(y) + range(y) / 4
  nbNodesPerModelDepth = arma::zeros<arma::uvec>(maxModelDepth + 1); // initialize nodes per model depth
  nbNodesPerModelDepth(0) = 1;                                       // 1 root node at depth 0

  PILOT::growTree(root.get(), y, X, Xrank, catIds);
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
  if (nd->obsIds.n_elem <= min_sample_fit)
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
    nd->intL = arma::mean(res(nd->obsIds));
    nd->slopeL = arma::datum::nan;
    nd->splitVal = arma::datum::nan;
    nd->predId = arma::datum::nan;
    nd->intR = arma::datum::nan;
    nd->slopeR = arma::datum::nan;
    nd->rangeL = arma::datum::nan;
    nd->rangeR = arma::datum::nan;
    res(nd->obsIds) -= nd->intL; // subtract mean
    nd->rss = arma::sum(arma::square(res(nd->obsIds)));
    // set type to con, and no new call to growtree
    nd->type = 0;
  }
  else
  {
    bestSplitOut newSplit = findBestSplit(nd->obsIds,
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
    { 
      // con is best split
      // update res and rss
      res(nd->obsIds) -= newSplit.best_intL; // subtract mean

      // no new call to growtree
    }
    else if (newSplit.best_type == 1)
    { 
      // lin is best split

      arma::vec x = X.col(newSplit.best_feature);
      x = x(nd->obsIds);
      // update res
      double rss_old = arma::sum(arma::square(res(nd->obsIds)));
      res(nd->obsIds) = res(nd->obsIds) - newSplit.best_intL - newSplit.best_slopeL * x;
      // truncate prediction
      // we clip the overall predictions (=y - current residuals) between lowerBound and upperBound

      res(nd->obsIds) = y(nd->obsIds) - arma::clamp(y(nd->obsIds) - res(nd->obsIds), lowerBound, upperBound);

      double rss_new = arma::sum(arma::square(res(nd->obsIds)));

      if ((rss_old - rss_new) / rss_old < rel_tolerance)
      {
        nd->intL = arma::mean(res(nd->obsIds));
        nd->slopeL = arma::datum::nan;
        nd->splitVal = arma::datum::nan;
        nd->predId = arma::datum::nan;
        nd->intR = arma::datum::nan;
        nd->slopeR = arma::datum::nan;
        nd->rangeL = arma::datum::nan;
        nd->rangeR = arma::datum::nan;
        res(nd->obsIds) -= nd->intL; // subtract mean
        nd->rss = arma::sum(arma::square(res(nd->obsIds)));
        // set type to con, and no new call to growtree
        nd->type = 0;
      }
      else
      {
        // construct a new node (left node only here)
        nd->left = std::make_unique<node>();

        nd->left->obsIds = nd->obsIds;
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
    { // split on categorical variable (pconc)

      arma::uvec indL_local = newSplit.ind_local_c;
      arma::uvec indR_local = arma::regspace<arma::uvec>(0, (nd->obsIds).n_elem - 1);
      indR_local.shed_rows(indL_local);

      arma::uvec indL = (nd->obsIds).elem(indL_local);
      arma::uvec indR = (nd->obsIds).elem(indR_local);

      res(indL) = res(indL) - newSplit.best_intL;
      res(indR) = res(indR) - newSplit.best_intR;

      // construct new nodes
      nd->left = std::make_unique<node>();
      nd->right = std::make_unique<node>();

      nd->left->obsIds = indL;
      nd->right->obsIds = indR;
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

      arma::vec x = X.col(newSplit.best_feature);
      x = x(nd->obsIds);

      arma::uvec indL_local = arma::find(x <= newSplit.best_splitVal);
      arma::uvec indR_local = arma::find(x > newSplit.best_splitVal);
      arma::uvec indL = (nd->obsIds).elem(indL_local);
      arma::uvec indR = (nd->obsIds).elem(indR_local);

      // compute the raw  residuals

      res(indL) = res(indL) - newSplit.best_intL - newSplit.best_slopeL * x(indL_local);
      res(indR) = res(indR) - newSplit.best_intR - newSplit.best_slopeR * x(indR_local);

      // truncate the residuals
      res(indL) = y(indL) - arma::clamp(y(indL) - res(indL), lowerBound, upperBound);
      res(indR) = y(indR) - arma::clamp(y(indR) - res(indR), lowerBound, upperBound);

      // construct new nodes
      nd->left = std::make_unique<node>();
      nd->right = std::make_unique<node>();

      nd->left->obsIds = indL;
      nd->right->obsIds = indR;
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

bestSplitOut PILOT::findBestSplit(
    arma::uvec &obsIds, // observations currently in this node
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
  arma::uvec ind_local_c; // An array of the indices of the best categorical variable x(obsIds) with levels in pivot_c
  ind_local_c.reset();

  const arma::uword d = X.n_cols;
  const double n = (double)obsIds.n_elem; // double since we use it in many double calculations
  const double logn = std::log(n);
  // first compute con as benchmark

  arma::vec y = res(obsIds);
  double sumy = arma::sum(y);
  double sumy2 = arma::sum(arma::square(y));

  // if (dfs(0) >= 0) { // check if con model is allowed. Should always be the case
  double intercept = sumy / n;
  double rss = sumy2 + intercept * intercept * n - 2 * intercept * sumy;
  double bic = n * (std::log(rss) - logn) + logn * dfs(0);

  best_intL = intercept;
  best_slopeL = arma::datum::nan;
  best_bic = bic;
  best_rss = rss;
  //}// end of con model

  // now check the features whether model building is worth it.
  // declare variables
  arma::uvec xorder(obsIds.n_elem);
  arma::vec xs(obsIds.n_elem);
  arma::vec ys(obsIds.n_elem);
  arma::vec xunique;
  arma::vec::fixed<3> u;
  arma::vec sums;
  arma::uvec counts;
  arma::vec pivot_c;
  arma::vec means;
  arma::uvec means_order;
  arma::vec sums_s;
  arma::uvec counts_s;
  arma::uvec cumcounts;
  arma::vec yR;
  arma::uvec sub_index;
  arma::uvec new_index;

  double sumx, sumx2, sumxy, nL, nR, sumxL, sumx2L, sumyL, sumy2L, sumxyL, sumxR, sumx2R, sumyR, sumy2R, sumxyR, slopeL, intL, slopeR, intR, varL, varR, splitVal_old, splitVal, delta, xdiff, x2diff, ydiff, y2diff, xydiff, sumL, sumR;
  ;
  double *last;
  arma::u64 *lastuint;
  arma::mat::fixed<3, 3> XtX;
  arma::vec::fixed<3> XtY;
  arma::uvec splitCandidates(obsIds.n_elem);
  arma::uword splitID, nsteps, mini, nSplitCandidates, k, startID, stopID, nbUnique, ii, stepSize;

  bool sortLocally = false;
  bool approxed;

  if (n < 500000)
  {
    sortLocally = true;
  }
  else
  {
    xorder.set_size(X.n_rows);
  }

  arma::uvec featuresToConsider = arma::randperm(d, maxFeatures);

  for (arma::uword j : featuresToConsider)
  { // iterate over features

    // unsorted variables
    // x = X.submat(obsIds, arma::uvec{j});

    // get sorted variables.
    if (sortLocally)
    {                                                             // if n is small, it is faster to sort locally.
      xorder = arma::sort_index(X.submat(obsIds, arma::uvec{j})); // local order
      xs = X.submat(obsIds(xorder), arma::uvec{j});
      ys = res(obsIds(xorder));
    }
    else
    { // This O(n), but can be slow if obsIds.n_elem is small and X.n_rows is large, due to copying large vectors

      xorder.fill(X.n_rows); // fill with an out of bound value
      xorder(Xrank.submat(obsIds, arma::uvec{j})) = arma::regspace<arma::uvec>(0, obsIds.n_elem - 1);

      arma::uword counter = 0;
      for (arma::uword i = 0; i < xorder.n_elem; i++)
      {
        if (xorder(i) != X.n_rows)
        {
          xs(counter) = X(obsIds(xorder(i)), j);
          ys(counter) = res(obsIds(xorder(i)));
          xorder(counter) = xorder(i); // this is needed for categorical variables, the first obsIds.n_elem elements now contain the order of x = X(obsIds, j)
          counter++;
        }
      }
    }

    // non-categorical variables
    if (catIds(j) == 0)
    { // check whether numerical feature

      // compute moments

      sumx = arma::sum(xs);
      sumx2 = arma::sum(arma::square(xs));
      sumxy = arma::sum(xs % ys);

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

      if ((approx > 0) && (n > approx)) {
        stepSize = std::round(n / (double)approx);
        approxed = true;
      }

      nSplitCandidates = 0;
      for (arma::uword i = 0; i < n; i += stepSize)
      { // xs(i) is the new splitcandidate
        ii = 0;
        while((i+(ii) < n) && (xs(i+ii) - xs(i) < precScale) ){ // counter the number of occurences of x(i)
          ii++;
        }
        splitCandidates(nSplitCandidates) = i + (ii - 1);
        nSplitCandidates++;

        if (ii > 1) { // shift the iterator in case of duplicates
          i = i + (ii - 1);
        }        
      }

      for (arma::uword i = 0; i < nSplitCandidates - 1; i++)
      { // iterate over all candidate split points

        splitID = splitCandidates(i);
        splitVal = xs(splitID);

        // count number of skipped steps from the previous splitVal 
        // This can also be > 1 when binning is used (i.e. approx > 0)
        if (i == 0) {
          nsteps = splitID + 1;
        } else {
          nsteps = splitID - splitCandidates(i-1);
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
          y2diff = arma::sum(ys.subvec(mini, splitID) % ys.subvec(mini, splitID));
          if (approxed) {
          xdiff = arma::sum(xs.subvec(mini, splitID));
          x2diff = arma::sum(xs.subvec(mini, splitID) % xs.subvec(mini, splitID));
          xydiff = arma::sum(xs.subvec(mini, splitID) % ys.subvec(mini, splitID)); 
          } else { // in this case, nsteps only counts the number of duplicate values, and we can update xdiff and x2diff efficiently.
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

          rss = sumy2L + nL * intL * intL +
                (2 * slopeL * intL * sumxL) + (slopeL * slopeL * sumx2L) -
                2 * intL * sumyL - 2 * slopeL * sumxyL +
                sumy2R + nR * intR * intR +
                (2 * slopeR * intR * sumxR) + (slopeR * slopeR * sumx2R) -
                2 * intR * sumyR - 2 * slopeR * sumxyR;

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
    { // categorical variables

      xunique = xs;
      last = std::unique(xunique.begin(), xunique.end());
      if (xunique.end() - last > 0)
      { // check if duplicates found
        xunique.shed_rows(xs.n_elem - (xunique.end() - last), xs.n_elem - 1);
      }

      if (xunique.n_elem >= 2)
      {
        // xorder = xorder.head(xs.n_elem); // is this necessary?
        //  at least two unique predictor variables needed for pcon
        k = arma::conv_to<arma::uword>::from(xunique.tail(1)) + 1;
        sums.zeros(k);
        counts.zeros(k);
        pivot_c.set_size(k);
        pivot_c.fill((double)k);

        // xunique contains the unique elements in ascending order
        // we want the counts per unique element in the same order
        for (arma::uword i = 0; i < xs.n_elem; i++)
        {
          sums(xs(i)) += ys(i);
          counts(xs(i))++;
        }
        sums = sums.elem(arma::conv_to<arma::uvec>::from(xunique));
        counts = counts.elem(arma::conv_to<arma::uvec>::from(xunique));

        means = sums / counts;
        // # sort unique values w.r.t. the mean of the responses

        means_order = arma::sort_index(means);
        sums_s = sums(means_order);
        counts_s = counts(means_order);
        cumcounts = arma::cumsum(counts);
        // loop over the sorted possible_p and find the best partition
        sub_index;
        sub_index.reset();

        sumL = 0;
        sumR = arma::sum(y);
        nL = 0;
        nR = y.n_elem;
        for (arma::uword i = 0; i < xunique.n_elem - 1; i++)
        {
          nL += counts_s(i);
          nR -= counts_s(i);
          sumL += sums_s(i);
          sumR -= sums_s(i);

          // now need to insert this so that pivot_c is sorted in the end

          pivot_c(xunique(means_order(i))) = xunique(means_order(i));
          startID = (means_order(i) == 0) ? 0 : cumcounts(means_order(i) - 1);
          stopID = startID + counts(means_order(i)) - 1;
          // std::cout << "startID: " << startID << " stopID: " << stopID << std::endl;
          new_index = xorder.subvec(startID, stopID);
          sub_index = arma::join_cols(sub_index, new_index); // indices of (local!) x with x(idx) \in means_order(0,i)

          // here should check that nL and nR are at least min_leaf etc.
          if (nL < min_sample_leaf)
          {
            continue; // skip to next splitpoînt
          }
          else if (nR < min_sample_leaf)
          {
            break; // all splitpoints have been considered for this variable
          }

          yR = y;
          yR.shed_rows(sub_index);

          rss = arma::sum(arma::square(y.elem(sub_index) - sumL / nL)) +
                arma::sum(arma::square(yR - sumR / nR));
          bic = n * (std::log(rss) - logn) + logn * dfs(5);

          // update if better
          if (bic < best_bic)
          {
            best_feature = j;
            best_type = 5; // node type 5 for pconc
            best_rangeL = xs.front();
            best_rangeR = xs.back();
            best_intL = sumL / nL;
            best_intR = sumR / nR;
            best_bic = bic;
            best_rss = rss;
            best_pivot_c = pivot_c(arma::find(pivot_c < k));
            ind_local_c = sub_index;
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
  result.ind_local_c = ind_local_c;

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
      // y-truncation
      yhati = std::clamp(yhati, lowerBound, upperBound);
    }
    // now at con node: still need to subtract intercept

    yhati += (nd->intL);
    yhati = std::clamp(yhati, lowerBound, upperBound);
    yhat(i) = yhati;
  }

  return (yhat);
}
