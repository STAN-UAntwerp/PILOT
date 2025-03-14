#' Random Forest Featuring Linear Extensions (RaFFLE)
#'
#' Runs RaFFLE to fit a forest of piecewise linear trees. 
#' @param X an \eqn{n \times d} set of training data
#' @param y a vector of length \eqn{n} with the training response
#' @param nTrees the number of trees in the forest
#' @param alpha a value between 0 and 1 determining the "degrees of freedom" of the models con/lin/pcon/blin/plin/pconc used in the individual PILOT trees. See details below.
#' @param min_sample_leaf a positive integer. Do not consider splits which result in less than \code{min_sample_leaf} in a leaf node.
#' @param min_sample_alpha a positive integer. Do not consider splits which result in less than \code{min_sample_alpha} in a leaf node.
#' @param min_sample_fit a positive integer. Stop splitting the tree in a given node once no more than \code{min_sample_fit} observations are left in that node.
#' @param maxDepth an integer specifying the maximum depth of the tree. The depth of a tree is incremented when pcon/blin/plin/pconc models are fit.
#' @param maxModelDepth an integer specifying the maximum model depth of the tree. Model depth is incremented when any model is fit, including lin models.
#' @param n_features_node the number of features randomly sampled at each node for fitting the next model.
#' @param rel_tolerance a lin model is only fit when the relative RSS is decreased by \code{rel_tolerance}.
#' @return an object of the \code{RAFFLE} class, i.e. a list with the following components:
#' \itemize{
#'   \item \code{modelcube}: A 3-dimensional array describing the fitted model in raw form. Used for plotting and printing the individual trees.
#'   \item \code{residuals}: the residuals of the RAFFLE model on the training data
#'   \item \code{parameters}: A list containing the parameters used for fitting the model
#'   \item \code{data_names}: A vector with the column names of the training data.  Used for out-of-sample prediction.
#'   \item \code{catInfo}: A list containing information on the categorical variables in the data. Used for out-of-sample prediction.
#'   \item \code{modelpointer}: A pointer to the C++ object from the C++ class RAFFLE 
#'   \item \code{jsonString}: A string describing the fitted  model
#' }
#' @details \code{alpha} can be considered as a regularization parameter. High values of \code{alpha} makes splitting nodes more costly.
#' @examples
#' data <- iris
#' y <- as.vector(data[, 1])
#' X <- as.data.frame(data[, 2:4])
#' raffle.out <- raffle(X, y, maxDepth = 3)
#' # plot residuals
#' plot(raffle.out$residuals)
#' # generate predictions in-sample
#' preds.out <- predict(raffle.out, newdata = X)
#' plot(preds.out, y)
#' # plot the first tree in the ensemble
#' plot(raffle.out, treeNb = 1)
#' # print model matrix of the first tree
#'  print(raffle.out, treeNb = 1)
#' 


raffle <- function(X, y,
                   nTrees = 50,
                   alpha = 0.5, 
                   min_sample_leaf = 5, 
                   min_sample_alpha = 5,
                   min_sample_fit = 10,
                   maxDepth = 20,
                   maxModelDepth = 100,
                   n_features_node = 1,
                   rel_tolerance = 1e-2) {
  
  if (!is.data.frame(X)) stop("X must be a data frame")
  if (!is.vector(y) && !is.numeric(y)) stop("y must be a numeric vector.")
  if (nrow(X) != length(y)) stop("Number of rows in X must match the length of y.")
  if (anyNA(X)) stop("raffle cannot handle NA values in X (for now).")
  if (anyNA(y)) stop("raffle cannot handle NA values in y (for now).")
  
  data_names <- colnames(X)
  X <- as.data.frame(X)
  output <- list(df = X,
                 response = y)
  y <- as.vector(y)
  
  catIDs <- sapply(X, function(col) is.factor(col) || is.character(col))+0.0
  
  catInfo <- list(catIDs = catIDs)
  # now convert categorical into integers 0, ..., 1
  if (any(catIDs == 1)) {
    catInds <- which(catIDs == 1)
    for (j in 1:length(catInds)) {
      catID        <- catInds[j]
      factorlevels <- levels(as.factor(X[, catID]))
      X[, catID]   <- as.integer(as.factor(X[, catID])) - 1
      catInfo$factorlevels[[j]] <- factorlevels
    }
    catInfo$catInds = catInds
  }
  
  
  dfs    <- 1 + alpha * (c(1, 2, 5, 5, 7, 5) - 1)
  dfs[4] <- -1 # disable blin
  
  modelParams <- c(min_sample_leaf,  
                   min_sample_alpha,
                   min_sample_fit,
                   maxDepth,
                   maxModelDepth,
                   round(n_features_node * ncol(X)),
                   0)
  fo <- new(RAFFLEcpp,
            nTrees = nTrees,
            dfs = dfs,
            modelParams = modelParams,
            rel_tolerance = rel_tolerance,
            precScale = 1e-10)
  X <- as.matrix(X)
  fo$train(X, y, catIDs)
  
  modelcube <- fo$print()
  # return output as a RAFFLE class
  output <- c(output, 
              list(modelcube = modelcube,
                   residuals = fo$getResiduals(X, y, maxDepth),
                   parameters = list(nTrees = nTrees, 
                                     alpha = alpha, 
                                     min_sample_leaf = min_sample_leaf, 
                                     min_sample_alpha = min_sample_alpha,
                                     min_sample_fit = min_sample_fit,
                                     maxDepth = maxDepth,
                                     maxModelDepth = maxModelDepth,
                                     n_features_node = n_features_node,
                                     rel_tolerance = rel_tolerance),
                   data_names = data_names,
                   catInfo = catInfo,
                   modelpointer = fo, 
                   jsonString = fo$toJson(0)))
  class(output) <- "RAFFLE" 
  
  return(output)
}





#' Print a RAFFLE model
#'
#' Print a given tree in the RAFFLE model
#' @param x an object of the RAFFLE class
#' @param treeNb the tree that is printed. defaults to the first tree
#' @param ... other print parameters 
#' @examples
#' data <- iris
#' y <- as.vector(data[, 1])
#' X <- as.data.frame(data[, 2:4])
#' raffle.out <- raffle(X, y, maxDepth = 3)
#'  print(raffle.out, treeNb = 1)
#'  print(raffle.out, treeNb = 2)


print.RAFFLE <- function(x, treeNb = 1, ...) {
  if (!inherits(x, "RAFFLE")) {
    stop("Object is not of class 'RAFFLE'")
  }
  if (!is.array(x$modelcube)) {
    stop("RAFFLE object is corrupted: does not have a model array")
  }
  if (!(treeNb %in% 1:dim(x$modelcube)[3])) {
    stop(paste0("treeNb should be an integer between 1 and ", dim(x$modelcube)[3], " to select which tree to print."))
  }
  
  individualTree <- extractTree(x, treeNb)
  print(individualTree)
}


#' Plot a RAFFLE model
#'
#' Plot a RAFFLE model
#' @param x an object of the RAFFLE class
#' @param treeNb the tree that is printed. defaults to the first tree
#' @param infoType If 0, prints model coefficients in leaf nodes. If 1, prints variable importance. Defaults to 0
#' @param ... other graphical parameters 
#' @examples
#' data <- iris
#' y <- as.vector(data[, 1])
#' X <- as.data.frame(data[, 2:4])
#' raffle.out <- raffle(X, y, maxDepth = 3)
#' # plot the first tree in the ensemble
#' plot(raffle.out, treeNb = 1)
#' plot(raffle.out, treeNb = 1, infoType = 1)

plot.RAFFLE <- function(x, treeNb = 1, infoType = 0, ...) {
  if (!inherits(x, "RAFFLE")) {
    stop("Object is not of class 'RAFFLE'")
  }
  if (!(treeNb %in% 1:dim(x$modelcube)[3])) {
    stop(paste0("treeNb should be an integer between 1 and ",
                dim(x$modelcube)[3], " to select which tree to print."))
  }
  
  individualTree <- extractTree(x, treeNb)
  plot(individualTree, infoType = infoType, ...)
}



#' Predict with a RAFFLE model
#'
#' Predict with a RAFFLE model
#' @param object an object of the RAFFLE class
#' @param newdata a matrix or data frame with new data
#' @param maxDepth predict using all nodes of depth up to maxDepth. If NULL, predict using full tree.
#' @examples
#' data <- iris
#' y <- as.vector(data[, 1])
#' X <- as.data.frame(data[, 2:4])
#' raffle.out <- raffle(X, y, maxDepth = 3)
#' # generate predictions in-sample
#' preds.out <- predict(raffle.out, newdata = X)
#' plot(preds.out, y)
#' 
predict.RAFFLE <- function(object, newdata, maxDepth = NULL) {
  
  if (!inherits(object, "RAFFLE")) {
    stop("Object is not of class 'RAFFLE'")
  }
  # first check if the object was loaded from a file. If so, reconstruct it:
  if (capture.output(object$modelpointer[[".module"]])  == "<pointer: (nil)>") {
    fo <- new(RAFFLEcpp)
    fo$fromJson(object$jsonString) # read the PILOt object from the Json string
    object$modelpointer = fo
  } else {
    fo <- object$modelpointer
  }
  
  # now check new input newdata, mainly the categorical features
  if (is.null(newdata)) {newdata <- object$df}
  
  if (anyNA(newdata)) stop("RAFFLE cannot handle NA values in new data (for now).")
  
  if (!isTRUE(all.equal(colnames(newdata), object$data_names))){
    stop("Column names of new data do not match the training data.")
  }
  
  catIDs <-  sapply(newdata, function(col) is.factor(col) || is.character(col))+0.0
  if (!isTRUE(all.equal(catIDs, object$catInfo$catIDs))) {
    stop("Categorical/factor variables of new data do not match those in the training data.")
  }
  
  if (any(catIDs == 1)) {
    catInds <- object$catInfo$catInds 
    for (j in 1:length(catInds)) {
      catID        <- catInds[j]
      factorlevels <- levels(as.factor(newdata[, catID]))
      if (length(setdiff(factorlevels, object$catInfo$factorlevels[[j]])) > 0) {
        stop(paste0("Variable ", catID, " has categories not present in the training data."))
      }
      xf <- factor(newdata[, catID], levels = object$catInfo$factorlevels[[j]])
      newdata[, catID]   <- as.integer(xf) - 1
    }
  }
  
  if (is.null(maxDepth)) {
    maxDepth = object$parameters$maxDepth
  }
  
  newdata <- as.matrix(newdata)
  preds <- fo$predict(newdata, maxDepth)
  return(preds)
}


extractTree <- function(x, treeNb) {
  # extract a given tree from a raffle forest, and
  # put it in PILOT s3 format
  
  
  if (!inherits(x, "RAFFLE")) {
    stop("Object is not of class 'RAFFLE'")
  }
  if (!(treeNb %in% 1:dim(x$modelcube)[3])) {
    stop(paste0("treeNb should be an integer between 1 and ",
                dim(x$modelcube)[3], " to select which tree to print."))
  }
  
  Jsontree <- x$modelpointer$toJson(treeNb) # extract json string for tree number treeNb
  
  # now construct a new pointer to the extracted tree.
  tr <- new(PILOTcpp)
  tr$fromJson(Jsontree) # read the PILOt object from the Json string
  
  prepare_modelmat.out <- prepare_modelmat(x$modelcube[, , treeNb], ncol(x$df))
  
  # return output as a PILOT class
  individualpilot <- list(df = x$df,
                          response = x$response,
                          modelmat = prepare_modelmat.out$modelmat,
                          leafnodemodels = prepare_modelmat.out$leafnodemodels,
                          nodeIdMap = prepare_modelmat.out$nodeIdMap,
                          residuals = NULL,
                          parameters = x$parameters[c("dfs", "min_sample_leaf", "min_sample_alpha", "min_sample_fit", 
                                                      "maxDepth", "maxModelDepth", "rel_tolerance")],
                          data_names = x$data_names,
                          catInfo = x$catInfo,
                          modelpointer = tr, 
                          jsonString = Jsontree)
  
  class(individualpilot) <- "PILOT" 
  
  
  predictions <- predict(individualpilot, newdata = x$df, NULL, 0)
  residuals <- x$response - predictions
  individualpilot$residuals <- residuals
  
  return(individualpilot)
}

