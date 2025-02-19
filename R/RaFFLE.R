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
#'   \item \code{modelmat}: A matrix describing the fitted model 
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
#' X <- as.matrix(data[, 2:4])
#' raffle.out <- raffle(X, y)
#' # plot residuals
#' plot(raffle.out$residuals)
#' # generate predictions in-sample
#' preds.out <- predict(raffle.out, newdata = X)
#' plot(raffle.out, y); abline(0, 1)
#' # print model matrix of the first tree
#' round(raffle.out$modelmat[, , 1], 3)
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
  
  if (!is.matrix(X)) stop("X must be a matrix.")
  if (!is.vector(y) && !is.numeric(y)) stop("y must be a numeric vector.")
  if (nrow(X) != length(y)) stop("Number of rows in X must match the length of y.")
  if (anyNA(X)) stop("raffle cannot handle NA values in X (for now).")
  if (anyNA(y)) stop("raffle cannot handle NA values in y (for now).")
  
  data_names <- colnames(X)
  X <- as.matrix(X)
  y <- as.vector(y)
  
  catIDs <- apply(X, 2,  function(col) is.factor(col) || is.character(col)) + 0.0
  
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
  fo$train(X, y, catIDs)
  
  modelmat <- fo$print()
  modelmat[, 5, ] <- modelmat[, 5, ] + 1 # C++ to R indexing
  dimnames(modelmat)[[2]] <- c("depth", "modeldepth", "nodeId", "node type", "feature index",
                               "split value", "left intercept", "left slope", "right intercept", 
                               "right slope")
  # return output as a RAFFLE class
  output <- list(modelmat = modelmat,
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
                 jsonString = fo$toJson)
  class(output) <- "RAFFLE" 
  
  return(output)
}





#' Print a RAFFLE model
#'
#' Print a RAFFLE model
#' @param x an object of the RAFFLE class
#' @param ... other print parameters 
#' @examples
#' x <- rnorm(10)
#' 

print.RAFFLE <- function(x, ...) {
  if (!inherits(x, "RAFFLE")) {
    stop("Object is not of class 'RAFFLE'")
  }
  if (!is.array(x$modelmat)) {
    stop("RAFFLE object is corrupted: does not have a model array")
  }
  print(x$modelmat)
}


#' Plot a RAFFLE model
#'
#' Plot a RAFFLE model
#' @param x an object of the RAFFLE class
#' @param ... other graphical parameters 
#' @examples
#' x <- rnorm(10)
#' 

plot.RAFFLE <- function(x, ...) {
  if (!inherits(x, "RAFFLE")) {
    stop("Object is not of class 'RAFFLE'")
  }
  
  plot(x$residuals, xlab = "index", ylab = "residuals", ...)
}



#' Predict with a RAFFLE model
#'
#' Predict with a RAFFLE model
#' @param x an object of the RAFFLE class
#' @param newdata a matrix or data frame with new data
#' @param maxDepth predict using all nodes of depth up to maxDepth. If NULL, predict using full tree.
#' @examples
#' x <- rnorm(10)
#' 

predict.RAFFLE <- function(x, newdata, maxDepth = NULL) {
  
  if (!inherits(x, "RAFFLE")) {
    stop("Object is not of class 'RAFFLE'")
  }
  # first check if the object was loaded from a file. If so, reconstruct it:
  if (capture.output(x$modelpointer[[".module"]])  == "<pointer: (nil)>") {
    fo <- new(RAFFLEcpp)
    fo$fromJSON(x$jsonString) # read the PILOt object from the Json string
    x$modelpointer = fo
  } else {
    fo <- x$modelpointer
  }
  
  # now check new input newdata, mainly the categorical features
  newdata <- as.matrix(newdata)
  
  if (anyNA(newdata)) stop("RAFFLE cannot handle NA values in new data (for now).")
  
  if (!isTRUE(all.equal(colnames(newdata), x$data_names))){
    stop("Column names of new data do not match the training data.")
  }
  
  catIDs <- apply(X, 2,  function(col) is.factor(col) || is.character(col)) + 0.0
  if (!isTRUE(all.equal(catIDs, x$catInfo$catIDs))) {
    stop("Categorical/factor variables of new data do not match those in the training data.")
  }
  
  if (any(catIDs == 1)) {
    catInds <- x$catInfo$catInds 
    for (j in 1:length(catInds)) {
      catID        <- catInds[j]
      factorlevels <- levels(as.factor(newdata[, catID]))
      if (length(setdiff(factorlevels, x$catInfo$factorlevels[[j]])) > 0) {
        stop(paste0("Variable ", catID, " has categories not present in the training data."))
      }
      xf <- factor(newdata[, catID], levels = x$catInfo$factorlevels[[j]])
      X[, catID]   <- as.integer(xf) - 1
    }
  }
  
  if (is.null(maxDepth)) {
    maxDepth = x$parameters$maxDepth
  }
  
  preds <- fo$predict(newdata, maxDepth)
  return(preds)
}



