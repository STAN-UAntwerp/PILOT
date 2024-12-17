#' PIecewise LInear Organic Tree
#'
#' Fits a linear model tree using the PILOT algorithm. PILOT builds a tree in a greedy way, much like CART, but incorporates linear models resulting in a linear model tree.
#' @param X an \eqn{n \times d} set of training data
#' @param y a vector of length \eqn{n} with the training response
#' @param dfs a vector of length 6 indicating the "degrees of freedom" given to each of the models con/lin/pcon/blin/plin/pconc. A negative value results in the algorithm not considering this time of model while building the tree.
#' @param min_sample_leaf a positive integer. Do not consider splits which result in less than \code{min_sample_leaf} in a leaf node.
#' @param min_sample_alpha a positive integer. Do not consider splits which result in less than \code{min_sample_alpha} in a leaf node.
#' @param min_sample_fit a positive integer. Stop splitting the tree in a given node once no more than \code{min_sample_fit} observations are left in that node.
#' @param maxDepth an integer specifying the maximum depth of the tree. The depth of a tree is incremented when pcon/blin/plin/pconc models are fit.
#' @param maxModelDepth an integer specifying the maximm model depth of the tree. Model depth is incremented when any model is fit, including lin models.
#' @param rel_tolerance a lin model is only fit when the relative RSS is decreased by \code{rel_tolerance}.
#' @return an object of the \code{PILOT} class, i.e. a list with the following components:
#' \itemize{
#'   \item \code{modelmat}: A matrix describing the fitted linear model tree.
#'   \item \code{residuals}: the residuals of the PILOT model on the training data
#'   \item \code{parameters}: A list containing the parameters used for fitting the model
#'   \item \code{data_names}: A vector with the column names of the training data.  Used for out-of-sample prediction.
#'   \item \code{catInfo}: A list containing information on the categorical variables in the data. Used for out-of-sample prediction.
#'   \item \code{modelpointer}: A pointer to the C++ object from the C++ class PILOT. 
#'   \item \code{jsonString}: A string describing the fitted linear model tree
#' }
#' @examples
#' data <- iris
#' y <- as.vector(data[, 1])
#' X <- as.matrix(data[, 2:4])
#' dfs <- c(1, 2, 5, 5, 7, 5)
#' min_sample_leaf <- 5
#' min_sample_alpha <- 10
#' min_sample_fit <- 5
#' maxDepth <- 10
#' maxModelDepth <- 100
#' rel_tolerance <- 1e-2
#' pilot.out <- pilot(X, y, dfs, min_sample_leaf, min_sample_alpha,
#' min_sample_fit, maxDepth, maxModelDepth, rel_tolerance)
#' # plot residuals
#' plot(pilot.out$residuals)
#' # generate predictions in-sample
#' preds.out <- predict(pilot.out, newdata = X)
#' plot(preds.out, y); abline(0, 1)
#' # print model matrix
#' round(pilot.out$modelmat, 3)
#' 


pilot <- function(X, y, dfs, 
                  min_sample_leaf, 
                  min_sample_alpha,
                  min_sample_fit,
                  maxDepth,
                  maxModelDepth,
                  rel_tolerance = 1e-2) {
  
  if (!is.matrix(X)) stop("X must be a matrix.")
  if (!is.vector(y) && !is.numeric(y)) stop("y must be a numeric vector.")
  if (nrow(X) != length(y)) stop("Number of rows in X must match the length of y.")
  if (anyNA(X)) stop("PILOT cannot handle NA values in X (for now).")
  if (anyNA(y)) stop("PILOT cannot handle NA values in y (for now).")
  
  
  
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
  
  
  
  modelParams <- c(min_sample_leaf,  
                   min_sample_alpha,
                   min_sample_fit,
                   maxDepth,
                   maxModelDepth,
                   ncol(X),
                   0)
  tr <- new(PILOTcpp,
            dfs = dfs,
            modelParams = modelParams,
            rel_tolerance = rel_tolerance,
            precScale = 1e-10)
  tr$train(X, y, catIDs)
  
  modelmat <- tr$print()
  modelmat[, 5] <- modelmat[, 5] + 1 # C++ to R indexing
  colnames(modelmat) <- c("depth", "modeldepth", "nodeId", "node type", "feature index",
                          "split value", "left intercept", "left slope", "right intercept", 
                          "right slope")
  # return output as a PILOT class
  output <- list(modelmat = modelmat,
                 residuals = tr$getResiduals(),
                 parameters = list( dfs = dfs, 
                                    min_sample_leaf = min_sample_leaf, 
                                    min_sample_alpha = min_sample_alpha,
                                    min_sample_fit = min_sample_fit,
                                    maxDepth = maxDepth,
                                    maxModelDepth = maxModelDepth,
                                    rel_tolerance = rel_tolerance),
                 data_names = data_names,
                 catInfo = catInfo,
                 modelpointer = tr, 
                 jsonString = tr$toJson)
  class(output) <- "PILOT" 
  
  return(output)
}





#' Print a PILOT model
#'
#' Print a PILOT model
#' @param object an object of the PILOT class
#' @examples
#' x <- rnorm(10)
#' 

print.PILOT <- function(object, ...) {
  if (!inherits(object, "PILOT")) {
    stop("Object is not of class 'PILOT'")
  }
  if (!is.matrix(object$modelmat)) {
    stop("PILOT object is corrupted: does not have a model matrix.")
  }
  print(object$modelmat)
}


#' Plot a PILOT model
#'
#' Plot a PILOT model
#' @param object an object of the PILOT class
#' @examples
#' x <- rnorm(10)
#' 

plot.PILOT <- function(object, ...) {
  if (!inherits(object, "PILOT")) {
    stop("Object is not of class 'PILOT'")
  }
  
  plot(object$residuals, xlab = "index", ylab = "residuals")
}



#' Plot a PILOT model
#'
#' Plot a PILOT model
#' @param object an object of the PILOT class
#' @param newdata a matrix or data frame with new data
#' @examples
#' x <- rnorm(10)
#' 

predict.PILOT <- function(object, newdata) {
  
  if (!inherits(object, "PILOT")) {
    stop("Object is not of class 'PILOT'")
  }
  # first check if the object was loaded from a file. If so, reconstruct it:
  if (capture.output(object$modelpointer[[".module"]])  == "<pointer: (nil)>") {
    tr <- new(PILOTcpp)
    tr$fromJSON(object$jsonString) # read the PILOt object from the Json string
    object$modelpointer = tr
  } else {
    tr <- object$modelpointer
  }
  
  # now check new input newdata, mainly the categorical features
  newdata <- as.matrix(newdata)
  
  if (anyNA(newdata)) stop("PILOT cannot handle NA values in new data (for now).")
  
  if (!isTRUE(all.equal(colnames(newdata), object$data_names))){
    stop("Column names of new data do not match the training data.")
  }
  
  catIDs <- apply(X, 2,  function(col) is.factor(col) || is.character(col)) + 0.0
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
      xf <- factor(newdata[, catID], levels = catInfo$factorlevels[[j]])
      X[, catID]   <- as.integer(xf) - 1
    }
  }
  
  preds <- tr$predict(newdata)
  return(preds)
}



