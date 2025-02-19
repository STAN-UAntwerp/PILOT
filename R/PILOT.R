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
#' @param maxModelDepth an integer specifying the maximum model depth of the tree. Model depth is incremented when any model is fit, including lin models.
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


pilot <- function(X, y,
                  dfs = c(1, 2, 5, 5, 7, 5), 
                  min_sample_leaf = 5, 
                  min_sample_alpha = 5,
                  min_sample_fit = 10,
                  maxDepth = 20,
                  maxModelDepth = 100,
                  rel_tolerance = 1e-4) {
  
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
  colnames(modelmat) <- c("depth", "modeldepth", "nodeId", "node_type", "feature_index",
                          "split_value", "left_intercept", "left_slope", "right_intercept", 
                          "right_slope")
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
#' @param x an object of the PILOT class
#' @param ... other print parameters 
#' @examples
#' x <- rnorm(10)
#' 

print.PILOT <- function(x, ...) {
  if (!inherits(x, "PILOT")) {
    stop("Object is not of class 'PILOT'")
  }
  if (!is.matrix(x$modelmat)) {
    stop("PILOT object is corrupted: does not have a model matrix.")
  }
  print(x$modelmat)
}


#' Plot a PILOT model
#'
#' Plot a PILOT model
#' @param x an object of the PILOT class
#' @param df the data matrix as data frame
#' @param ... other graphical parameters 
#' @examples
#' x <- rnorm(10)
#' 

plot.PILOT <- function(x, df, ...) {
  
  
  
  prepare_modelmat <- function(modelmat) {
    # Step 1: for each depth, add unique nodeId for each node which takes into acount skipped con nodes
    modelmat$newNodeId <- rep(NA, nrow(modelmat))
    
    node_id_map <- integer(max(modelmat$depth) + 1)
    # Process each row sequentially
    for (i in seq_len(nrow(modelmat))) {
      if (modelmat$node_type[i] == 1) {next}
      depth <- modelmat$depth[i]
      
      # Assign the next sequential nodeId for the current depth
      modelmat$newNodeId[i] <- node_id_map[depth + 1]
      
      # Increment the counter for this depth
      node_id_map[depth + 1] <- node_id_map[depth + 1] + 1
      # check if leaf node
      if (modelmat$node_type[i] == 0) { # increment all child node counters
        if (depth < max(modelmat$depth)) {
          skippedchilds <- 1 + (1 + depth):max(modelmat$depth)
          node_id_map[skippedchilds] <- node_id_map[skippedchilds]  + 2^(seq_along(skippedchilds))
        }
      }
    }
    for (i in rev(seq_len(nrow(modelmat)))){
      if (is.na(modelmat$newNodeId[i])) {
        modelmat$newNodeId[i] = modelmat$newNodeId[i+1]
      }
    }
    remove(node_id_map, skippedchilds, depth); gc()
    
    # Step 2: 
    # add unique id by appending depth  + newNodeId
    # note that lin models generate duplicate IDs
    modelmat$unique_id <- paste(modelmat$depth, modelmat$newNodeId, sep = "-")
    
    # Step 3:
    # for each non-root node, add  the unique ID of the parents: 
    
    modelmat$parentId <- rep(NA, nrow(modelmat))
    
    for (i in 1:nrow(modelmat)) {
      if (modelmat$node_type[i] %in% 2:6) {
        childrows <- which(modelmat$unique_id %in% c(paste0(modelmat$depth[i] + 1, "-", modelmat$newNodeId[i] * 2 ),
                                                     paste0(modelmat$depth[i] + 1, "-", modelmat$newNodeId[i] * 2 + 1)))
        modelmat$parentId[childrows] <- modelmat$unique_id[i]
      }
    }
    
    # Step 4:
    # create a list of leafnodemodels which contains the coefficients of the linear model
    # active in that leaf node. 
    
    leafnodemodels <- list()
    for (i in 1:nrow(modelmat)) {
      if (modelmat$node_type[i] == 0) { # leaf node
        coefvec <- rep(NA, ncol(X) + 1 ) # first index is intercept
        parentId <- modelmat$parentId[i]
        coefvec[1] <- modelmat$left_intercept[i]
        idx <- i
        while(!is.na(parentId)) {
          parents <- which(modelmat$unique_id %in% parentId)
          for (j in parents) {
            if (modelmat$node_type[j] == 1) { # lin fit
              coefvec[modelmat$feature_index[j] + 1] <- modelmat$left_slope[j]
              coefvec[1] <- coefvec[1] + modelmat$left_intercept[j]
            } else {
              # first check whether this is left or right node
              if (idx == max(which(modelmat$parentId == parentId))) {
                leftchild = FALSE
              } else {
                leftchild = TRUE
              }
              if (leftchild) {
                coefvec[modelmat$feature_index[j] + 1] <- modelmat$left_slope[j]
                coefvec[1] <- coefvec[1] + modelmat$left_intercept[j]
              } else {
                coefvec[modelmat$feature_index[j] + 1] <- modelmat$right_slope[j]
                coefvec[1] <- coefvec[1] + modelmat$right_intercept[j]
              }
            }
          }
          idx <- tail(parents, 1)
          parentId <- modelmat$parentId[idx]
        }
        leafnodemodels[[modelmat$unique_id[i]]] <- coefvec
      }
    }
    return(list(modelmat = modelmat, leafnodemodels = leafnodemodels)) 
  }
  
  
  
  
  
  build_party <- function(prepare_modelmat.out,
                          df) {
    # takes in output of prepare_modelmat()
    # and df, which is a data frame of the predictors
    
    
    
    build_partynode <- function(node_unique_id, tree_matrix) {
      
      # Get the node data
      node_data <- tree_matrix[tree_matrix$unique_id == node_unique_id, ]
      
      if (nrow(node_data) == 0) return(NULL)  # Safety check
      
      # If it's a leaf node (split_value is NA), return a leaf node
      if (is.na(node_data$split_value)) {
        return(partynode(as.numeric(rownames(node_data)), info = paste("Leaf", node_data$unique_id)))
      }
      
      # Create a split node
      split <- partysplit(as.integer(node_data$feature_index), breaks = node_data$split_value)
      
      # Find children: First two rows with depth one higher than current depth *below* the current row
      children <- tree_matrix[tree_matrix$depth == node_data$depth + 1, , drop = FALSE]
      
      # Filter out the rows that come before the current row
      children <- children[as.numeric(rownames(children)) > as.numeric(rownames(node_data)), , drop = FALSE]
      
      # Select first two children that correspond to this node's children
      children <- head(children, 2)
      
      if (nrow(children) == 0) {
        return(partynode(as.numeric(rownames(node_data)), info = paste("Leaf", node_data$unique_id)))
      }
      
      # Recursively create child nodes
      kids <- lapply(children$unique_id, build_partynode, tree_matrix)
      kids <- Filter(Negate(is.null), kids)  # Remove NULL values
      
      return(partynode(as.numeric(rownames(node_data)), split = split, kids = kids))
    }
    
    
    
    modelmat <- prepare_modelmat.out$modelmat
    leafnodemodels <- prepare_modelmat.out$leafnodemodels
    
    prepareFIplot <- function(coefs, maxNpreds = 5) {
      slopes       <- coefs[-1]
      intercept    <- coefs[1]
      totpred      <- as.matrix(df) %*% slopes + intercept
      centercept   <- mean(totpred, na.rm = TRUE)
      totpred      <- totpred - centercept
      totpredscale <- sd(totpred, na.rm = TRUE)
      predictors   <- t(t(df) * slopes)
      colm         <- apply(predictors, 2, mean, na.rm=TRUE)
      predictors   <- scale(predictors, center = colm, scale = FALSE)
      predscales   <- apply(predictors, 2, sd, na.rm=TRUE)
      predscales   <- predscales / max(predscales)
      predscales[which(is.na(predscales))] <- 0
      names(predscales) <- colnames(df)
      ordering <- order(predscales, decreasing = TRUE)[1:min(ncol(df), maxNpreds)]
      output <- data.frame(cbind(names(predscales), predscales)[ordering, ])
      output[, 2] <- as.numeric(output[, 2])
      output[, 1] <- factor(output[, 1], levels = output[, 1])
      colnames(output) <- c("feature", "importance")
      rownames(output) <- NULL
      output
    }
    
    
    
    leaf_data_frames <- lapply(leafnodemodels, prepareFIplot)
    
    
    reducedmat             <- modelmat[which(modelmat$node_type!=1), ]
    reducedmat$split_value <- round(reducedmat$split_value, 2)
    
    # Build tree starting from root node (depth 0, nodeId 0)
    tree  <- build_partynode("0-0", reducedmat)
    py    <- party(tree, df)
    return(list(py = py, reducedmat = reducedmat, leaf_data_frames = leaf_data_frames))
  }
  
  plot_party <- function(prepare_modelmat.out,
                         df,
                         build_party.out,
                         leaf_data_frames, 
                         autoLayout = FALSE,
                         plotVarImportance = FALSE) {
    reducedmat <- build_party.out$reducedmat
    py         <- build_party.out$py
    leaf_data_frames <- build_party.out$leaf_data_frames
    
    if (!autoLayout)
      nodedf <- data.frame(id = 1:nrow(reducedmat),
                           x = (reducedmat$newNodeId + 1) / (1+2^(reducedmat$depth)),
                           y = 1 - reducedmat$depth / (max(reducedmat$depth)+1))
    else {
      nodedf = NULL
    }
    
    leafnodes <- which(reducedmat$node_type==0)
    
    gg <-  ggparty(py, layout = nodedf) +
      geom_edge() +
      geom_edge_label() 
    
    if (plotVarImportance) {
      gg <- gg + geom_node_label(aes(label = ifelse(!is.na(info), info, splitvar)), ids = "inner")  
      for (i in 1:length(leafnodes)) {
        gg <- gg + 
          geom_node_plot(
            gglist = list(
              geom_col(aes(x = feature, y = importance), width = 0.5, color = "black",
                       data = leaf_data_frames[[i]], fill = "steelblue")
            ),
            scales = "free",
            ids = leafnodes[i]
          )
        
      }
    } else  {
      gg <- gg + geom_node_label(aes(label = ifelse(!is.na(info), info, splitvar)))  
    }
    
    
    gg
    return(gg)
  }
  
  
  
  
  if (!inherits(x, "PILOT")) {
    stop("Object is not of class 'PILOT'")
  }
  
  
  
  
  prepare_modelmat.out <- prepare_modelmat(modelmat)
  
  
  build_party.out <- build_party(prepare_modelmat.out, df)
  
  gg <- plot_party(prepare_modelmat.out, df,
                   build_party.out,
                   autoLayout = TRUE,
                   plotVarImportance = FALSE)
  
  gg
  
  return(gg)
}



#' Predict with a PILOT model
#'
#' Predict with a PILOT model
#' @param object an object of the PILOT class
#' @param newdata a matrix or data frame with new data
#' @param maxDepth predict using all nodes of depth up to maxDepth. If NULL, predict using full tree.
#' @examples
#' x <- rnorm(10)
#' 

predict.PILOT <- function(object, newdata, maxDepth = NULL) {
  
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
      xf <- factor(newdata[, catID], levels = object$catInfo$factorlevels[[j]])
      X[, catID]   <- as.integer(xf) - 1
    }
  }
  
  if (is.null(maxDepth)) {
    maxDepth = object$parameters$maxDepth
  }
  
  preds <- tr$predict(newdata, maxDepth)
  return(preds)
}



