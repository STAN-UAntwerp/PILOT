# test that truncation works
test_that("training truncation works", {
  X <- matrix(c(100, -60, 1, -5, -7, 12,  4, -3,  5,  7, 100,  -60,
                17, -13, -4,  4,  -1, 18, -20, -5), ncol = 2)
  y <- X %*% c(2, 1) + seq(-2, 2, length.out = 10)
  y[1] <- 20
  dfs <- c(1, 1, -1, -1, -1, -1) 
  modelParams <- c(2, 2, 2, 3, 1, ncol(X), 0)
  tr <- new(PILOTcpp,
            dfs = dfs,
            modelParams = modelParams,
            rel_tolerance = 1e-4,
            precScale = 1e-10)
  catIDs <- rep(0, ncol(X))
  
  tr$train(X, y, catIDs)
  tr.out <- tr$print()
  
  lm.out <- lm(y~X[, 1])
  
  truncresids <- y - pmax(pmin(y - lm.out$residuals, 
                               max(y)),
                          min(y))
  
  expect_equal(as.vector(tr.out[2, 7] - (truncresids - tr$getResiduals())), rep(0, 10))
  
  ## now with plin
  set.seed(123)
  n <- 10^3
  X <- matrix(rnorm(n*2), ncol = 2) %*% matrix(c(0.97, 0.26, 0.26, 0.97), ncol = 2)
  y <- -(X[, 1] < 0) * X[, 1] * 10 + (X[, 1] >= 0) * X[, 1] * 10 +
    -(X[, 2] < 0) * X[, 2] * 9 + (X[, 2] >= 0) * X[, 2] * 9 + rnorm(n)
  
  X[1, ] <- c(1, 1) * 8
  y[1] <- 0
  dfs <- c(1, -1, -1, -1, 1, -1) 
  
  tr <- new(PILOTcpp,
            dfs = dfs,
            modelParams = modelParams,
            rel_tolerance = 1e-4,
            precScale = 1e-10)
  catIDs <- rep(0, ncol(X))
  
  tr$train(X, y, catIDs)
  tr.out <- tr$print(); tr.out
  
  idL <- which(X[, tr.out[1, 5]+1] <= tr.out[1, 6])
  xL <- X[idL, tr.out[1, 5]+1]
  xR <- X[-idL, tr.out[1, 5]+1]
  yL <- y[idL]
  yR <- y[-idL]
  lm.outL <- lm(yL~xL)
  lm.outR <- lm(yR~xR)
  
  resids <- rep(0, n)
  resids[idL] <- lm.outL$residuals
  resids[-idL] <- lm.outR$residuals
  
  truncresidsR <- y[-idL] - pmax(pmin(y[-idL] - lm.outR$residuals, 
                               max(y)),
                          min(y) )
  expect_equal(as.vector(tr.out[3, 7] - (truncresidsR - tr$getResiduals()[-idL])), rep(0, length(yR)))
  
  
})