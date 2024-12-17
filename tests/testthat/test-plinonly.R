# test that plin node works by fitting a plin-only model of max depth 3
test_that("plin node works", {
  X <- matrix(c(-6, 16, 1, -5, -7, 12,  4, -3,  5,  7, -2,  1,
                17, -13, -4,  4,  -1, 18, -20, -5), ncol = 2)
  y <- X %*% c(2, 1) + seq(-2, 2, length.out = 10)
  dfs <- c(1, -1, -1, -1, 1, -1) 
  modelParams <- c(2, 2, 2, 3, 10, ncol(X), 0)
  tr <- new(PILOTcpp,
            dfs = dfs,
            modelParams = modelParams,
            rel_tolerance = 1e-4,
            precScale = 1e-10)
  catIDs <- rep(0, ncol(X))
  
  tr$train(X, y, catIDs)
  tr.out <- tr$print()
  
  expect_equal(drop(var(tr$getResiduals())),
               64.50311, tolerance = 1e-3)
  
  expect_equal(tr.out[, 1], c(0, 1, 1))
  expect_equal(tr.out[, 2], c(0, 1, 1))
  expect_equal(tr.out[, 3], c(0, 0, 1))
  expect_equal(tr.out[, 4], c(4, 0, 0))
  expect_equal(tr.out[1, 5], 0)
  expect_equal(tr.out[1, 6], 1)
  
  idL <- which(X[, 1] <= 1)
  xL <- X[idL, 1]
  xR <- X[-idL, 1]
  yL <- y[idL]
  yR <- y[-idL]
  lm.outL <- lm(yL~xL)
  lm.outR <- lm(yR~xR)
  
  expect_equal(tr.out[1, 7:8], as.vector(lm.outL$coefficients)[1:2])
  expect_equal(tr.out[1, 9:10], as.vector(lm.outR$coefficients)[1:2])
})