# test that pcon node works by fitting a pcon-only model of max depth 3
test_that("pcon node works", {
  X <- matrix(c(-6, 16, 1, 5, -7, 12,  4, -6,  5,  7, -2,  1,
                17, -13, -4,  4,  1, 18, -20, -5), ncol = 2)
  y <- X %*% c(2, 1) + seq(-2, 2, length.out = 10)
  dfs <- c(1, -1, 1, -1, -1, -1) 
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
               30.71491, tolerance = 1e-3)
  
  expect_equal(tr.out[, 1], c(0, rep(c(1, 2, 2), 2)))
  expect_equal(tr.out[, 2], c(0, rep(c(1, 2, 2), 2)))
  expect_equal(tr.out[, 3], c(0, 0, 0, 1, 1, 2, 3))
  expect_equal(tr.out[, 4], c(2, 2, 0, 0, 2, 0, 0))
  expect_equal(tr.out[c(1, 2, 5), 5], c(1, 0, 0))
  expect_equal(tr.out[c(1, 2, 5), 6], c(-2, -6, 4))
  
  expect_equal(tr.out[1, 7], mean(y[which(X[, 2] <= -2)]))
  expect_equal(tr.out[1, 9], mean(y[which(X[, 2] > -2)]))
  expect_equal(tr.out[2, 7], mean(y[which((X[, 2] <= -2 )& (X[, 1 ]<= -6))]) - mean(y[which(X[, 2] <= -2)]))
  expect_equal(tr.out[2, 9], mean(y[which((X[, 2] <= -2 )& (X[, 1 ]> -6))]) - mean(y[which(X[, 2] <= -2)]))
  expect_equal(tr.out[5, 7], mean(y[which((X[, 2] > -2 )& (X[, 1 ]<= 4))]) - mean(y[which(X[, 2] > -2)]))
  expect_equal(tr.out[5, 9], mean(y[which((X[, 2] > -2 )& (X[, 1 ]> 4))]) - mean(y[which(X[, 2] > -2)]))
  
  expect_equal(tr.out[c(1, 2, 5), 8], rep(0, 3))
  expect_equal(tr.out[c(1, 2, 5), 10], rep(0, 3))
})