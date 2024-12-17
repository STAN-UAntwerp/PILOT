test_that("con node works", {
  X <- matrix(c(-6, 16, 1, 5, -7, 12,  4, -6,  5,  7, -2,  1,
                17 -13, -4,  4,  1, 18 -20, -5), ncol = 2) 
  y <- X %*% c(2, 1)
  dfs <- c(1, -1, -1, -1, -1, -1)
  modelParams <- c(2, 2, 2, 3, 10, ncol(X), 0)
  tr <- new(PILOTcpp,
           dfs = dfs,
           modelParams = modelParams,
           rel_tolerance = 1e-2,
           precScale = 1e-10)
  catIDs <- rep(0, ncol(X))
  
  tr$train(X, y, catIDs)
  tr.out <- tr$print()
  
  expect_equal(tr$getResiduals(), y - mean(y))
  expect_identical(tr.out[1], 0)
  expect_identical(tr.out[2], 0)
  expect_identical(tr.out[3], 0)
  expect_identical(tr.out[4], 0)
  expect_equal(tr.out[7], mean(y))
})

