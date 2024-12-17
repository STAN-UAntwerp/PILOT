# test that lin node works by fitting a lin-only model until maxModelDepth of 10.
# it should just perform l_2 boosting by iterating between the two predictors.
test_that("lin node works", {
  X <- matrix(c(-6, 16, 1, 5, -7, 12,  4, -6,  5,  7, -2,  1,
                17, -13, -4,  4,  1, 18, -20, -5), ncol = 2)
  y <- X %*% c(2, 1) + seq(-2, 2, length.out = 10)
  dfs <- c(1, 1, -1, -1, -1, -1) 
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
               1.271856, tolerance = 1e-3)
  
  expect_identical(tr.out[1:11, 1], rep(0, 11))
  expect_equal(tr.out[1:11, 2], as.vector(0:10))
  expect_equal(tr.out[1:11, 3], rep(0, 11))
  expect_equal(tr.out[1:11, 4], c(rep(1, 10), 0))
  expect_equal(tr.out[1:10, 5], c(0, 1, 0, 1, 0, 0, 1, 0, 0, 1))
  
  lm.out1 <- lm(y ~ X[, 1]); res1 <- lm.out1$residuals;
  res1t <- y - pmin(pmax(y-res1, min(y)), max(y))
  lm.out2 <- lm(res1t ~ X[, 2]); res2 <- lm.out2$residuals;
  res2t <- y - pmin(pmax(y-res2, min(y)), max(y))
  lm.out3 <- lm(res2t ~ X[, 1]); res3 <- lm.out3$residuals;
  res3t <- y - pmin(pmax(y-res3, min(y)), max(y))
  lm.out4 <- lm(res3t ~ X[, 2]); res4 <- lm.out4$residuals;
  res4t <- y - pmin(pmax(y-res4, min(y)), max(y))
  lm.out5 <- lm(res4t ~ X[, 1]); res5 <- lm.out5$residuals;
  res5t <- y - pmin(pmax(y-res5, min(y)), max(y))
  lm.out6 <- lm(res5t ~ X[, 1]); res6 <- lm.out6$residuals;
  res6t <- y - pmin(pmax(y-res6, min(y)), max(y))
  lm.out7 <- lm(res6t ~ X[, 2]); res7 <- lm.out7$residuals;
  res7t <- y - pmin(pmax(y-res7, min(y)), max(y))
  lm.out8 <- lm(res7t ~ X[, 1]); res8 <- lm.out8$residuals;
  res8t <- y - pmin(pmax(y-res8, min(y)), max(y))
  lm.out9 <- lm(res8t ~ X[, 1]); res9 <- lm.out9$residuals;
  res9t <- y - pmin(pmax(y-res9, min(y)), max(y))
  lm.out10 <- lm(res9t ~X[, 2]); res10 <- lm.out10$residuals;
  res10t <- y - pmin(pmax(y-res10, min(y)), max(y))
  expect_equal(tr.out[1, 7:8], unname(lm.out1$coefficients))
  expect_equal(tr.out[2, 7:8], unname(lm.out2$coefficients))
  expect_equal(tr.out[3, 7:8], unname(lm.out3$coefficients))
  expect_equal(tr.out[4, 7:8], unname(lm.out4$coefficients)) 
  expect_equal(tr.out[5, 7:8], unname(lm.out5$coefficients))
  expect_equal(tr.out[6, 7:8], unname(lm.out6$coefficients))
  expect_equal(tr.out[7, 7:8], unname(lm.out7$coefficients))
  expect_equal(tr.out[8, 7:8], unname(lm.out8$coefficients))
  expect_equal(tr.out[9, 7:8], unname(lm.out9$coefficients))
  expect_equal(tr.out[10, 7:8], unname(lm.out10$coefficients))
})