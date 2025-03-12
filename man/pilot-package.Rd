\name{pilot-package}
\alias{pilot-package}
\docType{package}
\title{
\packageTitle{pilot}
}
\description{
\packageDescription{pilot}
}
\details{
This package provides an implementation of
\itemize{
  \item the PILOT algorithm described in Raymaekers, Rousseeuw, Verdonck and Yao (2024) <doi:10.1007/s10994-024-06590-3> (open access).
  \item the RaFFLE algorithm described in Raymaekers, Rousseeuw, Servotte, Verdonck and Yao (2024) <doi:10.48550/arXiv.2502.10185> (open access).
}
The package uses Rcpp modules to expose C++ classes to R.
The R functions pilot and raffle wrap around these C++ classes, and generate an R object of the s3 class \code{pilot} and \code{raffle} respectively.
These objects have \code{print}, \code{plot} and \code{predict} functions.

In order to interact with the C++ class directly, the \code{pilot} and \code{raffle} classes have an attribute
\code{modelpointer}, which allows direct access to some of the underlying C++ functionality.
The objects also have an attribute called \code{jsonString}. This string describes the model, and allows for saving it as an \code{rdata} file.
When the model is loaded at a later stage, the string will be used to create a new C++ object and pointer to that object with the exact same properties as
the original saved model.
}
\author{
\packageAuthor{pilot}

Maintainer: \packageMaintainer{pilot}
}
\references{
%%  ~~ Literature or other references for background information ~~
}
\keyword{package}

%% Uncomment below to imitate parts of library(help = rpilotdevel)
%\section{The \file{DESCRIPTION} File}{\packageDESCRIPTION{rpilotdevel}}
%\section{Documentation Index}{\packageIndices{rpilotdevel}}

\seealso{
%%  ~~ Optional links to other man pages, e.g. ~~
%%  ~~ \code{\link[<pkg>:<pkg>-package]{<pkg>}} ~~
}
\examples{
%%  ~~ Optional simple examples of the most important functions ~~
}
