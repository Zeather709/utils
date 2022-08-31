library(tictoc)

tic.clearlog()

tic('ex1')
# code example 1
toc(log = TRUE, quiet = TRUE)

tic('ex2')
# code example 2
toc(log = TRUE, quiet = TRUE)

# Retrieve list of execution times for ex1 & ex2
time_log <- tic.log(format = TRUE)