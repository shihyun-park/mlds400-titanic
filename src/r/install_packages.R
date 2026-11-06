# Installs the R packages used by titanic.R.
# rocker/r-ver pins CRAN to a dated snapshot, so versions are reproducible.
pkgs <- c("readr", "dplyr")
install.packages(pkgs, repos = getOption("repos"))

# Fail the Docker build if anything did not install
missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) stop("Failed to install: ", paste(missing, collapse = ", "))
cat("Installed:", paste(pkgs, collapse = ", "), "\n")