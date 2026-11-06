FROM rocker/r-ver:4.4.1

WORKDIR /app

# Install R packages (cached layer)
COPY install_packages.R .
RUN Rscript install_packages.R

# Copy the script
COPY titanic.R .

# Data is mounted at runtime to /app/data (not baked into the image)
CMD ["Rscript", "titanic.R"]