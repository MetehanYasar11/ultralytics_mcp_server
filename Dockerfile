# Stage 1: Environment setup
FROM continuumio/miniconda3:latest AS builder

# Copy environment file
COPY environment.yml /tmp/environment.yml

# Create conda environment
RUN conda env create -f /tmp/environment.yml

# Stage 2: Runtime
FROM continuumio/miniconda3:latest

# Label the image
LABEL name="ultra-api"

# Copy the environment from builder stage
COPY --from=builder /opt/conda/envs/ultra-dev /opt/conda/envs/ultra-dev

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application using conda run
CMD ["conda", "run", "-n", "ultra-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
