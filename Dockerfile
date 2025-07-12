# Stage 1: Environment setup
FROM continuumio/miniconda3:latest AS builder

# Copy environment file
COPY environment.yml /tmp/environment.yml

# Create conda environment and install PyTorch CPU-only
RUN conda env create -f /tmp/environment.yml && \
    conda install -n ultra-dev pytorch torchvision torchaudio cpuonly -c pytorch

# Set shell for conda activation
SHELL ["/bin/bash", "-c"]

# Configure conda environment activation in bashrc
RUN echo "source activate ultra-dev" >> ~/.bashrc

# Stage 2: Runtime
FROM continuumio/miniconda3:latest

# Label the image
LABEL name="ultralytics-mcp-server"
LABEL version="1.0.0"
LABEL description="Ultralytics MCP Server API"

# Copy the environment from builder stage
COPY --from=builder /opt/conda/envs/ultra-dev /opt/conda/envs/ultra-dev

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Install uvicorn with standard extras and sse-starlette for SSE support
RUN conda run -n ultra-dev pip install "uvicorn[standard]" sse-starlette

# Expose port
EXPOSE 8000

# Run the application using conda run with improved configuration and app reload
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ultra-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "app", "--log-level", "info"]
