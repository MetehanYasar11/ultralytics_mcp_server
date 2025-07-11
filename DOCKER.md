# Docker Deployment Guide

This directory contains Docker Compose configurations for deploying the Ultralytics MCP Server in different environments.

## Files Overview

- **`docker-compose.yml`** - Base configuration for production
- **`docker-compose.override.yml`** - Development overrides (auto-loaded)
- **`docker-compose.prod.yml`** - Production-specific settings
- **`.env.example`** - Environment variables template

## Quick Start

### 1. Basic Deployment

```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env

# Start the service
docker-compose up -d
```

### 2. Development Mode

```bash
# Development mode with auto-reload
docker-compose up

# Or explicitly use override
docker-compose -f docker-compose.yml -f docker-compose.override.yml up
```

### 3. Production Deployment

```bash
# Production mode
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# With monitoring stack
docker-compose -f docker-compose.yml -f docker-compose.prod.yml --profile monitoring up -d
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ULTRA_API_KEY` | `your-api-key-here` | API authentication key |
| `CUDA_VISIBLE_DEVICES` | - | GPU devices to use |
| `MEMORY_LIMIT` | `4G` | Container memory limit |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DATASETS_PATH` | `./datasets` | Host path for datasets |
| `RUNS_PATH` | `./runs` | Host path for results |

### Volume Mounts

- **`./datasets:/datasets`** - Dataset storage
- **`./runs:/runs`** - Training results and outputs
- **`./config:/app/config`** - Configuration files
- **`ultralytics-cache`** - Model cache (named volume)

### Health Check

The service includes a health check that:
- **Endpoint**: `http://localhost:8000/`
- **Interval**: 30 seconds
- **Timeout**: 3 seconds
- **Retries**: 3 attempts
- **Start Period**: 30 seconds

## Service Access

Once running, the API is available at:

- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json
- **Health Check**: http://localhost:8000/

## Common Commands

### Service Management

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart services
docker-compose restart

# View logs
docker-compose logs -f ultra-api

# Check service status
docker-compose ps
```

### Development Commands

```bash
# Build local image
docker-compose build

# Pull latest image
docker-compose pull

# Run specific profile
docker-compose --profile monitoring up -d

# Scale service (if configured)
docker-compose up -d --scale ultra-api=3
```

### Maintenance

```bash
# Remove all containers and volumes
docker-compose down -v

# Remove orphaned containers
docker-compose down --remove-orphans

# Update images
docker-compose pull && docker-compose up -d

# View resource usage
docker stats
```

## Directory Structure

Ensure the following directories exist and have proper permissions:

```
├── datasets/          # Dataset files
├── runs/             # Training outputs
├── config/           # Configuration files
├── logs/             # Application logs
└── monitoring/       # Monitoring configs (optional)
    ├── prometheus.yml
    └── grafana/
```

### Create Directories

```bash
# Create required directories
mkdir -p datasets runs config logs monitoring/grafana

# Set permissions (Linux/macOS)
chmod 755 datasets runs config logs
chmod 644 .env
```

## GPU Support

To enable GPU support:

1. Install [NVIDIA Docker Runtime](https://github.com/NVIDIA/nvidia-docker)
2. Set environment variable:
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1
   ```
3. Start with GPU access:
   ```bash
   docker-compose up -d
   ```

## Monitoring (Optional)

### Enable Monitoring Stack

```bash
# Start with monitoring
docker-compose -f docker-compose.yml -f docker-compose.prod.yml --profile monitoring up -d
```

**Services:**
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Custom Metrics

The API exposes metrics at `/metrics` for Prometheus scraping.

## Reverse Proxy (Production)

For production deployments with SSL/TLS:

```bash
# Create external network
docker network create web

# Start with Traefik proxy
docker-compose -f docker-compose.yml -f docker-compose.prod.yml --profile proxy up -d
```

Configure your domain in `docker-compose.prod.yml`:
```yaml
labels:
  - "traefik.http.routers.ultra-api.rule=Host(`api.yourdomain.com`)"
```

## Troubleshooting

### Common Issues

**Port Already in Use:**
```bash
# Check what's using port 8000
lsof -i :8000
# Or change port in docker-compose.yml
```

**Permission Denied:**
```bash
# Fix volume permissions
sudo chown -R $USER:$USER datasets runs config logs
```

**Out of Disk Space:**
```bash
# Clean up Docker
docker system prune -a
docker volume prune
```

**Container Won't Start:**
```bash
# Check logs
docker-compose logs ultra-api

# Check container status
docker-compose ps

# Restart service
docker-compose restart ultra-api
```

### Debug Mode

Enable debug logging:

```bash
# Set in .env file
LOG_LEVEL=DEBUG

# Or override temporarily
LOG_LEVEL=DEBUG docker-compose up
```

## Security Considerations

1. **API Key**: Always set a strong `ULTRA_API_KEY`
2. **Firewall**: Restrict access to port 8000
3. **Updates**: Regularly update the Docker image
4. **Volumes**: Secure dataset and output directories
5. **Secrets**: Use Docker secrets for sensitive data

## Backup and Recovery

### Backup Important Data

```bash
# Backup datasets and results
tar -czf backup-$(date +%Y%m%d).tar.gz datasets/ runs/ config/

# Backup Docker volumes
docker run --rm -v ultralytics-cache:/data -v $(pwd):/backup alpine tar czf /backup/cache-backup.tar.gz /data
```

### Restore Data

```bash
# Restore from backup
tar -xzf backup-20240101.tar.gz

# Restore volume
docker run --rm -v ultralytics-cache:/data -v $(pwd):/backup alpine tar xzf /backup/cache-backup.tar.gz -C /
```

## Performance Tuning

### Resource Limits

Adjust in `.env` or `docker-compose.prod.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '4.0'
    reservations:
      memory: 4G
      cpus: '2.0'
```

### Volume Performance

For better I/O performance:
- Use SSD storage for volumes
- Consider using bind mounts instead of named volumes
- Mount `/tmp` as tmpfs for temporary files

## Support

For issues and questions:
- Check the [main README](../README.md)
- Review [API documentation](http://localhost:8000/docs)
- Check Docker logs: `docker-compose logs -f`
