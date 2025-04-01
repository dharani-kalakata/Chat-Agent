# Deployment Instructions

## Development Setup

### Prerequisites
- Docker and Docker Compose installed
- Python 3.12 (for local development without Docker)
- OpenAI API key (if using OpenAI models)
- Git
- **Ollama installed and running on your host machine**

### Initial Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Ai_AutonmationBackend
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Install and setup Ollama on your host machine**:
   ```bash
   # Install Ollama following instructions at https://ollama.ai/download
   # Pull required models
   ollama pull llama3.1
   ollama pull deepseek-r1:1.5b  # or any other models you need
   # Ensure Ollama is running
   # On Windows: Check Task Manager for Ollama service
   # On Mac/Linux: Run `ps aux | grep ollama` to verify it's running
   ```

4. Build and start the development environment:
   ```bash
   docker-compose build
   docker-compose up -d
   ```

5. Access the API at http://localhost:8000
   - API documentation available at http://localhost:8000/docs

### Development Workflow
1. Make code changes - the API server will automatically reload
2. Check logs: `docker-compose logs -f api`
3. Run a specific command: `docker-compose exec api python -m pytest`
4. Pull new Ollama models: `ollama pull <model-name>` (runs on your host, not in container)

### Linux Host Configuration
If you're running Docker on Linux:
1. Find your host machine's IP address: `ip addr show docker0 | grep -Po 'inet \K[\d.]+'`
2. Set the HOST_OLLAMA_URL in your .env file: `HOST_OLLAMA_URL=http://<your-host-ip>:11434`
3. Uncomment the Linux-specific OLLAMA_BASE_URL line in docker-compose.yml

## Production Deployment

### Prerequisites
- Docker and Docker Compose installed on production server
- Docker registry access (optional, for custom image hosting)
- Domain name and SSL certificate for HTTPS
- **Ollama installed and running on your production host machine**

### Deployment Steps
1. Prepare production environment:
   ```bash
   # Copy files to production server
   scp -r .env docker-compose.prod.yml Dockerfile server:/path/to/deployment/
   ```

2. **Install and setup Ollama on the production host**:
   ```bash
   # Install Ollama following instructions at https://ollama.ai/download
   # Pull required models
   ollama pull llama3.1
   ollama pull deepseek-r1:1.5b  # or any other models you need
   ```

3. Build and deploy on production server:
   ```bash
   cd /path/to/deployment
   docker-compose -f docker-compose.prod.yml build
   docker-compose -f docker-compose.prod.yml up -d
   ```

4. Set up a reverse proxy (Nginx/Traefik) with HTTPS

### Updating Production
1. Pull latest code changes:
   ```bash
   git pull
   ```

2. Rebuild and restart containers:
   ```bash
   docker-compose -f docker-compose.prod.yml build
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Scaling Considerations
- For higher throughput, increase the number of Gunicorn workers
- Consider using Docker Swarm or Kubernetes for multi-node deployments
- For distributed setups, ensure each node has Ollama installed or configure all nodes to access a central Ollama service

## Monitoring and Maintenance

### Health Checks
- Monitor the `/healthcheck` endpoint
- Set up alerting based on container health status

### Logs
- View logs: `docker-compose -f docker-compose.prod.yml logs -f`
- Consider integrating with a log aggregation system

### Backups
- Backup Chroma DB volume regularly:
  ```bash
  docker run --rm -v ai_automation_chroma_data:/data -v $(pwd):/backup alpine tar -czf /backup/chroma_backup.tar.gz /data
  ```

### Model Management
- Manage models directly on the host machine:
  ```bash
  # List models
  ollama list
  # Pull new models
  ollama pull <model-name>
  # Remove unused models
  ollama rm <model-name>
  ```
