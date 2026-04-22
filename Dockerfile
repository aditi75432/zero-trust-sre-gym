FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

# Install Nginx (Rogue Service) and procps (for the EDR pkill command)
RUN apt-get update && \
    apt-get install -y nginx procps && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your environment files
COPY . /app

# Ensure the orchestrator script has execution permissions
RUN chmod +x start.sh

EXPOSE 7860

# Boot the Monolith
CMD ["./start.sh"]