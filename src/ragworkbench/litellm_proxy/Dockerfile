# syntax=docker/dockerfile:1.3
FROM ghcr.io/berriai/litellm:main-v1.82.1-nightly

ENTRYPOINT []

# Define build arguments
ARG BUILD_VERSION=latest
ARG BUILD_DATE

# Label the image with metadata
LABEL version="$BUILD_VERSION"
LABEL date="$BUILD_DATE"

# Set the working directory
WORKDIR /app

# Environment variables for LiteLLM UI configuration
ENV LITELLM_ASSET_PREFIX=""
ENV UI_ACCESS_MODE="admin_only"

# Install PostgreSQL adapter for database connectivity
RUN pip install psycopg2-binary

# Copy configuration files (config.yaml, user keys, SSL certs)
COPY . .

# Expose a port
EXPOSE 4000

# Set the entrypoint command
CMD echo "Starting litellm (version $BUILD_VERSION built on $BUILD_DATE)" && echo "" && litellm --config config.yaml
