#!/bin/bash

# MedExplain AI Pro Setup Script
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo -e "${GREEN}"
echo "====================================="
echo "MedExplain AI Pro - Setup Script"
echo "====================================="
echo -e "${NC}"

# Check if Docker is installed
echo -e "${YELLOW}Checking dependencies...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed.${NC}"
    echo "Please install Docker Compose from https://docs.docker.com/compose/install/"
    exit 1
fi

echo -e "${GREEN}✓ Dependencies satisfied${NC}"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓ Created .env file${NC}"
    echo -e "${YELLOW}Please edit the .env file to set your configuration.${NC}"
else
    echo -e "${YELLOW}Note: .env file already exists. Skipping creation.${NC}"
fi

# Create required directories
echo -e "${YELLOW}Setting up directory structure...${NC}"
mkdir -p api
mkdir -p frontend
mkdir -p data/ml_models
echo -e "${GREEN}✓ Directory structure created${NC}"

# Build Docker images
echo -e "${YELLOW}Building Docker images...${NC}"
docker-compose build
echo -e "${GREEN}✓ Docker images built successfully${NC}"

echo -e "${GREEN}"
echo "====================================="
echo "Setup complete! Start the application with:"
echo "docker-compose up"
echo "====================================="
echo -e "${NC}"
