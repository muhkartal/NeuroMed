# MedExplain AI Pro - Installation Guide

This document provides comprehensive instructions for installing and deploying MedExplain AI Pro in various environments. Follow the steps appropriate for your deployment scenario.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Local Development Installation](#local-development-installation)
3. [Production Deployment](#production-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Configuration Options](#configuration-options)
7. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

| Component        | Specification                       |
| ---------------- | ----------------------------------- |
| Operating System | Windows 10, macOS 11, Ubuntu 20.04  |
| Processor        | Dual-core 2.0 GHz                   |
| RAM              | 4 GB                                |
| Storage          | 2 GB available space                |
| Python           | 3.8                                 |
| Browser          | Chrome 90+, Firefox 88+, Safari 14+ |
| Network          | Broadband Internet connection       |

### Recommended Specifications

| Component        | Specification                                |
| ---------------- | -------------------------------------------- |
| Operating System | Windows 11, macOS 12+, Ubuntu 22.04+         |
| Processor        | Quad-core 2.5 GHz or better                  |
| RAM              | 8 GB or more                                 |
| Storage          | 5 GB available space                         |
| Python           | 3.10 or newer                                |
| Browser          | Latest version of Chrome, Firefox, or Safari |
| Network          | High-speed Internet connection               |

## Local Development Installation

### Prerequisites

Ensure the following tools are installed on your system:

-  Git
-  Python 3.8 or higher
-  pip (Python package installer)
-  Virtual environment tool (venv or conda)

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/medexplain-ai-pro.git
   cd medexplain-ai-pro
   ```

2. Create and activate a virtual environment:

   ```bash
   # Using venv
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:

   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file with appropriate values, including:

   -  API keys for OpenAI
   -  Database credentials
   -  Application settings

5. Launch the application:

   ```bash
   streamlit run medexplain/app.py
   ```

6. Access the application in your browser at:
   ```
   http://localhost:8501
   ```

## Production Deployment

### Server Preparation

1. Provision a server with the following specifications:

   -  4+ CPU cores
   -  16+ GB RAM
   -  20+ GB storage
   -  Ubuntu 20.04 LTS or newer

2. Update the system and install dependencies:

   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y python3 python3-pip python3-venv git nginx
   ```

3. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/medexplain-ai-pro.git
   cd medexplain-ai-pro
   ```

4. Set up a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

5. Configure environment variables:

   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file with production values.

### Streamlit Configuration

1. Create a Streamlit configuration file:

   ```bash
   mkdir -p ~/.streamlit
   touch ~/.streamlit/config.toml
   ```

2. Add the following configuration:
   ```toml
   [server]
   headless = true
   port = 8501
   enableCORS = false
   ```

### Service Setup

1. Create a systemd service file:

   ```bash
   sudo nano /etc/systemd/system/medexplain.service
   ```

2. Add the following content:

   ```
   [Unit]
   Description=MedExplain AI Pro
   After=network.target

   [Service]
   User=your_username
   WorkingDirectory=/path/to/medexplain-ai-pro
   ExecStart=/path/to/medexplain-ai-pro/venv/bin/streamlit run medexplain/app.py
   Restart=on-failure
   Environment="PATH=/path/to/medexplain-ai-pro/venv/bin"
   Environment="PYTHONPATH=/path/to/medexplain-ai-pro"
   EnvironmentFile=/path/to/medexplain-ai-pro/.env

   [Install]
   WantedBy=multi-user.target
   ```

3. Start and enable the service:
   ```bash
   sudo systemctl start medexplain
   sudo systemctl enable medexplain
   ```

### Nginx Configuration (Optional)

1. Create an Nginx configuration file:

   ```bash
   sudo nano /etc/nginx/sites-available/medexplain
   ```

2. Add the following configuration:

   ```
   server {
       listen 80;
       server_name your_domain.com;

       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_read_timeout 86400;
       }
   }
   ```

3. Enable the site and restart Nginx:

   ```bash
   sudo ln -s /etc/nginx/sites-available/medexplain /etc/nginx/sites-enabled/
   sudo systemctl restart nginx
   ```

4. Set up SSL with Certbot (recommended):
   ```bash
   sudo apt install -y certbot python3-certbot-nginx
   sudo certbot --nginx -d your_domain.com
   ```

## Docker Deployment

### Prerequisites

-  Docker
-  Docker Compose (optional, for multi-container setup)

### Using Pre-built Image

1. Pull the latest image:

   ```bash
   docker pull medexplain/medexplain-ai-pro:latest
   ```

2. Run the container:

   ```bash
   docker run -d -p 8501:8501 \
     --name medexplain \
     -e OPENAI_API_KEY=your_key_here \
     -e OTHER_ENV_VAR=value \
     medexplain/medexplain-ai-pro:latest
   ```

3. Access the application at `http://localhost:8501`

### Building Custom Image

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/medexplain-ai-pro.git
   cd medexplain-ai-pro
   ```

2. Build the Docker image:

   ```bash
   docker build -t medexplain-ai-pro:custom .
   ```

3. Run the container:
   ```bash
   docker run -d -p 8501:8501 \
     --name medexplain \
     -e OPENAI_API_KEY=your_key_here \
     medexplain-ai-pro:custom
   ```

### Using Docker Compose

1. Create a `docker-compose.yml` file:

   ```yaml
   version: "3"
   services:
      medexplain:
         image: medexplain/medexplain-ai-pro:latest
         ports:
            - "8501:8501"
         environment:
            - OPENAI_API_KEY=your_key_here
            - DATABASE_URL=postgresql://user:password@db:5432/medexplain
         depends_on:
            - db
         restart: always

      db:
         image: postgres:14
         volumes:
            - postgres_data:/var/lib/postgresql/data
         environment:
            - POSTGRES_PASSWORD=password
            - POSTGRES_USER=user
            - POSTGRES_DB=medexplain
         restart: always

   volumes:
      postgres_data:
   ```

2. Launch with Docker Compose:
   ```bash
   docker-compose up -d
   ```

## Cloud Deployment

### AWS Deployment

1. Launch an EC2 instance (t3.large or better recommended)
2. Install Docker:
   ```bash
   sudo amazon-linux-extras install docker
   sudo service docker start
   sudo usermod -a -G docker ec2-user
   ```
3. Pull and run the Docker container as described in the Docker section
4. Configure security groups to allow traffic on port 8501
5. (Optional) Set up an Elastic IP and Route 53 record for DNS

### Google Cloud Platform

1. Create a VM instance (e2-standard-2 or better)
2. Install Docker:
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io
   ```
3. Pull and run the Docker container
4. Configure firewall rules to allow traffic on port 8501

### Azure

1. Create a Virtual Machine (B2s or better)
2. Install Docker following Azure documentation
3. Deploy using the Docker instructions above
4. Configure Network Security Group to allow port 8501

## Configuration Options

The application can be configured through environment variables in the `.env` file:

| Variable                | Description                      | Default                      |
| ----------------------- | -------------------------------- | ---------------------------- |
| `OPENAI_API_KEY`        | API key for OpenAI services      | None (Required)              |
| `STREAMLIT_SERVER_PORT` | Port for Streamlit server        | 8501                         |
| `DATABASE_URL`          | Database connection string       | sqlite:///data/medexplain.db |
| `LOG_LEVEL`             | Logging level                    | INFO                         |
| `ENABLE_ANALYTICS`      | Enable usage analytics           | False                        |
| `MAX_SYMPTOM_HISTORY`   | Maximum symptom history to store | 100                          |
| `DEMO_MODE`             | Run in demonstration mode        | False                        |

## Troubleshooting

### Common Installation Issues

**Issue**: Package installation fails
**Solution**: Try upgrading pip and setuptools:

```bash
pip install --upgrade pip setuptools wheel
```

**Issue**: "ModuleNotFoundError" when running the application
**Solution**: Ensure you're in the correct virtual environment and dependencies are installed:

```bash
pip install -r requirements.txt
```

**Issue**: OpenAI API connection fails
**Solution**: Check your API key in the `.env` file and ensure internet connectivity

**Issue**: Docker container exits immediately
**Solution**: Check logs with `docker logs medexplain` and ensure environment variables are set correctly

### Getting Help

If you encounter issues not covered in this guide:

1. Check the logs:

   ```bash
   # For local installation
   tail -f logs/medexplain.log

   # For Docker
   docker logs medexplain
   ```

2. Verify system requirements are met
3. Ensure all environment variables are correctly set
4. Check for firewall or network restrictions

For additional support, contact the development team or consult the community forum.

---

Developed by Muhammad Ibrahim Kartal | [kartal.dev](https://kartal.dev)

Copyright © 2025 MedExplain AI Pro. All rights reserved.
