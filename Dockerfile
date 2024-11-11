# Use Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

RUN apt-get update && apt-get install git ffmpeg -y

# Install Python dependencies
RUN pip install uv
RUN uv pip install --no-cache-dir --system -r requirements.txt

# Switch to non-root user
USER nobody

# Copy application code
COPY . .

# Set the default command
CMD ["python", "./main.py"]