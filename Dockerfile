# Multi-stage Dockerfile for Fall Detection System
# Optimized for edge devices with minimal image size

# Stage 1: Builder - Install Python dependencies
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install to user directory
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# Stage 2: Runtime - Minimal production image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2

# Install runtime dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Add Python packages to PATH
ENV PATH=/root/.local/bin:$PATH

# Create app directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /tmp/fall_events /app/logs

# Copy application code
COPY src/ ./src/
COPY main.py .
COPY .env* ./

# Health check (optional - checks if process is running)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pgrep -f "python main.py" || exit 1

# Run as non-root user for security (optional but recommended)
# RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app /tmp/fall_events
# USER appuser

# Default command
CMD ["python", "main.py"]
