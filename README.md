# Fall Detection System - Edge Device Optimized

A modular fall detection system using MediaPipe pose estimation and rule-based algorithms, optimized for edge devices with video recording, ring buffer, and API integration.

## Features

✅ **Real-time Fall Detection**: MediaPipe pose estimation with multiple detection algorithms
✅ **Video Recording**: Ring buffer with JPEG compression (95% memory savings)
✅ **API Integration**: Automatic video upload when fall detected
✅ **Edge Optimized**: Headless mode, async I/O, minimal resource usage
✅ **Container Ready**: Docker support with resource limits
✅ **Configurable**: Environment-based configuration

## Quick Start

### 1. Prerequisites

- Docker and Docker Compose (for container deployment)
- OR Python 3.11+ (for local development)
- USB camera or webcam
- MediaPipe pose model (download instructions below)

### 2. Download MediaPipe Model

```bash
# Create utils directory if not exists
mkdir -p src/utils

# Download the pose landmarker lite model
# URL: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

# Save to: src/utils/pose_landmarker_lite.task
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API endpoints and settings
nano .env
```

Key settings to configure:
- `API_SUCCESS_ENDPOINT`: Your API endpoint for fall event uploads
- `API_FAILURE_ENDPOINT`: Your API endpoint for failure notifications
- `CAMERA_ID`: Camera device ID (0 for default webcam)
- `CAMERA_RESOLUTION`: Resolution (e.g., 640x480)

### 4. Deploy with Docker (Recommended for Edge Devices)

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### 5. Run Locally (Development with uv)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run
uv run python main.py

# Or use the console script
uv run fall-detect
```

## Project Structure

```
BinaryHitters-WatchOut-DL/
├── main.py                           # Entry point with async architecture
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Multi-stage Docker build
├── docker-compose.yml               # Container orchestration
├── .env.example                     # Environment variables template
├── CLAUDE.md                        # Development guidelines
└── src/                             # Main source package
    ├── __init__.py                  # Package exports
    ├── detectors/                   # Detection algorithms
    │   ├── __init__.py
    │   └── fall_detector.py         # FallDetectorRuleBased class
    ├── camera/                      # Camera integration
    │   ├── __init__.py
    │   └── live_detector.py         # LiveCameraFallDetector (refactored)
    ├── video/                       # NEW: Video processing
    │   ├── __init__.py
    │   ├── ring_buffer.py           # Lightweight ring buffer with JPEG compression
    │   └── encoder.py               # Async video encoder (MP4/H264)
    ├── api/                         # NEW: API integration
    │   ├── __init__.py
    │   └── client.py                # Async API client with retry logic
    ├── events/                      # NEW: Event management
    │   ├── __init__.py
    │   └── manager.py               # Fall event manager with cooldown
    ├── config/                      # NEW: Configuration management
    │   ├── __init__.py
    │   └── settings.py              # Environment-based settings
    └── utils/                       # Utility modules
        ├── __init__.py
        ├── constants.py             # Constants and configurations
        ├── visualization.py         # Drawing and visualization functions
        └── pose_landmarker_lite.task # MediaPipe model (download separately)
```

## Architecture Overview

### System Flow

```
Camera → Pose Detection → Fall Detection → Ring Buffer
                              ↓
                         Fall Detected?
                              ↓
                    Event Manager (async)
                              ↓
                    Video Encoder (async)
                              ↓
                    API Upload (async)
```

### Components

#### 1. **Fall Detector** (`src/detectors/fall_detector.py`)
Rule-based fall detection with 5 methods:
- Angle-based (body tilt)
- Height-based (vertical position)
- Velocity-based (rapid movement)
- Landmark-based (key point positions)
- Multi-criteria (voting, recommended)

#### 2. **Ring Buffer** (`src/video/ring_buffer.py`)
Memory-efficient video storage:
- JPEG compression (70% quality)
- 30s circular buffer
- ~20MB memory (vs ~400MB uncompressed)
- Auto-cleanup old frames

#### 3. **Video Encoder** (`src/video/encoder.py`)
Async MP4 encoding:
- Background processing (non-blocking)
- mp4v codec (CPU-optimized)
- Configurable bitrate
- Extracts 15s clips on fall detection

#### 4. **API Client** (`src/api/client.py`)
Dual API integration:
- **Success API**: Upload video + metadata
- **Failure API**: Send text notification (when upload fails)
- Retry logic (3 attempts, exponential backoff)
- Async/non-blocking

#### 5. **Event Manager** (`src/events/manager.py`)
Coordinates fall events:
- 15s cooldown (prevent duplicates)
- Event queue (async processing)
- Clip extraction → Encoding → Upload
- Statistics tracking

#### 6. **Configuration** (`src/config/settings.py`)
Environment-based config:
- Load from .env file
- Validation and defaults
- Edge device optimizations

## Usage Examples

### Basic Usage (Container)

```bash
# Start system
docker-compose up -d

# Check status
docker-compose ps

# View real-time logs
docker-compose logs -f fall-detector

# Stop system
docker-compose down
```

### Local Development (with GUI)

```python
# Edit .env
HEADLESS_MODE=false

# Run
python main.py

# Press 'q' to quit
```

### Custom Configuration

```python
from src.config import get_settings
from src import LiveCameraFallDetector

# Load settings
settings = get_settings()

# Override programmatically
settings.BUFFER_DURATION = 60  # 60s buffer
settings.CLIP_DURATION = 20    # 20s clips

# Initialize detector
detector = LiveCameraFallDetector(
    model_path=str(settings.MODEL_PATH),
    camera_id=settings.CAMERA_ID,
    ring_buffer=ring_buffer,
    event_manager=event_manager,
    settings=settings,
    headless=True
)
```

### API Endpoint Requirements

#### Success Endpoint (Video Upload)
```
POST /api/fall-events
Content-Type: multipart/form-data

Fields:
- video: MP4 file (15s clip)
- timestamp: Unix timestamp
- datetime: ISO 8601 datetime
- device_id: Camera ID
- clip_duration: Video duration in seconds
- fall_info: JSON with detection details
```

#### Failure Endpoint (Text Notification)
```
POST /api/fall-failures
Content-Type: application/json

Body:
{
  "event_type": "fall_detection_failure",
  "error": "Error message",
  "timestamp": 1234567890.123,
  "datetime": "2024-01-01T12:00:00",
  "device_id": 0,
  ...fall_info
}
```

## Configuration Options

### Environment Variables (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_ID` | 0 | Camera device ID |
| `CAMERA_RESOLUTION` | 640x480 | Video resolution |
| `CAPTURE_FPS` | 30 | Camera capture FPS |
| `BUFFER_DURATION` | 30 | Ring buffer duration (seconds) |
| `BUFFER_FPS` | 15 | Buffer storage FPS |
| `FRAME_SKIP` | 2 | Save every Nth frame |
| `JPEG_QUALITY` | 70 | JPEG compression (0-100) |
| `CLIP_DURATION` | 15 | Fall clip duration (seconds) |
| `VIDEO_CODEC` | mp4v | Video codec (mp4v/H264) |
| `VIDEO_BITRATE` | 1000000 | Bitrate in bps (1Mbps) |
| `API_SUCCESS_ENDPOINT` | - | Success upload URL |
| `API_FAILURE_ENDPOINT` | - | Failure notification URL |
| `API_KEY` | - | Optional API key |
| `API_TIMEOUT` | 30 | Request timeout (seconds) |
| `API_RETRY_ATTEMPTS` | 3 | Retry count |
| `COOLDOWN_PERIOD` | 15 | Event cooldown (seconds) |
| `HEADLESS_MODE` | true | Run without GUI |

### Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fall_angle_threshold` | 45.0 | Max angle from vertical (degrees) |
| `height_drop_threshold` | 0.2 | Min height drop ratio (20%) |
| `velocity_threshold` | 0.08 | Min vertical velocity |
| `temporal_window` | 5 | Frames for temporal analysis |
| `min_visibility` | 0.5 | Min landmark visibility |

## Performance Optimization (Edge Devices)

### Memory Usage
- **Ring Buffer**: JPEG compression saves 95% memory
  - Uncompressed: ~400MB (30s @ 30fps, 640x480)
  - Compressed: ~20MB (same duration)
- **Frame Skip**: Reduces buffer writes by 50%
- **Total RAM**: ~250-300MB (entire system)

### CPU Usage
- **MediaPipe Lite**: Optimized pose model
- **Async I/O**: Video encoding doesn't block detection
- **mp4v Codec**: Faster than H264 on CPU
- **Headless Mode**: No GUI rendering overhead
- **Expected**: 40-60% CPU (single core, Raspberry Pi 4 equivalent)

### Resource Limits (Docker)
```yaml
resources:
  limits:
    cpus: '2.0'      # 2 cores max
    memory: 512M     # 512MB max
  reservations:
    memory: 256M     # 256MB reserved
```

## Troubleshooting

### Camera Not Found
```bash
# Check available cameras
ls -la /dev/video*

# Update CAMERA_ID in .env
CAMERA_ID=1  # Try different IDs
```

### Model Not Found
```bash
# Ensure model is in correct location
ls -lh src/utils/pose_landmarker_lite.task

# Download if missing (see Quick Start section)
```

### API Upload Fails
```bash
# Check logs
docker-compose logs -f fall-detector

# Verify API endpoints in .env
# Check network connectivity
# Review API_RETRY_ATTEMPTS setting
```

### High Memory Usage
```bash
# Reduce buffer duration
BUFFER_DURATION=20  # From 30s to 20s

# Reduce JPEG quality
JPEG_QUALITY=60  # From 70 to 60

# Increase frame skip
FRAME_SKIP=3  # From 2 to 3
```

### High CPU Usage
```bash
# Lower resolution
CAMERA_RESOLUTION=320x240

# Reduce capture FPS
CAPTURE_FPS=15

# Increase frame skip
FRAME_SKIP=3
```

## Development

### Running Tests (Local)
```bash
# Install dev dependencies
pip install -r requirements.txt

# Run with test mode
python main.py
```

### Logs Location
- Container: `docker-compose logs -f`
- Local: `./logs/fall_detection.log`

### Debug Mode
```python
# Edit main.py
logging.basicConfig(level=logging.DEBUG)  # Change from INFO
```

## Design Principles (from CLAUDE.md)

✅ **Loose Coupling**: Independent modules with clear interfaces
✅ **Clear Architecture**: Layered design (detection → events → api)
✅ **English Comments**: Comprehensive documentation
✅ **Proper Logging**: Using logging module (no print statements)

## License

This project is part of the BinaryHitters-WatchOut-DL system.

## Credits

- **MediaPipe**: Google's pose estimation framework
- **OpenCV**: Computer vision library
- **aiohttp**: Async HTTP client
