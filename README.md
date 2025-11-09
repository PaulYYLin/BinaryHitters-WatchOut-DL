# WatchOut - Edge-Optimized Fall Detection System

An intelligent fall detection system designed for edge devices, combining MediaPipe pose estimation with rule-based algorithms for real-time monitoring. Features hybrid processing architecture, lightweight video recording, and automated API integration.

## Highlights

- **Real-time Fall Detection**: Multi-algorithm fusion with 5 detection methods
- **Edge-Optimized Architecture**: Hybrid sync/async processing for resource-constrained devices
- **Ultra-lightweight Video Recording**: JPEG-based ring buffer with 95% memory savings
- **Privacy Protection**: Optional skeleton-only visualization mode
- **Automated Event Management**: Cooldown mechanism, retry logic, and dual API endpoints
- **Production Ready**: Docker containerization with resource limits and health checks
- **Experiment Mode**: Comprehensive data collection for model training and analysis

## Technology Stack

### Core Technologies
- **Computer Vision**: OpenCV 4.8.1 (headless build for edge devices)
- **Pose Estimation**: MediaPipe 0.10.21 (Lite model for low-latency inference)
- **Async I/O**: aiohttp 3.9+ (non-blocking video encoding and API uploads)
- **Environment Management**: uv (fast Python package installer)

### Detection Algorithms
- **Angle-based Detection**: Body tilt analysis using hip-to-shoulder vectors
- **Height-based Detection**: Vertical position tracking with dynamic thresholds
- **Velocity-based Detection**: Rapid movement analysis using temporal derivatives
- **Landmark-based Detection**: Key point position analysis (nose-to-hip ratio)
- **Multi-criteria Voting**: Ensemble method combining all algorithms

### Key Components
- **Hybrid Processing Architecture**: Synchronous OpenCV capture + asynchronous event processing
- **JPEG-compressed Ring Buffer**: Circular buffer with 70% quality compression
- **Async Video Encoder**: Background MP4 encoding with mp4v codec
- **Dual API Client**: Success (multipart upload) + Failure (JSON notification) endpoints
- **Event Manager**: Queue-based processing with 15s cooldown and retry logic
- **Configuration System**: YAML + Environment variables with validation

## Quick Start

### Docker Deployment (Recommended)

**Prerequisites**: Docker and Docker Compose installed

1. Download MediaPipe model:
   - URL: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task`
   - Save to: `src/utils/pose_landmarker_lite.task`

2. Configure environment:
   - Copy `.env.example` to `.env`
   - Set `API_BASE_URL` and API endpoints
   - Adjust `CAMERA_ID` if needed (default: 0)

3. Deploy:
   - `docker-compose up -d` (start in background)
   - `docker-compose logs -f` (view logs)
   - `docker-compose down` (stop)

### Local Development

**Prerequisites**: Python 3.11+ and uv installed

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Install dependencies: `uv sync`
3. Download MediaPipe model (see Docker step 1)
4. Configure `.env` file
5. Run: `uv run python main.py` or `uv run fall-detect`

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

## Core Architecture

### System Flow

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌─────────────┐
│   Camera    │────▶│ MediaPipe    │────▶│ Fall Detector  │────▶│ Ring Buffer │
│  (OpenCV)   │     │ Pose Model   │     │  (5 Methods)   │     │  (JPEG)     │
└─────────────┘     └──────────────┘     └────────┬───────┘     └─────────────┘
                                                   │
                                           Fall Detected?
                                                   │
                                                   ▼
                                         ┌──────────────────┐
                                         │  Event Manager   │
                                         │   (Async Queue)  │
                                         └────────┬─────────┘
                                                  │
                                   ┌──────────────┼──────────────┐
                                   ▼              ▼              ▼
                            Clip Extract    Video Encode    API Upload
                             (15s window)    (MP4/H264)     (Retry x3)
```

### Key Mechanisms

#### 1. Hybrid Processing Architecture
- **Main Thread**: Synchronous OpenCV capture + pose detection (for macOS GUI compatibility)
- **Worker Thread**: Asynchronous event loop for video encoding and API uploads
- **Benefit**: Maintains real-time camera processing while handling I/O-intensive tasks in background

#### 2. Multi-algorithm Fall Detection
- **Voting System**: Configurable threshold (default: 1/5 methods must detect fall)
- **Temporal Analysis**: 5-frame window for velocity calculation
- **Adaptive Thresholds**: Height drop ratio (20%), angle threshold (45°), velocity threshold (0.08)
- **Visibility Filtering**: Ignores low-confidence landmarks (min: 50%)

#### 3. Memory-Efficient Ring Buffer
- **JPEG Compression**: Reduces frame size by 95% (70% quality setting)
- **Circular Storage**: 30-second rolling window at 15 FPS (~450 frames)
- **Frame Skipping**: Saves every 2nd frame (configurable) to reduce buffer writes
- **Memory Footprint**: ~20MB vs ~400MB uncompressed

#### 4. Event Deduplication & Cooldown
- **15-Second Cooldown**: Prevents duplicate uploads for same fall incident
- **Timestamp Tracking**: Maintains last fall detection time
- **Queue-based Processing**: Ensures sequential handling of fall events

#### 5. Resilient API Integration
- **Dual Endpoint Strategy**:
  - Primary: Multipart video upload with metadata
  - Fallback: JSON-only notification when video upload fails
- **Exponential Backoff**: 3 retry attempts with increasing delays (1s, 2s, 4s)
- **Async Upload**: Non-blocking I/O prevents camera pipeline stalls
- **Device Check-in**: Periodic heartbeat to backend (configurable interval)

#### 6. Privacy Mode
- **Skeleton-only Rendering**: Displays stick figure without background video
- **No Raw Frame Storage**: Only skeleton landmarks saved in privacy mode
- **GDPR-friendly**: Reduces personal data exposure

#### 7. Experiment Mode
- **Data Collection**: Saves landmarks, detection details, and video clips
- **Timestamped Output**: Organized directory structure for analysis
- **Training Support**: Facilitates ML model improvement and validation

## Configuration

### Environment Variables

Key settings in `.env` file:

| Category | Variable | Default | Description |
|----------|----------|---------|-------------|
| **Camera** | `CAMERA_ID` | 0 | Device ID (0=default webcam) |
| | `CAMERA_RESOLUTION` | 640x480 | Video resolution |
| | `CAPTURE_FPS` | 30 | Camera capture frame rate |
| **Buffer** | `BUFFER_DURATION` | 30 | Ring buffer size (seconds) |
| | `JPEG_QUALITY` | 70 | Compression quality (0-100) |
| | `FRAME_SKIP` | 2 | Save every Nth frame |
| **Video** | `CLIP_DURATION` | 15 | Fall video clip length (seconds) |
| | `VIDEO_CODEC` | mp4v | Codec (mp4v/H264) |
| | `VIDEO_BITRATE` | 1000000 | Encoding bitrate (1 Mbps) |
| **API** | `API_BASE_URL` | - | Backend server base URL |
| | `API_UPLOAD_ENDPOINT` | - | Video upload path |
| | `API_FAILURE_ENDPOINT` | - | Failure notification path |
| | `API_CHECKIN_ENDPOINT` | - | Device heartbeat path |
| | `API_TIMEOUT` | 30 | Request timeout (seconds) |
| | `API_RETRY_ATTEMPTS` | 3 | Retry count for failed uploads |
| **Detection** | `FALL_ANGLE_THRESHOLD` | 45.0 | Max body tilt angle (degrees) |
| | `HEIGHT_DROP_THRESHOLD` | 0.2 | Min height drop ratio |
| | `VELOCITY_THRESHOLD` | 0.08 | Min vertical velocity |
| | `VOTE_THRESHOLD` | 1 | Min algorithms to trigger (1-5) |
| **System** | `HEADLESS_MODE` | true | Run without GUI |
| | `PRIVACY_MODE` | false | Skeleton-only display |
| | `COOLDOWN_PERIOD` | 15 | Event cooldown (seconds) |
| | `EXP_MODE` | false | Enable experiment mode |

### API Integration

#### Success Endpoint (Video Upload)
- **Method**: POST (multipart/form-data)
- **Fields**: video (MP4), timestamp, datetime, device_id, clip_duration, fall_info (JSON)
- **Response**: 200 OK on successful upload

#### Failure Endpoint (Notification)
- **Method**: POST (application/json)
- **Body**: event_type, error, timestamp, datetime, device_id, fall_info
- **Use Case**: Triggered when video upload fails after retries

#### Check-in Endpoint (Heartbeat)
- **Method**: POST (application/json)
- **Body**: device_id, timestamp, status
- **Interval**: Configurable (default: 300s)

## Performance Characteristics

### Resource Optimization

**Memory Usage**:
- JPEG-compressed ring buffer: ~20 MB (vs ~400 MB uncompressed)
- Total system RAM: 250-300 MB
- Docker container limit: 512 MB (with 256 MB reserved)

**CPU Usage**:
- MediaPipe Lite model: Optimized for edge devices
- Async I/O: Video encoding doesn't block detection pipeline
- mp4v codec: Faster than H264 on CPU-only devices
- Expected usage: 40-60% single core (Raspberry Pi 4 equivalent)

**Network**:
- Video upload: ~1-2 MB per fall event (15s clip at 1 Mbps)
- Retry mechanism: Exponential backoff reduces burst traffic
- Heartbeat: Lightweight JSON payload every 5 minutes

### Edge Device Optimizations

1. **Headless Mode**: Disables GUI rendering to save CPU cycles
2. **Frame Skipping**: Reduces buffer writes by 50% (saves every 2nd frame)
3. **JPEG Compression**: 95% memory reduction with minimal quality loss
4. **Async Processing**: I/O operations don't block real-time detection
5. **Lite Pose Model**: MediaPipe's smallest model for low-latency inference
6. **Resource Limits**: Docker constraints prevent OOM on constrained devices

## Design Principles

Following [CLAUDE.md](CLAUDE.md) guidelines:

- **Loose Coupling**: Modular architecture with clear interfaces (detectors, video, API, events)
- **Clear Architecture**: Layered design (camera → detection → events → API)
- **English Documentation**: Comprehensive inline comments and docstrings
- **Proper Logging**: Using Python logging module (no print statements)
- **Environment Management**: uv-based dependency management for reproducible builds


## License

MIT License - Part of the BinaryHitters-WatchOut-DL system

## Acknowledgments

- **MediaPipe** (Google): Pose estimation framework
- **OpenCV**: Computer vision library
- **aiohttp**: Asynchronous HTTP client library

---

**Project**: WatchOut Fall Detection System
**Organization**: BinaryHitters
**Maintained by**: BinaryHitters Team
