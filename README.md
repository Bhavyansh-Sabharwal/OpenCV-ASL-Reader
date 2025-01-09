# OpenCV ASL Reader

A real-time American Sign Language (ASL) letter detection system using OpenCV and MediaPipe. This project uses computer vision to detect and interpret hand gestures corresponding to ASL letters.

## Features

- Real-time hand tracking and landmark detection
- ASL letter recognition based on finger angles
- Visual feedback with hand landmarks
- Display of finger angles and normalized values
- Mirrored display for intuitive interaction
- Web interface for easy access
- Support for all 26 ASL alphabet letters

## Prerequisites

- Python 3.7+
- Webcam or camera device

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/OpenCV-ASL-Reader.git
cd OpenCV-ASL-Reader
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

There are two ways to run the ASL Reader:

### 1. Web Interface (Recommended)
1. Start the web server:
```bash
python app.py
```
2. Open your web browser and navigate to:
```
http://localhost:5000
```
3. The webcam feed will start automatically in your browser

### 2. Terminal Application
1. Run the standalone application:
```bash
python asl_reader.py
```

In both versions, the application will:
- Display hand landmarks
- Show finger angles and normalized values
- Predict and display ASL letters in real-time
- Press 'q' to quit (terminal version only)

## Current Features

- Supports all 26 letters of the ASL alphabet
- Real-time detection and visualization
- User-friendly web interface
- Detailed angle measurements and normalization
- Mirrored display for intuitive interaction

## Future Improvements

- Implement machine learning for better accuracy
- Add support for dynamic gestures
- Improve robustness in various lighting conditions
- Add word and sentence recognition
- Add tutorial mode for learning ASL

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
