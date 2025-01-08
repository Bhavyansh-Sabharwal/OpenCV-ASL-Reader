# OpenCV ASL Reader

A real-time American Sign Language (ASL) letter detection system using OpenCV and MediaPipe. This project uses computer vision to detect and interpret hand gestures corresponding to ASL letters.

## Features

- Real-time hand tracking and landmark detection
- ASL letter recognition based on finger angles
- Visual feedback with hand landmarks
- Display of finger angles and normalized values
- Mirrored display for intuitive interaction

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
pip install opencv-python mediapipe numpy scikit-learn
```

## Usage

1. Run the ASL Reader:
```bash
python asl-reader.py
```

2. Position your hand in front of the camera
3. Make ASL letter gestures
4. The program will display:
   - Hand landmarks
   - Finger angles
   - Normalized values
   - Predicted ASL letter (when confidence is high enough)
5. Press 'q' to quit the application

## Current Limitations

- Limited to basic ASL letters (A, B, C)
- Requires good lighting conditions
- Single hand detection only
- Basic pattern matching system

## Future Improvements

- Expand letter recognition to full ASL alphabet
- Implement machine learning for better accuracy
- Add support for dynamic gestures
- Improve robustness in various lighting conditions
- Add word and sentence recognition

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
