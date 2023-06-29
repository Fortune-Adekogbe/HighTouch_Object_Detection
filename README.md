# High-Touch Object Detection System

## Description
The High-Touch Object Detection System is a real-time object detection application that uses webcam input to identify and locate high-touch objects in the video stream. These objects typically include door handles, chairs, dining tables, etc., that are frequently touched by multiple people, potentially contributing to the spread of diseases. The system can be used in various environments like offices, hospitals, public transportation, and homes for tracking and encouraging frequent disinfection of such high-risk areas.

## Features

1. **Real-time Object Detection**: The system analyzes webcam feed in real-time and can accurately detect high-touch objects.

2. **Bounding Box Visualization**: Detected objects are highlighted with bounding boxes on the video feed.

3. **Confidence Score**: Each detected object is tagged with a confidence score, reflecting the system's confidence in its detection.

4. **Webcam Compatibility**: The system works with standard webcam input.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Fortune-Adekogbe/HighTouch_Object_Detection.git
```

2. Navigate to the project directory:

```bash
cd HighTouch_Object_Detection
```

3. Install the necessary packages. It's recommended to do this in a Python virtual environment:

```bash
pip install -r requirements.txt
```

## Usage

After installation, you can start the object detection system with the following command:

```bash
python inference.py
```

Once the system is running, point your webcam towards the area you want to monitor for high-touch objects. The application window will display the webcam feed with bounding boxes and confidence scores overlaid on detected objects.

## Note

The High-Touch Object Detection System was designed for educational and research purposes. Although it may help in identifying high-touch areas that require frequent disinfection, it is not a substitute for professional advice or solution for disease prevention.

## Contributions

Pull requests are welcome. For significant changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)