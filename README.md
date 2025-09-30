# Real-Time Object Detection with Pepper Robot using YOLOv5

This project is a Bachelor's degree thesis in Computer Science, focusing on the integration of a state-of-the-art object detection model (YOLOv5) with the SoftBank Robotics Pepper platform. The goal is to enable the robot to autonomously search for, identify, and interact with specific objects in its environment based on a voice command.

## 🌟 Abstract

Human-robot interaction is a rapidly growing field, and a key component of this is the ability for robots to understand and perceive their surroundings. This project equips the Pepper robot with an advanced vision system powered by YOLOv5. The robot can be activated by a simple voice command (e.g., "find the bottle"), after which it will autonomously navigate its environment, process its camera feed in real-time to detect the target object, and upon successful identification, stop, center itself, point towards the object, and announce its finding. This demonstrates a complete perception-action loop, combining voice recognition, autonomous navigation, real-time computer vision, and physical interaction.

## ✨ Key Features

- **🗣️ Voice-Activated Operation**: The search process is initiated by a spoken command.
- **🧭 Autonomous Exploration**: The robot navigates an area to actively search for the target object.
- **👁️ Real-Time Object Detection**: Utilizes the YOLOv5 model for fast and accurate detection on Pepper's live camera feed.
- **🎯 Object Verification & Centering**: Implements a routine to confirm the object's presence and precisely align the robot's body and head with it.
- **👉 Physical Interaction**: Pepper points at the successfully identified object to indicate its location.
- **🤖 Verbal Feedback**: The robot provides spoken confirmation once the object is found.

## ⚙️ How It Works

The operational flow is designed as a sequential, state-driven process:

1.  **Initialization**: The script connects to the Pepper robot via its IP address and initializes all the necessary NAOqi OS services (motion, speech, vision, etc.).
2.  **Listening for Command**: The robot activates its speech recognition engine and waits for a specific keyword from a predefined vocabulary (e.g., "find", "bottle").
3.  **Exploration Phase**: Once the command is recognized, Pepper begins an autonomous exploration of a predefined area using its built-in navigation capabilities.
4.  **Detection Loop**: Simultaneously, the `start_searching` function captures frames from the top camera, preprocesses them, and feeds them into the loaded YOLOv5 model for inference.
5.  **Object Sighting**: If the target object is detected in a frame with sufficient confidence, the system flags the object as `seen`.
6.  **Verification Phase**: To avoid false positives, a verification routine (`handle_object_recognition`) is triggered. The robot stops exploring and spends a few seconds confirming if the object is consistently visible. If the object is lost, the robot will perform a head-scanning motion to try and re-locate it.
7.  **Centering and Interaction**: Once the object is confirmed, the `center_on_object` function calculates the necessary head and body movements to face the object directly. Pepper then points its arm towards it and announces the finding via Text-To-Speech.
8.  **Task Completion**: The robot returns to a neutral stance, waiting for further commands. If the object is not found after verification, the robot resumes the exploration phase.

## 🛠️ Technologies Used

- **Hardware**:
  - SoftBank Robotics Pepper
- **Software & Frameworks**:
  - **Operating System**: NAOqi OS
  - **Core Language**: Python
  - **Robot SDK**: NAOqi Python SDK (`qi`)
  - **Computer Vision**:
    - PyTorch
    - YOLOv5
    - OpenCV
  - **Libraries**:
    - Pillow (PIL)
    - NumPy

## 🚀 Setup and Installation

### Prerequisites

- A Pepper robot connected to the same network as your development machine.
- Python 2.7 or Python 3, depending on your NAOqi version.
- The NAOqi Python SDK correctly configured in your `PYTHONPATH`.
- Git for cloning the repository.

### Installation Steps

1.  **Clone the YOLOv5 repository and this project:**
    ```bash
    # Clone the official YOLOv5 repository
    git clone https://github.com/ultralytics/yolov5.git
    
    # Navigate into the yolov5 directory
    cd yolov5

    # Clone this project's files into the yolov5 directory
    # git clone <URL_of_your_project_repository> .
    # Or simply copy cam_detection.py and your .pt model file here
    ```

2.  **Install dependencies:**
    It is highly recommended to set up a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `requirements.txt` is provided by the YOLOv5 repository.*

3.  **Download Model Weights:**
    Download the pre-trained YOLOv5 weights or use your own custom-trained model. Ensure the `yolov5s.pt` file (or your custom model file) is in the root directory of the project.

4.  **Configure the Script:**
    Open `cam_detection.py` and modify the `YOLO_SETTINGS` and `ROBOT_SETTINGS` dictionaries at the top of the file to match your needs (e.g., change `model_path` or `object_names`).

## ▶️ Usage

To run the program, execute the main script from your terminal, providing the robot's IP address as an argument.

```bash
python cam_detection.py --ip <robot_ip_address>
```

**Arguments:**
- `--ip`: (Required) The IP address of the Pepper robot on the network.
- `--port`: (Optional) The NAOqi port number. Defaults to `9559`.

Example:
```bash
python cam_detection.py --ip 192.168.1.101
```

Once running, the terminal will print "Speech recognition engine started." Approach the robot and say "find bottle" to begin the search.

## 💡 Future Improvements

- **Multi-Object Support**: Extend the vocabulary and logic to search for multiple, different classes of objects.
- **Enhanced Navigation**: Integrate more advanced obstacle avoidance and path planning instead of relying solely on the built-in `explore` function.
- **Grasping and Manipulation**: Add a pipeline to allow the robot to not only find the object but also approach it and attempt to pick it up.
- **Improved Conversational AI**: Replace the basic keyword spotting with a more robust NLU (Natural Language Understanding) system for more flexible commands.
- **Performance Optimization**: Explore model quantization or a lighter detection model (e.g., YOLOv5n) to improve inference speed and reduce latency.