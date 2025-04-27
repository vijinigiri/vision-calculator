# Vision Calculator ✋➕🧠  
**Vision Calculator** is an air-handwriting recognition tool that uses **computer vision and hand tracking** to detect math expressions written in mid-air. Inspired by Apple Notes, this project lets users write mathematical equations with hand gestures, providing real-time computation without touching the keyboard.

have a look how it works
linkedln - https://www.linkedin.com/posts/vijinigiri-gowri-shankar_compuetervision-handtracking-opencv-activity-7321464320532385794-y6Hl?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEOGt5gB-inZKogxJE16RC3MCMV8L40E30M

## 🔧 Features

- ✍️ **Write math expressions using hand gestures** via webcam  
- 🧮 **Supports variables** like `x = 5`, `y = 3`, and expressions like `x + y =`  
- 🎯 **Pinch gesture** to activate pointer and begin writing  
- 🖐️ **Open hand** gesture to clear the screen  
- ⚙️ Real-time tracking using **MediaPipe** and **OpenCV**  
- 🧠 Uses symbolic parsing to compute and display expression results on-screen  
- 🧽 No-touch interaction: fully gesture-driven  

## 📦 Tech Stack

- **Python**  
- **OpenCV**  
- **MediaPipe** (for hand tracking)  
- **NumPy**  
- **Custom logic** for expression building and evaluation  

## 📁 Project Structure

- `main.py`: Core script for hand tracking and expression rendering  
- `utils/`: Helper functions for gesture handling and parsing  
- `expressions/`: Handles symbolic math parsing and evaluation logic  

## 🚀 Usage

1. Connect your webcam and run the script:  
   ```bash
   python main.py
   ```
2. Use the **pinch gesture** to activate the pointer and write your expression.  
3. Declare variables like `x = 4` or `y = 10`, then write `x + y =` to compute.  
4. **Open and close your hand** to clear the screen and start fresh.  

## 📌 Notes

- Make sure your environment has proper lighting for hand tracking to work smoothly.  
- The expression is evaluated once `=` is detected in the gesture-written input.  
- Ideal for educational tools, touchless interfaces, and math accessibility applications.


## 🏷️ Tags

#ComputerVision #HandTracking #OpenCV #VisionCalculator #GestureControl #PythonProjects #AIProjects #MachineLearning #WebcamTracking #MathTools #TouchlessInterface
