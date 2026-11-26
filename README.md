![Taekwondow Kick Analyzer Image](taekwondo_kick_analyzer.png)

![Built with Python](https://img.shields.io/badge/Built%20with-Python-000000?logo=python&logoColor=white&style=flat)
![NumPy](https://img.shields.io/badge/Library-NumPy-FF0000?logo=numpy&logoColor=white&style=flat)
![MediaPipe](https://img.shields.io/badge/Pose%20Estimation-MediaPipe-000000?logo=google&logoColor=white&style=flat)
![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-FF0000?logo=opencv&logoColor=white&style=flat)
![Real-time Analysis](https://img.shields.io/badge/Feature-Real--time%20Kick%20Analysis-000000?style=flat)
![Angle Detection](https://img.shields.io/badge/Feature-Angle%20Calculation-FF0000?style=flat)
![Kick Feedback](https://img.shields.io/badge/Feedback-Form%20%26%20Height-000000?style=flat)
![Platform](https://img.shields.io/badge/Platform-macOS-FF0000?logo=apple&logoColor=white&style=flat)
![Web App](https://img.shields.io/badge/Interface-Flask%20Web%20App-000000?style=flat)
![Repo Size](https://img.shields.io/github/repo-size/RayhanJamasi/taekwondo_kick_analyzer?color=FF0000&label=Repo%20Size&style=flat)
![File Count](https://img.shields.io/github/directory-file-count/RayhanJamasi/taekwondo_kick_analyzer?color=000000&label=Files&style=flat)

This is a program that I use to help improve my kicks. It utilizes packages like NumPy, MediaPipe, and OpenCV for an accurate analysis. It creates an outline of your body and analyzes the joints as you kick. This allows it to give accurate metrics and helps emphasize what I need to focus on to improve.

## Features

- **Full Body Detection**: Analyzes the entire body and creates a skeleton overlay for visualization of the user
- **Real-time Feedback**: Accurately gives feedback, metrics, and improvement steps in real time as you're kicking in a clear format
- **Dynamic Kick Height Detection**: Locates the chest height of the kicker and changes its values according to their height, allowing for proper calculations and feedback, no matter their height and proportions
-  **Angle Calculation**: Calculates the angle and extension for your joints, like the hips, knees, and ankle, to give accurate metrics about your form
-  **Speed Tracking**: Tracks the speed of your kick and helps ensure your movements are practical in a real taekwondo sparring match
-  **Adjustment Timer**: Has a 10-second visual timer when you start, as this allows the user to fix their position and ensure their full body is in frame
-  **Kick Logs**: A recording and metrics folder is created for each kick you make, so you can view it to evaluate your form and check old metrics

## Taekwondo Kick Analyzer Flow Chart

<img src="taekwondo_kick_flow.png" alt="Taekwondo Kick Analyzer Flow Chart" style="width:65%; height:65%"/>

## Understanding The Program

There are a few things that must be understood before using the program:
  1. **Adjustment**: This program was created to work for people of all heights. Due to this, I created a 10-second timer when the program starts that allows the user to get into position. During this time, the user needs to stand up straight, face the screen directly, and ensure all their limbs are shown. Once the timer is finished, a green dot will appear at your chest height. This green dot will be used as the threshold for your kicking height. A threshold is also created in relation to your feet. Because of this, it is important not to move closer/further from the screen as it can lead to inaccurate metrics.  
  2. **Kicking**: You can now turn 90 degrees in either direction and perform your kick. The program will start analyzing at the start of the kick and stop when the kick finishes. Please note that there is a 0.5-second cooldown time between analyses to prevent inaccurate results.
  3. **Results**: The latest kick metrics are shown in the web app. Select **Stop** to finish the session and save its summary.

## Quick Start

### 1. Set Up the Environment

Use Python 3.12. This project uses MediaPipe's legacy Solutions API, so MediaPipe is pinned to a compatible version.

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Run the Web App

```powershell
python app.py
```

Open http://127.0.0.1:5000, then select **Start** to use the webcam and **Stop** to finish a session. Camera access stays on this computer. Session summaries are stored in `data/sessions.db` and remain after restarting the app. Use **Sessions & videos** to pick a date, review kick metrics, play or download a saved clip, or delete a kick and its video. New kick videos and metrics are grouped by camera session under `data/sessions/`; older recordings under `data/` still appear in the archive.

## Status

Currently, there is a problem with the speed calculations. I am currently working to fix that, so please be patient!


## Future Features
  - **More Kicks**: Adding more kicks instead of just a front snap kick. This way, it can help give feedback for other kicks. 
  - **Better UI**: Update the UI to be clearer and have more configurations
  - **More Metric Calculations**: Get more metrics for better accuracy and feedback

## License

MIT License - see LICENSE file for details.


