INSTALL:
- Download or clone this repo 
- It is advised to create a new Anaconda environment with python version 3.6. On Anaconda Prompt type **conda create -n ets2 python=3.6**
- To activate new environment, use **conda activate ets2**
- Navigate to the downloaded repo and on the Anaconda Prompt, run **pip install -r requirements.txt** without quotes
- Download the model from http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz and extract in this repo's directory
- Follow the instructions on the following link for ETS 2 game settings https://youtu.be/eNt8YzhUZaY?t=433 [REFER TO RECOMMENDED SETTINGS SECTION BELOW]
- Run tacc.py for CARS and tacc-trucks.py for TRUCKS [REFER TO RECOMMENDED VEHICLES SECTION BELOW]

RECOMMENDED SETTINGS:
- Resolution should be full screen 1920x1080
- Keys 'W' and 'S' should be set to 'increase/decrease cruise control speed'
- Keys 'A' and 'D' should be set as the secondary controls for 'Steerng left/right'
- Keys 'Z' and 'X' should be set to 'Left/right turn indicator'
- Refer to "steering.jpg" for steering settings for ETS 2
- Truck Speed Limiter should be ON
- Automatic gearbox recommended 

HOW TO USE:
- Before starting the game, open an Anaconda command prompt and navigate (cd) to your 'ets2' folder (this downloaded repo)
- Start ETS 2 and load your game
- Alt-Tab out to the Anaconda Prompt
- Type **python tacc.py** and press Enter
- Go back into the game, play as normal, and activate TACC [REFER TO USGE IN-GAME SECTION BELOW]

USAGE IN-GAME:
- CARS ONLY: To activate/deactivate TACC with Autosteer ("Autopilot"), press C
- CARS ONLY: To change lane while Autosteer is active, press Z or X on a straight freeway
- CARS & TRUCKS: To activate/deactivate TACC without Autosteer, press N

RECOMMENDED VEHICLES:
- CARS: Tesla Cybertruck, Tesla Model S, Ford Transit
- TRUCKS: DAF

FILES:
- directkeys.py: Exports keycodes for use in tacc.py
- grabscreen.py: Library 
- tacc.py: Main script for use with cars (full "Autopilot")
- tacc-trucks.py: Main script for use with trucks (TACC only)
- PYTHONPATH.txt: Adds Tensorflow to PATH, must be run before main script
- steering.jpg: Recommended steering settings in ETS
- .wav files: Sounds used in Freeway Assist

