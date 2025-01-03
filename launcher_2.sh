#!/bin/bash

gnome-terminal --title="launcher_2.sh" -- bash -c "
echo 'cd /';
cd /;

echo 'cd /home/auto/DeepDetector';
cd /home/auto/DeepDetector;

echo 'sleep 5';
sleep 5;

# sh cameraSettings.sh;
# sleep 5;
# sh cameraSettings.sh;

echo 'python mainUIdetector_v4_Ab_6.py';
python mainUIdetector_v4_Ab_6.py;

bash;
"
