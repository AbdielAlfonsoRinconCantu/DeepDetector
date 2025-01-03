#!/bin/bash

gnome-terminal --title="launcher_3.sh" -- bash -c "
echo 'cd /';
cd /;

echo 'cd /home/auto/DeepDetector';
cd /home/auto/DeepDetector;

echo 'sleep 5';
sleep 5;

echo 'v4l2-ctl';
v4l2-ctl --set-ctrl=brightness=128 \
          --set-ctrl=contrast=128 \
          --set-ctrl=saturation=128 \
          --set-ctrl=white_balance_automatic=1 \
          --set-ctrl=gain=0 \
          --set-ctrl=power_line_frequency=2 \
          --set-ctrl=white_balance_temperature=3550 \
          --set-ctrl=sharpness=128 \
          --set-ctrl=backlight_compensation=1 \
          --set-ctrl=auto_exposure=1 \
          --set-ctrl=exposure_time_absolute=20 \
          --set-ctrl=exposure_dynamic_framerate=1 \
          --set-ctrl=pan_absolute=-10543 \
          --set-ctrl=tilt_absolute=-9778 \
          --set-ctrl=focus_automatic_continuous=0 \
          --set-ctrl=focus_absolute=38 \
          --set-ctrl=zoom_absolute=100;

echo 'python mainUIdetector_v4_Ab_6.py';
python mainUIdetector_v4_Ab_6.py;

bash;
"
