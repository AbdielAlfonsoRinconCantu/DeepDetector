# DeepDetector #
A tkinter interface to track objects and count once they've reached a specified region, graphs displaying performance against an hourly-based set goal are displayed, and dynamic pre-trained recognition models are supported.

# Before you install #

This project was tested on a Jetson Orin Nano running Jetson Linux paired with a USB camera, other setups may not be directly supported.

Make sure the following is installed:

    $ python3 --version
    Python 3.10.16

    $ pip --version
    pip 24.3.1 from /usr/lib/python3/dist-packages/pip (python 3.10)

    $ cd ~/DeepDetector/DeepDetector/

    $ sudo apt update
    $ sudo apt install -y $(cat packages.list)

    $ pip install -r requirements.txt 

Additional notes:
- Ensure CUDA drivers are installed if using a Jetson.
- OpenCV must be built with GStreamer enabled.
- Your system's date and time will be used.

# Installation #

Using git:

    git clone https://github.com/AbdielAlfonsoRinconCantu/DeepDetector.git

# Usage #
Run standalone:

    $ cd ~/DeepDetector/DeepDetector/
    $ python3 mainUIdetector_v4_Ab_6.py

Set camera parameters and run:

    $ cd ~/DeepDetector/
    $ ./launcher_3.sh

**Note:** camera parameters vary based on device, run `$ v4l2-ctl -l` on a terminal to see available controls and modify `launcher_3.sh` accordingly if desired, alternatively, leave the default settings or set camera controls using the terminal or the software of your choosing.
