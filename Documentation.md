## Hardware Interfacing
This brnach contains hardware interfacing guidelines and version control for all porjects

## Scripts
- Opencv Testing with camera                   : *picture_take.py*
- Opencv Video Recording with camera           : *video_record.py*
- TFlite Model running on wiht inference speed : *tflite_model.py*
- Pytorch Model running                        : *pytorch_test.py*

## Installations
- Ubuntu server 20.04 64bit
- sudo apt-get install python3-pip
- python3 -m pip install tflite-runtime
- pip install torch torchvision torchaudio
- pip install opencv-python
- sudo apt-get install ffmpeg libsm6 libxext6  -y



## Processes
- Enable ssh and wifi settings before burning OS on SD Card using Rpi-Imager
- Enabling Camera
    ```
    sudo nano /boot/firmware/config.txt
    ## add start_x=1 before all
    ##press ctrl+x and then y and enter
    ```
    - Then to check ls /dev/video
    - If you have *video0* then camera is enabled

## Testing Camera
- Install v4l2 driver
    ```
    sudo apt-get install v4l-utils
    ```
- To check if video 0 is availble
    ```
    v4l2-ctl --list-devices
    ```
- Libraries
```
python3 -c "import cv2; print(cv2.__version__)"
python3 -c "import torch; print(torch.__version__)"
<!-- python3 -c "import tflite_runtime.interpreter; print(tflite_runtime.interpreter.__version__)" -->
```

## Efficiencies
- Installating ncnn
## Errors
- ImportError: libGL.so.1:
    ```
    sudo apt-get install ffmpeg libsm6 libxext6  -y
    ```