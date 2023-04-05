## Hardware Interfacing
This brnach contains hardware interfacing guidelines and version control for all porjects
# Where I Left :
- Yolov4 tflite not working
- Sdd performing detection (Y)

## Scripts
- **PC using tensorflow**
    - Opencv Testing with Video                    : *0_read_video.py*
    - TFLite inferencing on image                  : *1_tflite_inference_image.py*
    - SSD Model on Video and log infr speed        : *2_ssd_tflite_inference_video.py*
    - Yolo Model on Video and log infr speed       : *3_yolo_tflite_inference_video.py*
    - Yolo Model running on image                  : *4_yolov4_tflite_testing.py*
    - Mobilenet on Video on cli                    : *4_mbnet_cli_video.py*

- **PI using tf-runtime-lite**
    - Opencv Testing with camera                   : *0_picture_take.py*
    - Opencv Video Recording with camera           : *1_video_record.py*
    - TFlite Model running and log inference speed : *3_tflite_model_loading_testing.py*
    - Pytorch Model running                        : *4_pytorch_test.py*
    - SSD Model running on image                   : *5_inference_image.py*
    - SSD Model running on image Video             : *6_inference_video.py*


## Installations
- Ubuntu server 20.04 64bit
- sudo apt-get install python3-pip
- python3 -m pip install tflite-runtime
- pip install torch torchvision torchaudio
- pip install opencv-python
- sudo apt-get install ffmpeg libsm6 libxext6  -y
- pip install bytetracker




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