## Hardware Interfacing
This brnach contains hardware interfacing guidelines and version control for all porjects



## Installations
-
## Software
- Ubuntu server 22.04 32bit
-


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
