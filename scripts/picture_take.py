# Python program to explain cv2.imwrite() method

import cv2

# Script to test if camera is working or not

def main():
    video_feed = cv2.VideoCapture(0)
    # while(1):
    _,frame = video_feed.read()

    cv2.imwrite('image', frame)
    print("\n\nSuccessfully read image from camera\n\n")
    cv2.imshow("camera_feed.jpg", frame)
    cv2.waitKey(1)


if __name__ == '__main__':
    main()

