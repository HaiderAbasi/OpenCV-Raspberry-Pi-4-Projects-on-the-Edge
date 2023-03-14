# This scritps saves the code

import cv2

def main():
    video_feed = cv2.VideoCapture(0)
    _,frame = video_feed.read()

    cv2.imwrite('picture.jpg', frame)
    print("\n\nSuccessfully saved picture from camera\n\n")
    video_feed.release()

if __name__ == '__main__':
    main()

