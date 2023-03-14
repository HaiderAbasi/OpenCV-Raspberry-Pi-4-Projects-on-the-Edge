import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import sys

def main():
    # interpreter = tf.lite.Interpreter('detect.tflite') ## To use on PC with tf install and not tflite
    interpreter = tflite.Interpreter('detect.tflite') # load Model using tflite on RPI
    input_details   = interpreter.get_input_details() # understand model input requirments
    output_details = interpreter.get_output_details() # understand model outputs
   # video_feed = cv2.VideoCapture(0) # to take video feed from camera
    video_feed = cv2.VideoCapture('Walkers.mp4')

    # To check frame sizes before inputting to model
    # print(video_feed.get(cv2.CAP_PROP_FRAME_WIDTH)," / ",video_feed.get(cv2.CAP_PROP_FRAME_HEIGHT) )
    while(1):
        _,frame = video_feed.read()
        input_image = cv2.resize(frame,(300,300))
        input_image = np.expand_dims(input_image,axis=0)
        input_image = input_image.astype(np.uint8)
        # print("\n\n ",input_image.shape,"\n\n")


        # setting model inputs
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'] , input_image)
        start_time = time.time() # measuring time

        interpreter.invoke()# Running model on this line
        predictions = interpreter.get_tensor(output_details[0]['index'])

        print("Inference time: {:.3f} seconds".format(time.time() - start_time))

        prediction = np.argmax(predictions)
        print(prediction)

        # cv2.imshow("Full Frame", frame)
        # cv2.waitKey(1)


if __name__ == '__main__':
    main()