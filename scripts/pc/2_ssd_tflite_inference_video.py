import numpy as np
import os
import cv2
import tensorflow as tf
import time
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
model_file = os.path.join(main_dir_path, 'models', 'ssd_model', 'lite-model_ssd_mobilenet_v1_100_320_uint8_nms_1.tflite')
label_file = os.path.join(main_dir_path, 'models', 'ssd_model', 'labels.txt')
video_file = os.path.join(main_dir_path, 'data', 'a.mkv')

# Loading labels
text_file = open(label_file, "r")
label_array = text_file.readlines()

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_file)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Open the video file
cap = cv2.VideoCapture(video_file)

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_image = cv2.resize(frame, (320, 320))
    input_image = input_image.reshape(1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
    input_image = input_image.astype(np.uint8)

    # Run inference on the frame
    input_data = np.array(input_image, dtype=np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # Time measuring
    start_time = time.time()
    interpreter.invoke()
    predicted_labels = interpreter.get_tensor(output_details[1]['index'])
    inferencing_time = time.time() - start_time

    bounding_boxes = interpreter.get_tensor(output_details[0]['index'])
    predicted_scores = interpreter.get_tensor(output_details[2]['index'])

    if(predicted_labels[0][0]>200): # some times gives a very huge number -> causing error list index out of bound
            predicted_labels[0][0]=0


    for i in range(5):

         print("Score : " , predicted_scores[0][i] ," Label : " ,label_array[int(predicted_labels[0][i])] , bounding_boxes[0][0] )

    print("Time for Inference : ",inferencing_time)
    print("-"*30)


    # Display the results
    # cv2.putText(frame, top_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
