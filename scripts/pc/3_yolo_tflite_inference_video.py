import numpy as np
import os
import cv2
import tensorflow as tf
import time

main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
model_file = os.path.join(main_dir_path, 'models', 'yolov4', 'yolov4-416-fp32.tflite')
label_file = os.path.join(main_dir_path, 'models', 'yolov4', 'labelmap.txt')
video_file = os.path.join(main_dir_path, 'data', 'road_recording.mkv')

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
    input_image = cv2.resize(frame, (416, 416))
    input_image = input_image.reshape(1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
    input_image = input_image.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the indices of the output tensor for class, scores, and boxes
    class_indices = [i for i in range(len(output_data[0])) if output_data[0][i][0] > 0.5]
    score_indices = [i for i in range(len(output_data[0])) if output_data[0][i][1] > 0.5]
    box_indices = [i for i in range(len(output_data[0])) if len(output_data[0][i]) >= 6 and output_data[0][i][2] > 0.5]

    # Draw boxes and labels around detected objects
    for i in box_indices:
        class_id = int(output_data[0][i][0])
        score = output_data[0][i][1]
        x1, y1, x2, y2 = output_data[0][i][2:6]
        x1 = int(x1 * frame.shape[1])
        y1 = int(y1 * frame.shape[0])
        x2 = int(x2 * frame.shape[1])
        y2 = int(y2 * frame.shape[0])
        label = label_array[class_id].rstrip()
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
        cv2.putText(frame, "{}: {:.2f}".format(label, score), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)


    # Display the frame with bounding boxes and labels
    cv2.imshow('Frame', frame)

    # Press Q to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
