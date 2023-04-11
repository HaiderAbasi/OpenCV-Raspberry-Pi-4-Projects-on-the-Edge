import numpy as np
import os
import cv2
from tflite_runtime.interpreter import Interpreter
import time

main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
model_file = os.path.join(main_dir_path, 'models', 'yolov4', 'yolov4-416-fp32.tflite')
video_file = os.path.join(main_dir_path, 'data', 'road_recording.mkv')

# Load the TFLite model and allocate tensors.
interpreter = Interpreter(model_path=model_file)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open the video file
cap = cv2.VideoCapture(video_file)

frame_count = 0
total_time = 0

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Preprocess the frame
    input_image = cv2.resize(frame, (416, 416))
    input_image = input_image.reshape(1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
    input_image = input_image.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_image)

    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    total_time += end_time - start_time

    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Postprocess results
    objects = []
    confidence_threshold = 0.5
    for i in range(len(output_data[0])):
        if output_data[0][i][1] > confidence_threshold:
            obj = {}
            obj['class_id'] = int(output_data[0][i][0])
            obj['prob'] = output_data[0][i][1]
            obj['rect'] = output_data[0][i][2:6]
            objects.append(obj)

    # Print classes and their count
    classes_count = {}
    for obj in objects:
        class_id = obj['class_id']
        if class_id not in classes_count:
            classes_count[class_id] = 1
        else:
            classes_count[class_id] += 1
    total_class = len(classes_count)
    print(f'Time taken to process frame {frame_count}: {end_time - start_time:.2f} seconds')
    # print(f'Frame {frame_count}: {len(classes_count)}')

# Write code to print the number of classes and their count
    print(f'Frame {frame_count}: {total_class}')
    # for class_id in classes_count:
    # print(f'Class {class_id}: {classes_count[class_id]}')



    # Measure total time for processing the frame

    # Display the frame with bounding boxes and labels
    # cv2.imshow('Frame', frame)

    # # Press Q to exit
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break

print(f'Average processing time per frame: {total_time/frame_count:.2f} seconds')

cap.release()
cv2.destroyAllWindows()
