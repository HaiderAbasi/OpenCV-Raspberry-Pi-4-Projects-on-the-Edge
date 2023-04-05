import cv2
import numpy as np
import tensorflow as tf
import os
import time

main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
model_file = os.path.join(main_dir_path, 'models', 'mobile_net', 'mbnet_v1.tflite')
label_file = os.path.join(main_dir_path, 'models', 'mobile_net', 'labels.txt')
input_file = os.path.join(main_dir_path, 'data', 'road_recording.mkv')

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=model_file)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label map
with open(label_file, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Open video capture
cap = cv2.VideoCapture(input_file)

# Process each frame
frame_count = 0
total_time = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Get input shape
    input_shape = input_details[0]['shape'][1:3]

    # Preprocess image
    image = cv2.resize(frame, input_shape)
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.uint8)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    total_time += end_time - start_time

    # Get output tensors
    detection_locations = interpreter.get_tensor(output_details[0]['index'])
    detection_classes = interpreter.get_tensor(output_details[1]['index'])
    detection_scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = int(interpreter.get_tensor(output_details[3]['index']))

    # Postprocess results
    objects = []
    confidence_threshold = 0.5
    for i in range(num_detections):
        if detection_scores[0, i] > confidence_threshold:
            obj = {}
            obj['label'] = labels[int(detection_classes[0, i])]
            obj['prob'] = detection_scores[0, i]
            obj['rect'] = detection_locations[0, i]
            objects.append(obj)

    # Print classes and their count
    classes_count = {}
    for obj in objects:
        label = obj['label']
        if label not in classes_count:
            classes_count[label] = 1
        else:
            classes_count[label] += 1
    print(f'Frame {frame_count}: {classes_count}')

    # Measure total time for processing the frame
    print(f'Time taken to process frame {frame_count}: {end_time - start_time:.2f} seconds')

print(f'Average processing time per frame: {total_time/frame_count:.2f} seconds')

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
