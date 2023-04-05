import cv2
import numpy as np
import tensorflow as tf
import os
main_dir_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))
model_file = os.path.join(main_dir_path,'models','mobile_net','mbnet_v1.tflite')
input_file = os.path.join(main_dir_path,'data','kite.jpg')
label_file = os.path.join(main_dir_path,'models','mobile_net','labels.txt')

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=model_file)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load input image
src = cv2.imread(input_file)
cam_width = src.shape[1]
cam_height = src.shape[0]

# Preprocess image
image = cv2.resize(src, (300, 300))
image = np.expand_dims(image, axis=0)
image = image.astype(np.uint8)

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], image)

# Run inference
interpreter.invoke()

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
        obj['label'] = int(detection_classes[0, i]) + 1
        obj['prob'] = detection_scores[0, i]
        y1, x1, y2, x2 = detection_locations[0, i]
        obj['rect'] = {
            'x': int(x1 * cam_width),
            'y': int(y1 * cam_height),
            'width': int((x2 - x1) * cam_width),
            'height': int((y2 - y1) * cam_height)
        }
        objects.append(obj)

# Display results
for obj in objects:
    if obj['rect']['width'] * obj['rect']['height'] > 20:
        color = (255, 0, 0)
        cv2.putText(src, str(obj['label']), (obj['rect']['x'], obj['rect']['y'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.rectangle(src, (obj['rect']['x'], obj['rect']['y']), (obj['rect']['x'] + obj['rect']['width'], obj['rect']['y'] + obj['rect']['height']), color, 2)

# Show image
cv2.imshow("Output", src)
cv2.waitKey(0)