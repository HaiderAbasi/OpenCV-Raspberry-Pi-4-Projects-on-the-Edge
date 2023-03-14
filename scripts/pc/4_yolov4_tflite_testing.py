import tensorflow as tf
import numpy as np
import cv2

try:
    from tensorflow import lite
    print("TensorFlow Lite module is installed")
except ImportError:
    print("TensorFlow Lite module is not installed")

# Load the YOLOv4-tiny-tflite model
model_path = '/home/luqman/OpenCV-Raspberry-Pi-4-Projects-on-the-Edge/models/yolov4/yolov4-416-fp32.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Define the input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the image
image_path = '/home/luqman/OpenCV-Raspberry-Pi-4-Projects-on-the-Edge/data/aeroplane.jpg'
image = cv2.imread(image_path)

# Resize the image to the input size expected by the model
input_shape = input_details[0]['shape'][1:3]
print("input_shape = ",input_shape)
resized_image = cv2.resize(image, tuple(input_shape))

# Preprocess the image
normalized_image = resized_image / 255.0
input_data = np.expand_dims(normalized_image, axis=0).astype(np.float32)

# Run the inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Postprocess the output
detections = output_data.squeeze()

# Draw the detected objects on the image
for detection in detections:
    print("detection = ",detection)
    class_id, score, x_min, y_min, x_max, y_max = detection
    if score > 0.5:
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, str(class_id), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the result
cv2.imshow('Object detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()