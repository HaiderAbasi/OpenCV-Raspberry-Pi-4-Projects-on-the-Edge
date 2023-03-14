import numpy as np
import os
import cv2
import tensorflow as tf
import time
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
model_file = os.path.join(main_dir_path, 'models', 'yolov4', 'yolov4-416-fp32.tflite')
label_file = os.path.join(main_dir_path, 'models', 'yolov4', 'labelmap.txt')
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
    input_image = cv2.resize(frame, (416, 416))
    input_image = input_image.reshape(1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
    input_image = input_image.astype(np.uint8)

    # Run inference on the frame
    input_data = np.array(input_image, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # Time measuring
    start_time = time.time()
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    detections = output_data.squeeze()
    # class_id, score, x_min, y_min, x_max, y_max = detections[0]

    predicted_labels = interpreter.get_tensor(output_details[1]['index'])
    inferencing_time = time.time() - start_time
    print("-"*10)
    print(predicted_labels.shape)
    print(predicted_labels)
    print(output_data.shape)
    print(output_data)
    print("*"*10)

    # print(interpreter.get_tensor(output_details[0]),"\n")
    # print(interpreter.get_tensor(output_details[1]),"\n")
    # predicted_scores = interpreter.get_tensor(output_details[2]['index'])

    # if(predicted_labels[0][0]>200): # some times gives a very huge number -> causing error list index out of bound
    #         predicted_labels[0][0]=0


    # print([f"{label_array[int(predicted_labels[0][i])]}:{predicted_scores[0][i]}" for i in range(5)])

    # print("Time for Inference : ",inferencing_time)
    # print("-"*30)


    # Display the results
    # cv2.putText(frame, top_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1) == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()
