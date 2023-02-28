import tflite_runtime.interpreter as tflite
import numpy as np
import time

interpreter = tflite.Interpreter(model_path='')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
##Inference speed code
input_image = np.zeros((1, 128, 128, 1), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_image)

start_time = time.time()
for i in range(1000):
    interpreter.invoke()
    inference_time = (time.time() - start_time) / 1000
    print("Inference time: ", inference_time, "seconds")