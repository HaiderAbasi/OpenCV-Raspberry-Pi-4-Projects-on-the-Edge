## Run it
## python video_record.py file.avi 4
import cv2, sys

if len(sys.argv) < 3:
    sys.exit("Please specify the output filename and recording time (in seconds) as arguments.")

filename, recording_time = sys.argv[1:3]
recording_time = int(recording_time)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("Error reading video file")

size = (int(cap.get(3)), int(cap.get(4)))
out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'MJPG'), 25, size)

start_time = cv2.getTickCount()

while cap.isOpened() and (cv2.getTickCount() - start_time) / cv2.getTickFrequency() <= recording_time:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"The video was successfully saved as {filename}")
