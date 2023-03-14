import cv2

cap = cv2.VideoCapture('/home/luqman/OpenCV-Raspberry-Pi-4-Projects-on-the-Edge/data/a.mkv')
cap.set(cv2.CAP_PROP_FPS, 10)
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Do some processing on the frame
    # processed_frame = some_function(frame)

    # Display the processed frame
    cv2.imshow('Processed Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
