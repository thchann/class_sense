import cv2

# Try different indexes: 1, 2, 3... until it works
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Couldn't read from OBS Virtual Camera")
        break

    cv2.imshow("OBS Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
