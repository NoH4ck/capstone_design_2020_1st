import cv2

image = cv2.imread('sample2.jpg')

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

eyes = eye_cascade.detectMultiScale(image)
for eye in eyes:
    (x, y, w, h) = eye
    x1, y1, x2, y2 = x, y, x + w, y + h
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

cv2.imshow('image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
