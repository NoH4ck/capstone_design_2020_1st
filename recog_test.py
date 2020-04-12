import tensorflow as tf
import cv2

# -----
'''
[nltk_data] Error loading punkt: <urlopen error [SSL:
[nltk_data]     CERTIFICATE_VERIFY_FAILED] certificate verify failed
[nltk_data]     (_ssl.c:777)>
http://blog.pengyifan.com/how-to-fix-python-ssl-certificate_verify_failed/
'''
import os
import ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context
# ------

file_path = tf.keras.utils.get_file(
    '24.jpg', 'https://github.com/kairess/age_gender_estimation/raw/master/img/24.jpg')
image = cv2.imread(file_path)

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
