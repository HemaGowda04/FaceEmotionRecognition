import cv2
import cv2
from keras.models import model_from_json
import numpy as np
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
def extract_features(image):
    image = cv2.resize(image, (48, 48))
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0
def load_cnn_model():
    json_file = open("facialemotionmodel.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("facialemotionmodel.h5")
    return model
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
model = load_cnn_model()
webcam = cv2.VideoCapture(0)
while True:
    ret, frame = webcam.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    try:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            img = extract_features(face)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(frame, '%s' % (prediction_label), (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) == 27:
            break
    except cv2.error:
        pass
webcam.release()
cv2.destroyAllWindows()