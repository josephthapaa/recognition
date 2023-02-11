import cv2


def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (200,200))
    
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        
        # Predicting the id of the user
        id, _ = clf.predict(cv2.resize((gray_img[y:y+h, x:x+w]), (200,200)))
        # Check for id of user and label the rectangle accordingly
        if id==1:
            cv2.putText(img, "Joseph", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        elif id==2:
            cv2.putText(img, "ronaldo", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]

    return coords

# Method to recognize the person
def recognize(img, clf, faceCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color["white"], "Face", clf)
    print(coords)
    return img


# Loading classifier
faceCascade = cv2.CascadeClassifier('C:\\Users\\hp\\Desktop\\Face-Detection-Recognition-Using-OpenCV-in-Python-master\\haarcascade_frontalface_default.xml')

# Loading custom classifier to recognize
clf = cv2.face.EigenFaceRecognizer_create()
clf.read("C:\\Users\\hp\\Desktop\\Face-Detection-Recognition-Using-OpenCV-in-Python-master\\classifier1.xml")

# Capturing real time video stream. 0 for built-in web-camsq, 0 or -1 for external web-cams
video_capture = cv2.VideoCapture(0)

while True:
    # Reading image from video stream
    _, img = video_capture.read()
    # Call method we defined above
    img = recognize(img, clf, faceCascade)
    # Writing processed image in a new window
    cv2.imshow("face detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing web-cam
video_capture.release()
# Destroying output window
cv2.destroyAllWindows()