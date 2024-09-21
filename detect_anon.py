import os
import cv2
import sys
import numpy as np

#Declare face Detectors
haar_cascade = cv2.CascadeClassifier('./models/haarCascadesClassifier/haarcascade_frontalface_default.xml')
modelFile = "./models/DNNFaceDetectorOpenCV/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "./models/DNNFaceDetectorOpenCV/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile) # creates deep learning network based on the files from above

if len(sys.argv) < 2:
    print("Usage: python script.py <image_name>")
    sys.exit(1)

# Get the image name from the command-line arguments
image_name = sys.argv[1]

# dataset folder
image_folder = "dataset/Dataset_LFPW"
image_path = os.path.join(image_folder, image_name)

if not os.path.isfile(image_path):
    print("Image {} doesn't exist!".format(image_name))
    sys.exit(1)

# output folder
output_folder = './output'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_haar = cv2.imread(image_path)
image_ddn = image_haar.copy() # copy so we can test the other model

# Haar cascade detector and blur

image_gray = cv2.cvtColor(image_haar, cv2.COLOR_BGR2GRAY)# convert to gray for the Haar detector

face_rectangle = haar_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=7) # Returns a list of rectangular coordinates for each detected face

for (x, y, w, h) in face_rectangle:
    face_roi = image_haar[y:y + h, x:x + w]
    blurred_face = cv2.blur(face_roi, (51, 51))
    image_haar[y:y + h, x:x + w] = blurred_face
    output_haar = os.path.join(output_folder, "haar_anon_" + image_name)
    cv2.imwrite(output_haar, image_haar)
    cv2.imshow('haar_avg_blur_image', image_haar)

# DDN and Gaussian blur
blob = cv2.dnn.blobFromImage(cv2.resize(image_ddn, (300, 300)), 1.0,(300, 300), (104.0, 117.0, 123.0))
net.setInput(blob)
faces = net.forward()

for i in range(faces.shape[2]):
    confidence = faces[0, 0, i, 2]

    if confidence > 0.5:
        height, width = image_ddn.shape[0:2]  # Extract h and w from the image
        box = faces[0, 0, i, 3:7] * np.array([width, height, width, height]) # The bounding box coordinates are normalized values relative to the image dimensions.
        (x, y, x1, y1) = box.astype("int")
        # blur
        face_roi = image_ddn[y:y1, x:x1]
        blur = cv2.GaussianBlur(face_roi, (51, 51), 0)
        image_ddn[y:y1, x:x1] = blur
        # save the results
        output_ddn = os.path.join(output_folder, "ddn_anon_" + image_name)
        cv2.imwrite(output_ddn, image_ddn)
        cv2.imshow('ddn_gaussian_blur_image', image_ddn)

print("See the results in the older output =)")
cv2.waitKey(0)
cv2.destroyAllWindows()




