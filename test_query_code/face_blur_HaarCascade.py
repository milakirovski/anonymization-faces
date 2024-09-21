import cv2
import os
import numpy as np

#Declare face Detector
haar_cascade = cv2.CascadeClassifier('../models/haarCascadesClassifier/haarcascade_frontalface_default.xml')

# put all the images into a list
folder_path = '../dataset/Dataset_LFPW'

output_folder_detect = '../output/faces_detected_haar'
output_folder_anonymize = '../output/faces_anonymized_haar'


# Ensure the output folder exists
if not os.path.exists(output_folder_detect) and not os.path.exists(output_folder_anonymize):
    os.makedirs(output_folder_detect)
    os.makedirs(output_folder_anonymize)

i = 1

for image_name in os.listdir(os.path.join(folder_path)):
    print(image_name)
    # Construct the full image path
    image_path = os.path.join(folder_path, image_name)

    if os.path.isfile(image_path):
        #read the image
        image = cv2.imread(image_path)
        image_ract = np.copy(image) # copy of the image to detect the face and save the image later with the rectangle

        # Check if the image was successfully loaded
        if image is None:
            print("Could not read the image: {}".format(image_path))
            continue

        # convert to gray for the detector
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_rectangle = haar_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=7) # Returns a list of rectangular coordinates for each detected face
        for (x, y, w, h) in face_rectangle:
            # x and y are the coordinates of the top-left corner of the rectangle.
            # w - width of the rectangle
            # h - height of the rectangle
            # draw rectangle around the roi
            cv2.rectangle(image_ract, (x, y), (x + w, y + h), (255, 0, 0), thickness=4)

            # save the result from detection
            output_path_detection = os.path.join(output_folder_detect, "dect_" + image_name)
            cv2.imwrite(output_path_detection, image_ract)

            #BLURRING
            # Extract the face ROI (Region of Interest)
            face_roi = image[y:y + h, x:x + w]

            # Apply averaging blur
            blurred_face = cv2.blur(face_roi, (51, 51))  # (99, 99) is the kernel size

            # Replace the original face ROI with the blurred version
            image[y:y + h, x:x + w] = blurred_face

            # save the result from blurring
            output_path_anonymize = os.path.join(output_folder_anonymize, "anon_" + image_name)
            cv2.imwrite(output_path_anonymize, image)

print("See the results in the output file =)")


