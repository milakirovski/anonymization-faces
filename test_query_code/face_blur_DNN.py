import cv2
import numpy as np
import os

modelFile = "../models/DNNFaceDetectorOpenCV/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "../models/DNNFaceDetectorOpenCV/deploy.prototxt"

net = cv2.dnn.readNetFromCaffe(configFile, modelFile) # creates deep learning network based on the files from above

folder_path = '../dataset/Dataset_LFPW'

output_folder_detect = '../output/faces_detected_DNN'
output_folder_anonymize = '../output/faces_anonymized_DNN'


if not os.path.exists(output_folder_detect) and not os.path.exists(output_folder_anonymize):
    os.makedirs(output_folder_detect)
    os.makedirs(output_folder_anonymize)


for image_name in os.listdir(os.path.join(folder_path)):
    print(image_name)

    image_path = os.path.join(folder_path, image_name)

    if os.path.isfile(image_path):
        image = cv2.imread(image_path)
        height, width = image.shape[:2] # Extract h and w from the image
        image_rect = np.copy(image)

        if image is None:
            print("Could not read the image: {}".format(image_path))
            continue

        # blob -> is a standard preprocessing step to normalize the image to the config standard before passing it into the network.
        # Scales the pixel values (1.0 is the scaling factor, meaning no scaling here).
        # Subtracts the mean values (104.0, 117.0, 123.0) from each channel

        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 117.0, 123.0))

        # Sets the preprocessed image as input to the neural network
        net.setInput(blob)

        # Run the neural network to detect faces:
        faces = net.forward()
        # net.forward() = Performs a forward pass through the network to get the face detections.
        # The faces variable contains the detection results, including the confidence scores and bounding boxes for detected faces.


        #to draw faces on image
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2] # Extracts the confidence score for the i-th detected face.

            # If the confidence is greater than 0.5, the face is considered valid.
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([width, height, width, height]) # The bounding box for the face is normalized between 0 and 1, so it's scaled back to the original image size using the width and height.

                (x, y, x1, y1) = box.astype("int") #converts the bounding box coordinates from floating point to integers.

                # draw rectangle
                cv2.rectangle(image_rect, (x, y), (x1, y1), (0, 0, 255), 2)

                # blur
                face_roi = image[y:y1, x:x1]
                blurred_face = cv2.GaussianBlur(face_roi, (51, 51),0)
                image[y:y1, x:x1] = blurred_face


                # save the result from detection
                output_path_detection = os.path.join(output_folder_detect, "dect_" + image_name)
                cv2.imwrite(output_path_detection, image_rect)

                # save the result from the blur
                output_path_anonymize = os.path.join(output_folder_anonymize, "blur_" + image_name)
                cv2.imwrite(output_path_anonymize, image)

print("See the results in the output file =)")
