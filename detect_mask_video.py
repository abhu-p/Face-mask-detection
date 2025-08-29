# detect_mask_video.py

# Import necessary packages
import os
import cv2
import numpy as np
import argparse # Used for command-line arguments, though not strictly used for args in this specific script
from imutils.video import VideoStream # For accessing webcam/video stream
import imutils # For utility functions like resizing frames

# TensorFlow/Keras imports
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# --- Configuration ---
# Define the categories that your model was trained on
CATEGORIES = ["with_mask", "without_mask"]

# --- Utility Function: Mask Detection and Prediction ---
def detect_and_predict_mask(frame, faceNet, maskNet):
    """
    Detects faces in a video frame and predicts whether they are wearing a mask.

    Args:
        frame (np.array): The current video frame.
        faceNet (cv2.dnn_Net): The pre-trained face detection model.
        maskNet (tensorflow.keras.Model): The pre-trained mask detection model.

    Returns:
        tuple: A 2-tuple containing:
            - locs (list): A list of bounding box coordinates (startX, startY, endX, endY) for detected faces.
            - preds (list): A list of predictions (mask_probability, withoutMask_probability) for each face.
    """
    # Grab the dimensions of the frame
    (h, w) = frame.shape[:2]

    # Construct a blob from the frame for face detection
    # The values (104.0, 177.0, 123.0) are mean subtraction values used for the Caffe model.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    # Pass the blob through the face detection network to get face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Initialize lists to store detected faces, their locations, and mask predictions
    faces = []
    locs = []
    preds = []

    # Loop over the raw detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than a minimum threshold
        if confidence > 0.5: # Confidence threshold for face detection
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding box coordinates fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endY), min(h - 1, endY)) # Corrected potential bug: endX should be min(w-1, endX)

            # Extract the face Region of Interest (ROI)
            face = frame[startY:endY, startX:endX]

            # Check if the face ROI is empty (e.g., if coordinates are invalid)
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue # Skip to the next detection if the ROI is empty

            # Convert the face ROI from BGR (OpenCV default) to RGB channel ordering
            # Resize it to 224x224 (required input size for MobileNetV2)
            # And preprocess it using MobileNetV2's specific preprocessing function
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # Add the processed face and its bounding box to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # Only make predictions if at least one face was detected
    if len(faces) > 0:
        # For faster inference, we'll make batch predictions on *all*
        # detected faces at the same time, instead of one-by-one.
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32) # Batch size for predictions

    # Return the face locations and their corresponding mask predictions
    return (locs, preds)

# --- Main Script Execution ---
if __name__ == "__main__":
    # Define paths to the pre-trained face detector models
    # Ensure these files are in a 'face_detector' subfolder relative to this script
    prototxtPath = os.path.join("face_detector", "deploy.prototxt")
    weightsPath = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")

    print("[INFO] Loading face detector model...")
    # Check if face detector model files exist
    if not os.path.exists(prototxtPath) or not os.path.exists(weightsPath):
        print(f"[ERROR] Face detector model files not found.")
        print(f"Expected at: {os.path.abspath('face_detector')}")
        print("Please ensure 'deploy.prototxt' and 'res10_300x300_ssd_iter_140000.caffemodel' are in the 'face_detector' folder.")
        exit() # Exit the script if files are missing

    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # Load the pre-trained face mask detector model from disk
    # Ensure 'mask_detector.model' is in the same directory as this script
    mask_model_path = "mask_detector.model"
    print(f"[INFO] Loading mask detector model from: {os.path.abspath(mask_model_path)}")
    # Check if mask detector model file exists
    if not os.path.exists(mask_model_path):
        print(f"[ERROR] Mask detector model file not found at: {os.path.abspath(mask_model_path)}")
        print("Please ensure 'mask_detector.model' is in the same directory as this script.")
        exit() # Exit the script if the file is missing

    maskNet = load_model(mask_model_path)

    # Initialize the video stream (e.g., from your default webcam)
    print("[INFO] Starting video stream...")
    vs = VideoStream(src=0).start() # src=0 usually refers to the default webcam

    # Loop over the frames from the video stream
    while True:
        # Grab the current frame from the threaded video stream
        frame = vs.read()

        # Check if a frame was successfully read
        if frame is None:
            print("[WARNING] Failed to grab frame, trying again...")
            continue # Skip this iteration and try to read the next frame

        # Resize the frame to have a maximum width of 400 pixels for faster processing
        frame = imutils.resize(frame, width=400)

        # Detect faces in the frame and determine if they are wearing a face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # Loop over the detected face locations and their corresponding predictions
        for (box, pred) in zip(locs, preds):
            # Unpack the bounding box coordinates and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred # Assuming the model outputs probabilities for 'mask' and 'without_mask'

            # Determine the class label and color based on the higher probability
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255) # Green for Mask, Red for No Mask

            # Include the prediction probability in the label text
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # Display the label and bounding box rectangle on the output frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # Show the output frame with detections
        cv2.imshow("Mask Detector", frame) # Window title
        key = cv2.waitKey(1) & 0xFF # Wait for 1ms for a key press

        # If the 'q' key was pressed, break from the loop to exit
        if key == ord("q"):
            break

    # Perform a bit of cleanup when the loop exits
    print("[INFO] Cleaning up...")
    cv2.destroyAllWindows() # Close all OpenCV windows
    vs.stop() # Stop the video stream

    print("✅ Script finished successfully!")