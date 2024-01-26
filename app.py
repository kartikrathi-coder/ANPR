import streamlit as st
from PIL import Image
import onnx
import easyocr
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import pytesseract as pt
import plotly.express as px
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xet
import pytesseract

from glob import glob
from skimage import io
from shutil import copy
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#Define the Image height and width
INPUT_WIDTH =  640
INPUT_HEIGHT = 640

# load YOLO model
net = cv2.dnn.readNetFromONNX('/Users/kartikrathi/Documents/ANPR_Project/yolov5/runs/train/Model/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def get_detections(img, net):
    # 1. CONVERT IMAGE TO YOLO FORMAT
    image = np.array(img)
    
    # Ensure the image has three channels (remove alpha channel if present)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    
    row, col, _ = image.shape  # Extract row, column, and channels of image

    max_rc = max(row, col)  # Calculate max number of rows and cols
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col, :] = image  # Assign values to the channels

    # 2. GET PREDICTION FROM YOLO MODEL
    blob = cv2.dnn.blobFromImage(input_image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]

    return input_image, detections


def non_maximum_supression(input_image,detections):
    
    '''
    This function takes the preprocessed image (input_image) and the raw detections obtained from the YOLO model 
    (detections) and performs non-maximum suppression (NMS) to filter out redundant detections.
    '''
    
    # 3. FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE
    
    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH #x_factor and y_factor are scaling factors to adjust bounding box coordinates based on the original image dimensions.
    y_factor = image_h/INPUT_HEIGHT

    
    #The confidence score indicates how sure the model is that the box contains an object and also how accurate it thinks the box is that predicts
    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5] # probability score of license plate
            if class_score > 0.25:
                cx, cy , w, h = row[0:4]

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])

                confidences.append(confidence)
                boxes.append(box)

    # CLEAN
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    
    # NMS
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)
    
    return boxes_np, confidences_np, index

def drawings(image, boxes_np, confidences_np, index):
    # Drawings
    image_draw = image.copy()
    image_draw = np.array(image_draw)# Create a copy to avoid modifying the original image
    for ind in index:
        x, y, w, h = boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf * 100)

        cv2.rectangle(image_draw, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 255), 2)
        cv2.rectangle(image_draw, (int(x), int(y-30)), (int(x+w), int(y)), (255, 0, 255), -1)
        cv2.rectangle(image_draw, (int(x), int(y+h)), (int(x+w), int(y+h+25)), (0, 0, 0), -1)

        cv2.putText(image_draw, conf_text, (int(x), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return image_draw

def yolo_predictions(img, net):
    # Step 1: Get detections
    input_image, detections = get_detections(img, net)
    
    # Step 2: Non-Maximum Suppression
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    
    # Step 3: Drawings
    result_img = drawings(img, boxes_np, confidences_np, index)
    
    # Extract bounding box coordinates
    bounding_box_coords = [boxes_np[i] for i in index]
    
    return result_img, bounding_box_coords

#Crop the image 
def crop_image(img, x, y, width, height):
        
    # Convert the image array to a Pillow Image object
    pil_img = Image.fromarray(img)

    # Crop the image using the provided coordinates
    cropped_img = pil_img.crop((x, y, x + width, y + height))

    # Convert the cropped image back to a NumPy array
    cropped_img_array = np.array(cropped_img)
    
    return cropped_img_array

def read_text_from_image_array(image_array):
    """
    Read text from a NumPy array representing an image using easyocr.

    Parameters:
    - image_array: NumPy array representing the image.

    Returns:
    - Extracted text in English.
    """
    # Convert the NumPy array to a PIL Image
    image_pil = Image.fromarray(np.uint8(image_array))

    # Convert the PIL Image to a NumPy array
    image_np = np.array(image_pil)

    # Perform OCR on the image
    reader = easyocr.Reader(['en'])
    output = reader.readtext(image_np)

    # Extract text from the result
    extracted_text = output[0][1]

    return extracted_text


def main():
    st.title("Automatic Number Plate Recognition (ANPR) App")
    st.write("Upload an image and let the app detect the vehicle number plate.")

    uploaded_image = st.file_uploader("Choose an image ", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Convert the uploaded image to a PIL Image
        image = Image.open(uploaded_image)

        # Detect the number plate
        number_plate_img, coords = yolo_predictions(image, net)
        
        # Display the detected number plate image
        st.image(number_plate_img, caption="Detected Number Plate", use_column_width=True)

        for box in coords:
            x, y, width, height = box
            
        # Crop the image
        cropped_image = crop_image(np.array(number_plate_img), x, y, width, height)
        
        # Display the coordinates
        st.write("Detected Number Plate Coordinates:", coords)

        # Display the cropped image
        st.image(cropped_image, caption="Detected Number Plate", use_column_width=True)
        
        #display the extracted number from number plate
        cropped_image = np.array(cropped_image)  # Replace ... with your actual image array
        extracted_text = read_text_from_image_array(cropped_image)
        
        st.write("Extracted text: ")
        st.text(extracted_text)


if __name__ == "__main__":
    main()
