import streamlit as st
import cv2
import numpy as np
import os
import sys
from pathlib import Path
from PIL import Image
from PIL import ImageDraw
from matplotlib import pyplot as plt

st.title('Image Reader')

# Allow user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "bmp", "gif"])

# Display the uploaded image
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Process the image
    img = Image.open(uploaded_file)
    img_array = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        # Equalize the histogram
    dst = cv2.equalizeHist(img_array)

        # Thresholding
    _, output2 = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Gaussian blur
    gaussian = cv2.GaussianBlur(output2, (5, 5), 5)

        # Edge detection
    edges = cv2.Canny(gaussian, 70, 210, apertureSize=3)
    
    kernel = np.ones((1,3), np.uint8)
    erosion = cv2.erode(edges, kernel, iterations = 1)
    dilation = cv2.dilate(erosion, kernel, iterations = 1)
    #Find weft、、
    lines = cv2.HoughLinesP(dilation, 1.0, np.pi/180, 50, minLineLength=100, maxLineGap=100)
    i=0
    sum=0
    result=[]
    if lines is None:
        st.write("NO WEFT DETECT")
        sys.exit()
    
    for line in lines:
                x1,y1,x2,y2=line[0]
                # if x1==x2:
                #     x2=x2+1
                x1=float(x1)
                x2=float(x2)
                y1=float(y1)
                y2=float(y2)
            #統計直線斜率及算誤差
                
                k=-(y2-y1)/(x2-x1)
            #求正反切，再把弧度轉成角度

                result=np.arctan(k)*57.29577
                sum=sum+result
                i=i+1
    avg=sum/i
    st.write("角度:",avg)
    