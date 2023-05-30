import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# split the image into 4 parts
def split_image(img):
    h, w, _ = img.shape
    return [img[:, :w//4, :], img[:, w//4:w//2, :], img[:, w//2:w*3//4, :], img[:, w*3//4:, :]]

# preprocess the image
def data_preprocess(img, denoise=True):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if denoise:
        img = cv2.medianBlur(img,3)
    img = img / 255.0
    img = 1 - img   # reverse the black and white, imshow figuring out why.
    
    img = cv2.resize(img, (28, 28))
    return img

def get_data(validcode_path):
    validcode_list = os.listdir(validcode_path)
    validcodes = []
    for validcode in validcode_list:
        if validcode.endswith('.bmp'):
            img = cv2.imread(os.path.join(validcode_path, validcode))
            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            validcodes.append(img)

    validcodes_split = []
    for validcode in validcodes:
        validcodes_split.append(split_image(validcode))
    
    return np.array(validcodes_split)
    