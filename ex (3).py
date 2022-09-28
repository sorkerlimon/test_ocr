from http import client
import os
import io
from google.cloud import vision
import cv2
import sys
from enum import Enum
import re
import numpy as np
from scipy import ndimage
import math
from typing import Tuple, Union
from deskew import determine_skew


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:/Users/21100002/Desktop/ocr/ServiceAccountToken.json'
client = vision.ImageAnnotatorClient()


class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5


def Detect_Text(image_file):
    def rotate(
            image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
    ) -> np.ndarray:
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
        height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

    image_0 = cv2.imread(image_file)
    grayscale = cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY)
    # resize = cv2.resize(grayscale, (1700, 2200), interpolation=cv2.INTER_LINEAR)
    resize = cv2.resize(grayscale, None, fx=0.8, fy=0.8)
    angle = determine_skew(resize)
    rotated = ndimage.rotate(resize, angle, reshape=True)
    rotated2 = ndimage.rotate(rotated, -0.5, reshape=True)
    success, encoded_image = cv2.imencode('.jpg', rotated2)
    cv2.imshow("", rotated2)
    cv2.waitKey(0)
    content = encoded_image.tobytes()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    document = response.full_text_annotation

    bounds = []
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        x = symbol.bounding_box.vertices[0].x
                        y = symbol.bounding_box.vertices[0].y
                        text = symbol.text
                        bounds.append([x, y, text, symbol.bounding_box])

    bounds.sort(key=lambda x: x[1])
    old_y = -1
    line = []
    lines = []
    threshold = 1
    for bound in bounds:
        x = bound[0]
        y = bound[1]
        if old_y == -1:
            old_y = y
        elif old_y - threshold <= y <= old_y + threshold:
            old_y = y
        else:
            old_y = -1
            line.sort(key=lambda x: x[0])
            lines.append(line)
            line = []
        line.append(bound)
    line.sort(key=lambda x: x[0])
    lines.append(line)
    return lines


lines = Detect_Text('306430300_888581982116824_2737954904624711935_n (1).jpg')
raw_text = ""
for line in lines:
    texts = [i[2] for i in line]
    texts = ''.join(texts)
    bounds = [i[3] for i in line]
    x = f"{texts} "
    # print(texts, end=' ')
    raw_text = raw_text + x
    raw_text = raw_text.upper()
print(raw_text)

# raw_text = raw_text.replace("TOTAL", "", raw_text.count("TOTAL"))
# raw_text = raw_text.replace("COUNT", "", raw_text.count("TOTAL"))
# raw_text = raw_text.replace(",", "", raw_text.count(","))

# text_list = raw_text.split(" ")
# # print(text_list)

# for i in text_list[:]:
#     if "RESULT" in i:
#         r_pos = text_list.index(i)
#         del text_list[:r_pos]
# hbg_found = False
# rbc_found = False
# wbc_found = False
# plt_found = False

# for i in text_list[:]:
#     if not wbc_found:
#         if "WHITEBLOODCELL" in i or "WBC" in i or "W.B.C" in i:
#             wbc_pos = text_list.index(i)
#             wbc_prob_val = text_list[wbc_pos:wbc_pos+3]
#             wbc_prob_val = " ".join(wbc_prob_val)
#             wbc_prob_val = re.findall('[-+]?\d*\.\d+|\d+', wbc_prob_val)
#             wbc_val = float(wbc_prob_val[0])
#             print("WBC =", wbc_val)
#             wbc_found = True

#             if wbc_val < 300:
#                 wbc_si_val = wbc_val
#                 print("WBC SI VALUE =", wbc_si_val, "x10^9/L")
#             else:
#                 wbc_si_val = wbc_val / 1000
#                 print("WBC SI VALUE =", wbc_si_val, "x10^9/L")

#     if not rbc_found:
#         if "REDBLOODCELL" in i or "RBC" in i or "R.B.C" in i:
#             rbc_pos = text_list.index(i)
#             rbc_prob_val = text_list[rbc_pos:rbc_pos+3]
#             rbc_prob_val = " ".join(rbc_prob_val)
#             rbc_prob_val = re.findall('[-+]?\d*\.\d+|\d+', rbc_prob_val)
#             rbc_val = float(rbc_prob_val[0])
#             print("RBC =", rbc_val)
#             rbc_found = True

#             if rbc_val < 50000:
#                 rbc_si_val = rbc_val
#                 print("RBC SI VALUE =", rbc_si_val, "x10^9/L")
#             else:
#                 rbc_si_val = rbc_val / 1000000
#                 print("RBC SI VALUE =", rbc_si_val, "x10^12/L")

#     if not plt_found:
#         if "PLATELET" in i:
#             plt_pos = text_list.index(i)
#             plt_prob_val = text_list[plt_pos:plt_pos+3]
#             plt_prob_val = " ".join(plt_prob_val)
#             plt_prob_val = re.findall('[-+]?\d*\.\d+|\d+', plt_prob_val)
#             plt_val = float(plt_prob_val[0])
#             print("PLT =", plt_val)
#             plt_found = True

#             if plt_val < 9999:
#                 plt_si_val = plt_val
#                 print("plt SI VALUE =", plt_si_val, "x10^9/L")
#             else:
#                 plt_si_val = plt_val / 1000
#                 print("plt SI VALUE =", plt_si_val, "x10^9/L")

#     if not hbg_found:
#         # if "MOGLOBIN" in i or "HGB" in i or "HB" in i:
#         if "MOGLOBIN" in i:
#             hgb_pos = text_list.index(i)
#             hgb_prob_val = text_list[hgb_pos:hgb_pos+3]
#             hgb_prob_val = " ".join(hgb_prob_val)
#             hgb_prob_val = re.findall('[-+]?\d*\.\d+|\d+', hgb_prob_val)
#             hgb_val = float(hgb_prob_val[0])
#             print("HGB =", hgb_val)
#             hbg_found = True

#             if hgb_val < 100:
#                 hgb_si_val = hgb_val * 10
#                 print("HGB SI VALUE =", hgb_si_val, "G/L")
#             else:
#                 hgb_si_val = hgb_val
#                 print("HGB SI VALUE =", hgb_si_val, "G/L")
