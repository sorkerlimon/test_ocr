from http import client
import os, io
from google.cloud import vision
import cv2
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
    with io.open(image_file,'rb') as image_file:
        x = image_file.read()
    img = np.asarray(bytearray(x), dtype="uint8")
    # image_0 = cv2.imread(image_file)
    image = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    # grayscale = cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY)
    # resize = cv2.resize(grayscale, (1700, 2200), interpolation=cv2.INTER_LINEAR)
    resize = cv2.resize(image, None, fx=0.8, fy=0.8)
    angle = determine_skew(resize)
    rotated = ndimage.rotate(resize, angle, reshape=True)
    rotated2 = ndimage.rotate(rotated, -0.5, reshape=True)
    success, encoded_image = cv2.imencode('.jpg', rotated2)
    # cv2.imshow("", rotated2)
    # cv2.waitKey(0)
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


# lines = Detect_Text('C:/Users/21100002/Desktop/all image/1.jpg')
# print(lines)
def raw_text(lines):
    raw_text = ""
    for line in lines:
        texts = [i[2] for i in line]
        texts = ''.join(texts)
        bounds = [i[3] for i in line]
        x = f"{texts} "
        # print(texts, end=' ')
        raw_text = raw_text + x
        raw_text = raw_text.lower()
    print(raw_text)



lines = Detect_Text('C:/Users/21100002/Desktop/all image/1.jpg')
raw = raw_text(lines)

print(raw)


# raw_text = raw_text.replace("total", "", raw_text.count("total"))
# raw_text = raw_text.replace("count", "", raw_text.count("count"))
# raw_text = raw_text.replace(",", "", raw_text.count(","))
# raw_text = raw_text.replace("µ", "u", raw_text.count("µ"))
# raw_text = raw_text.replace("μ", "u", raw_text.count("μ"))
# raw_text = raw_text.upper()
# text_list = raw_text.split(" ")
# # print(text_list)

# # Removing everything before "result"
# for i in text_list[:]:
#     if "RESULT" in i:
#         r_pos = text_list.index(i)
#         del text_list[:r_pos]
# # print(text_list)

# # initially all parameters are empty
# hbg_found = False
# rbc_found = False
# wbc_found = False
# plt_found = False
# neutrophil_found = False
# lymphocyte_found = False
# monocyte_found = False
# eosinophil_found = False
# basophil_found = False
# abs_neutrophil_found = False
# abs_lymphocyte_found = False
# abs_monocyte_found = False
# abs_eosinophil_found = False
# abs_basophil_found = False

# # Searching values for each parameter
# for i in text_list[:]:
#     # WBC START
#     if not wbc_found:
#         if "WHITEBLOODCELL" in i or "WBC" in i or "W.B.C" in i:
#             wbc_pos = text_list.index(i)
#             wbc_string_list = text_list[wbc_pos:wbc_pos + 3]
#             wbc_string = " ".join(wbc_string_list)
#             wbc_prob_val = re.findall('[-+]?\d*\.\d+|\d+', wbc_string)

#             if len(wbc_prob_val) > 0:
#                 wbc_val = float(wbc_prob_val[0])
#                 print("WBC          =", wbc_val)
#                 wbc_found = True
#                 wbc_prob_unit = re.sub("\s", "", wbc_string)

#                 if "K/UL" in wbc_prob_unit or "10^3/UL" in wbc_prob_unit or "/L" in wbc_prob_unit \
#                         or "10^3/MM^3" in wbc_prob_unit or "K/MM^3" in wbc_prob_unit:
#                     wbc_si_val = wbc_val
#                     print("WBC SI VALUE =", wbc_si_val, "x10^9/L")
#                 elif "CELLS/UL" in wbc_prob_unit or "CELLS/MM^3" in wbc_prob_unit \
#                         or "/CMM" in wbc_prob_unit or "CUMM" in wbc_prob_unit:
#                     wbc_si_val = wbc_val / 1000
#                     print("WBC SI VALUE =", wbc_si_val, "x10^9/L")
#                 else:
#                     if wbc_val < 300:
#                         wbc_si_val = wbc_val
#                         print("WBC SI VALUE =", wbc_si_val, "x10^9/L")
#                     else:
#                         wbc_si_val = wbc_val / 1000
#                         print("WBC SI VALUE =", wbc_si_val, "x10^9/L")
#     # WBC END

#     # RBC START
#     if not rbc_found:
#         if "REDBLOODCELL" in i or "RBC" in i or "R.B.C" in i:
#             rbc_pos = text_list.index(i)
#             rbc_string_list = text_list[rbc_pos:rbc_pos + 3]
#             rbc_string = " ".join(rbc_string_list)
#             rbc_prob_val = re.findall('[-+]?\d*\.\d+|\d+', rbc_string)

#             if len(rbc_prob_val) > 0:
#                 rbc_val = float(rbc_prob_val[0])
#                 print("RBC          =", rbc_val)
#                 rbc_found = True
#                 rbc_prob_unit = re.sub("\s", "", rbc_string)

#                 if "M/UL" in rbc_prob_unit or "10^6/UL" in rbc_prob_unit or "/L" in rbc_prob_unit \
#                         or "10^6/MM^3" in rbc_prob_unit or "M/MM^3" in rbc_prob_unit:
#                     rbc_si_val = rbc_val
#                     print("RBC SI VALUE =", rbc_si_val, "x10^12/L")
#                 elif "CELLS/UL" in rbc_prob_unit or "CELLS/MM^3" in rbc_prob_unit:
#                     rbc_si_val = rbc_val / 1000000
#                     print("RBC SI VALUE =", rbc_si_val, "x10^12/L")
#                 else:
#                     if rbc_val < 50000:
#                         rbc_si_val = rbc_val
#                         print("RBC SI VALUE =", rbc_si_val, "x10^9/L")
#                     else:
#                         rbc_si_val = rbc_val / 1000000
#                         print("RBC SI VALUE =", rbc_si_val, "x10^12/L")
#     # RBC END

#     # PLT START
#     if not plt_found:
#         if "PLATELET" in i or "PLT" in i:
#             plt_pos = text_list.index(i)
#             plt_string_list = text_list[plt_pos:plt_pos + 3]
#             plt_string = " ".join(plt_string_list)
#             plt_prob_val = re.findall('[-+]?\d*\.\d+|\d+', plt_string)
#             if len(plt_prob_val) > 0:
#                 plt_val = float(plt_prob_val[0])
#                 print("PLT          =", plt_val)
#                 plt_found = True
#                 plt_prob_unit = re.sub("\s", "", plt_string)

#                 if "K/UL" in plt_prob_unit or "10^3/UL" in plt_prob_unit or "/L" in plt_prob_unit \
#                         or "10^3/MM^3" in plt_prob_unit or "K/MM^3" in plt_prob_unit:
#                     plt_si_val = plt_val
#                     print("WBC SI VALUE =", plt_si_val, "x10^9/L")
#                 elif "CELLS/UL" in plt_prob_unit or "CELLS/MM^3" in plt_prob_unit \
#                         or "/CMM" in plt_prob_unit or "CUMM" in plt_prob_unit:
#                     plt_si_val = plt_val / 1000
#                     print("PLT SI VALUE =", plt_si_val, "x10^9/L")
#                 else:
#                     if plt_val < 9999:
#                         plt_si_val = plt_val
#                         print("PLT SI VALUE =", plt_si_val, "x10^9/L")
#                     else:
#                         plt_si_val = plt_val / 1000
#                         print("plt SI VALUE =", plt_si_val, "x10^9/L")
#     # PLT END

#     # HGB START
#     if not hbg_found:
#         # if "MOGLOBIN" in i or "HGB" in i or "HB" in i:
#         if "MOGLOBIN" in i:
#             hgb_pos = text_list.index(i)
#             hgb_string_list = text_list[hgb_pos:hgb_pos + 3]
#             hgb_string = " ".join(hgb_string_list)
#             hgb_prob_val = re.findall('[-+]?\d*\.\d+|\d+', hgb_string)
#             if len(hgb_prob_val) > 0:
#                 hgb_val = float(hgb_prob_val[0])
#                 print("HGB          =", hgb_val)
#                 hgb_found = True
#                 hgb_prob_unit = re.sub("\s", "", hgb_string)

#                 if "/L" in hgb_prob_unit or "/ML" in hgb_prob_unit:
#                     hgb_si_val = hgb_val
#                     print("HGB SI VALUE =", hgb_si_val, "G/L")
#                 elif "/DL" in hgb_prob_unit or "/100ML" in hgb_prob_unit or "g%" in hgb_prob_unit:
#                     hgb_si_val = hgb_val * 10
#                     print("HGB SI VALUE =", hgb_si_val, "G/L")
#                 else:
#                     if hgb_val < 100:
#                         hgb_si_val = hgb_val * 10
#                         print("HGB SI VALUE =", hgb_si_val, "G/L")
#                     else:
#                         hgb_si_val = hgb_val
#                         print("HGB SI VALUE =", hgb_si_val, "G/L")
#     # HGB END

#     # NEUTROPHIL START
#     if not neutrophil_found or not abs_neutrophil_found:
#         if "NEUTROPHIL" in i:
#             ntph_pos = text_list.index(i)
#             ntph_string_list = text_list[ntph_pos:ntph_pos + 3]
#             ntph_string = " ".join(ntph_string_list)

#             if "%" in ntph_string and not neutrophil_found:
#                 ntph_prob_val = re.findall('[-+]?\d*\.\d+|\d+', ntph_string)
#                 if len(ntph_prob_val) > 0:
#                     ntph_val = float(ntph_prob_val[0])
#                     print("NEUTROPHIL   =", ntph_val)
#                     neutrophil_found = True
#                     # wbc_prob_unit = re.sub("\s", "", wbc_string)
#                     ntph_si_val = ntph_val
#                     print("NEUTROPHIL SI Value =", ntph_si_val, "%")
#             else:
#                 if not abs_neutrophil_found:
#                     abs_ntph_prob_val = re.findall('[-+]?\d*\.\d+|\d+', ntph_string)
#                     if len(abs_ntph_prob_val) > 0:
#                         abs_ntph_val = float(abs_ntph_prob_val[0])
#                         print("ABS NEUTROPHIL   =", abs_ntph_val)
#                         abs_neutrophil_found = True
#                         abs_ntph_prob_unit = re.sub("\s", "", ntph_string)
#                         if "K/UL" in abs_ntph_prob_unit or "10^3/UL" in abs_ntph_prob_unit or "/L" in abs_ntph_prob_unit \
#                                 or "10^3/MM^3" in abs_ntph_prob_unit or "K/MM^3" in abs_ntph_prob_unit:
#                             abs_ntph_si_val = abs_ntph_val
#                             print("ABS NEUTROPHIL SI VALUE =", abs_ntph_si_val, "x10^9/L")
#                         elif "CELLS/UL" in abs_ntph_prob_unit or "CELLS/MM^3" in abs_ntph_prob_unit \
#                                 or "/CMM" in abs_ntph_prob_unit or "CUMM" in abs_ntph_prob_unit:
#                             abs_ntph_si_val = abs_ntph_val / 1000
#                             print("ABS NEUTROPHIL SI VALUE =", abs_ntph_si_val, "x10^9/L")
#                         else:
#                             if abs_ntph_val < 300:
#                                 abs_ntph_si_val = abs_ntph_val
#                                 print("ABS NEUTROPHIL SI VALUE =", abs_ntph_si_val, "x10^9/L")
#                             else:
#                                 abs_ntph_si_val = abs_ntph_val / 1000
#                                 print("ABS NEUTROPHIL SI VALUE =", abs_ntph_si_val, "x10^9/L")
#     # NEUTROPHIL END

#     # LYMPHOCYTE START
#     if not lymphocyte_found or not abs_lymphocyte_found:
#         if "LYMPHOCYTE" in i:
#             lmph_pos = text_list.index(i)
#             lmph_string_list = text_list[lmph_pos:lmph_pos + 3]
#             lmph_string = " ".join(lmph_string_list)

#             if "%" in lmph_string and not lymphocyte_found:
#                 lmph_prob_val = re.findall('[-+]?\d*\.\d+|\d+', lmph_string)
#                 if len(lmph_prob_val) > 0:
#                     lmph_val = float(lmph_prob_val[0])
#                     print("LYMPHOCYTE   =", lmph_val)
#                     lymphocyte_found = True
#                     lmph_si_val = lmph_val
#                     print("LYMPHOCYTE SI Value =", lmph_si_val, "%")
#             else:
#                 if not abs_lymphocyte_found:
#                     abs_lmph_prob_val = re.findall('[-+]?\d*\.\d+|\d+', lmph_string)
#                     if len(abs_lmph_prob_val) > 0:
#                         abs_lmph_val = float(abs_lmph_prob_val[0])
#                         print("ABS LYMPHOCYTE   =", abs_lmph_val)
#                         abs_lymphocyte_found = True
#                         abs_lmph_prob_unit = re.sub("\s", "", lmph_string)
#                         if "K/UL" in abs_lmph_prob_unit or "10^3/UL" in abs_lmph_prob_unit or "/L" in abs_lmph_prob_unit \
#                                 or "10^3/MM^3" in abs_lmph_prob_unit or "K/MM^3" in abs_lmph_prob_unit:
#                             abs_lmph_si_val = abs_lmph_val
#                             print("ABS lymphocyte SI VALUE =", abs_lmph_si_val, "x10^9/L")
#                         elif "CELLS/UL" in abs_lmph_prob_unit or "CELLS/MM^3" in abs_lmph_prob_unit \
#                                 or "/CMM" in abs_lmph_prob_unit or "CUMM" in abs_lmph_prob_unit:
#                             abs_lmph_si_val = abs_lmph_val / 1000
#                             print("ABS lymphocyte SI VALUE =", abs_lmph_si_val, "x10^9/L")
#                         else:
#                             if abs_lmph_val < 300:
#                                 abs_lmph_si_val = abs_lmph_val
#                                 print("ABS LYMPHOCYTE SI VALUE =", abs_lmph_si_val, "x10^9/L")
#                             else:
#                                 abs_lmph_si_val = abs_lmph_val / 1000
#                                 print("ABS LYMPHOCYTE SI VALUE =", abs_lmph_si_val, "x10^9/L")
#     # LYMPHOCYTE END

#     # MONOCYTE START
#     if not monocyte_found or not abs_monocyte_found:
#         if "MONOCYTE" in i:
#             mnct_pos = text_list.index(i)
#             mnct_string_list = text_list[mnct_pos:mnct_pos + 3]
#             mnct_string = " ".join(mnct_string_list)

#             if "%" in mnct_string and not monocyte_found:
#                 mnct_prob_val = re.findall('[-+]?\d*\.\d+|\d+', mnct_string)
#                 if len(mnct_prob_val) > 0:
#                     mnct_val = float(mnct_prob_val[0])
#                     print("MONOCYTE   =", mnct_val)
#                     monocyte_found = True
#                     mnct_si_val = mnct_val
#                     print("MONOCYTE SI Value =", mnct_si_val, "%")
#             else:
#                 if not abs_monocyte_found:
#                     abs_mnct_prob_val = re.findall('[-+]?\d*\.\d+|\d+', mnct_string)
#                     if len(abs_mnct_prob_val) > 0:
#                         abs_mnct_val = float(abs_mnct_prob_val[0])
#                         print("ABS MONOCYTE   =", abs_mnct_val)
#                         abs_monocyte_found = True
#                         abs_mnct_prob_unit = re.sub("\s", "", mnct_string)
#                         if "K/UL" in abs_mnct_prob_unit or "10^3/UL" in abs_mnct_prob_unit or "/L" in abs_mnct_prob_unit \
#                                 or "10^3/MM^3" in abs_mnct_prob_unit or "K/MM^3" in abs_mnct_prob_unit:
#                             abs_mnct_si_val = abs_mnct_val
#                             print("ABS MONOCYTE SI VALUE =", abs_mnct_si_val, "x10^9/L")
#                         elif "CELLS/UL" in abs_mnct_prob_unit or "CELLS/MM^3" in abs_mnct_prob_unit \
#                                 or "/CMM" in abs_mnct_prob_unit or "CUMM" in abs_mnct_prob_unit:
#                             abs_mnct_si_val = abs_mnct_val / 1000
#                             print("ABS MONOCYTE SI VALUE =", abs_mnct_si_val, "x10^9/L")
#                         else:
#                             if abs_mnct_val < 300:
#                                 abs_mnct_si_val = abs_mnct_val
#                                 print("ABS MONOCYTE SI VALUE =", abs_mnct_si_val, "x10^9/L")
#                             else:
#                                 abs_mnct_si_val = abs_mnct_val / 1000
#                                 print("ABS MONOCYTE SI VALUE =", abs_mnct_si_val, "x10^9/L")
#     # MONOCYTE END

#     # EOSINOPHIL START
#     if not eosinophil_found or not abs_eosinophil_found:
#         if "EOSINOPHIL" in i:
#             esnp_pos = text_list.index(i)
#             esnp_string_list = text_list[esnp_pos:esnp_pos + 3]
#             esnp_string = " ".join(esnp_string_list)

#             if "%" in esnp_string and not eosinophil_found:
#                 esnp_prob_val = re.findall('[-+]?\d*\.\d+|\d+', esnp_string)
#                 if len(esnp_prob_val) > 0:
#                     esnp_val = float(esnp_prob_val[0])
#                     print("EOSINOPHIL   =", esnp_val)
#                     eosinophil_found = True
#                     esnp_si_val = esnp_val
#                     print("EOSINOPHIL SI Value =", esnp_si_val, "%")
#             else:
#                 if not abs_eosinophil_found:
#                     abs_esnp_prob_val = re.findall('[-+]?\d*\.\d+|\d+', esnp_string)
#                     if len(abs_esnp_prob_val) > 0:
#                         abs_esnp_val = float(abs_esnp_prob_val[0])
#                         print("ABS EOSINOPHIL   =", abs_esnp_val)
#                         abs_eosinophil_found = True
#                         abs_esnp_prob_unit = re.sub("\s", "", esnp_string)
#                         if "K/UL" in abs_esnp_prob_unit or "10^3/UL" in abs_esnp_prob_unit or "/L" in abs_esnp_prob_unit \
#                                 or "10^3/MM^3" in abs_esnp_prob_unit or "K/MM^3" in abs_esnp_prob_unit:
#                             abs_esnp_si_val = abs_esnp_val
#                             print("ABS EOSINOPHIL SI VALUE =", abs_esnp_si_val, "x10^9/L")
#                         elif "CELLS/UL" in abs_esnp_prob_unit or "CELLS/MM^3" in abs_esnp_prob_unit \
#                                 or "/CMM" in abs_esnp_prob_unit or "CUMM" in abs_esnp_prob_unit:
#                             abs_esnp_si_val = abs_esnp_val / 1000
#                             print("ABS EOSINOPHIL SI VALUE =", abs_esnp_si_val, "x10^9/L")
#                         else:
#                             if abs_esnp_val < 300:
#                                 abs_esnp_si_val = abs_esnp_val
#                                 print("ABS EOSINOPHIL SI VALUE =", abs_esnp_si_val, "x10^9/L")
#                             else:
#                                 abs_esnp_si_val = abs_esnp_val / 1000
#                                 print("ABS EOSINOPHIL SI VALUE =", abs_esnp_si_val, "x10^9/L")
#     # EOSINOPHIL END

#     # BASOPHIL START
#     if not basophil_found or not abs_basophil_found:
#         if "BASOPHIL" in i:
#             bsph_pos = text_list.index(i)
#             bsph_string_list = text_list[bsph_pos:bsph_pos + 3]
#             bsph_string = " ".join(bsph_string_list)

#             if "%" in bsph_string and not basophil_found:
#                 bsph_prob_val = re.findall('[-+]?\d*\.\d+|\d+', bsph_string)
#                 if len(bsph_prob_val) > 0:
#                     bsph_val = float(bsph_prob_val[0])
#                     print("BASOPHIL   =", bsph_val)
#                     BASOPHIL_found = True
#                     bsph_si_val = bsph_val
#                     print("BASOPHIL SI Value =", bsph_si_val, "%")
#             else:
#                 if not abs_basophil_found:
#                     abs_bsph_prob_val = re.findall('[-+]?\d*\.\d+|\d+', bsph_string)
#                     if len(abs_bsph_prob_val) > 0:
#                         abs_bsph_val = float(abs_bsph_prob_val[0])
#                         print("ABS BASOPHIL   =", abs_bsph_val)
#                         abs_BASOPHIL_found = True
#                         abs_bsph_prob_unit = re.sub("\s", "", bsph_string)
#                         if "K/UL" in abs_bsph_prob_unit or "10^3/UL" in abs_bsph_prob_unit or "/L" in abs_bsph_prob_unit \
#                                 or "10^3/MM^3" in abs_bsph_prob_unit or "K/MM^3" in abs_bsph_prob_unit:
#                             abs_bsph_si_val = abs_bsph_val
#                             print("ABS BASOPHIL SI VALUE =", abs_bsph_si_val, "x10^9/L")
#                         elif "CELLS/UL" in abs_bsph_prob_unit or "CELLS/MM^3" in abs_bsph_prob_unit \
#                                 or "/CMM" in abs_bsph_prob_unit or "CUMM" in abs_bsph_prob_unit:
#                             abs_bsph_si_val = abs_bsph_val / 1000
#                             print("ABS BASOPHIL SI VALUE =", abs_bsph_si_val, "x10^9/L")
#                         else:
#                             if abs_bsph_val < 300:
#                                 abs_bsph_si_val = abs_bsph_val
#                                 print("ABS BASOPHIL SI VALUE =", abs_bsph_si_val, "x10^9/L")
#                             else:
#                                 abs_bsph_si_val = abs_bsph_val / 1000
#                                 print("ABS BASOPHIL SI VALUE =", abs_bsph_si_val, "x10^9/L")
#     # BASOPHIL END