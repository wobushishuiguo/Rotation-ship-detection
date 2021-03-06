# Copyright 2017 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import cv2
import random
import math
import numpy as np
import imutils

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm


# =========================================================================== #
# Some colormaps.
# =========================================================================== #
def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors

colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=21)
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


# =========================================================================== #
# OpenCV drawing.
# =========================================================================== #
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Draw a collection of lines on an image.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_rectangle(img, p1, p2, color=[255, 0, 0], thickness=2):
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)


def draw_bbox(img, bbox, shape, label, color=[255, 0, 0], thickness=2):
    p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
    p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
    p1 = (p1[0]+15, p1[1])
    cv2.putText(img, str(label), p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)


def bboxes_draw_on_img(img, classes, scores, bboxes, colors, thickness=2):
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = colors[classes[i]]
        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text...
        s = '%s/%.3f' % (classes[i], scores[i])
        p1 = (p1[0]-5, p1[1])
        cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)


# =========================================================================== #
# Matplotlib show...
# =========================================================================== #
def plt_bboxes(results, imgdir, score_thr=0.05, figsize=(15,10), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    fig = plt.figure(figsize=figsize)
    img = mpimg.imread(imgdir)
    plt.imshow(img)
    colors = dict()
    for result in results :
        cls_id = 0
        score = result[0][4]
        if cls_id >= 0 and score > score_thr:
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            xmin = int(result[0][0])
            ymin = int(result[0][1])
            xmax = int(result[0][2])
            ymax = int(result[0][3])
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = str(cls_id)
            plt.gca().text(xmin, ymin - 2,
                           '{:s} | {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    plt.show()

def cv2_bboxes(results, imgdir, score_thr=0.05, figsize=(15,10), linewidth=1.5):
        """Visualize bounding boxes. Largely inspired by SSD-MXNET!
        """
        img = mpimg.imread(imgdir)
        colors = dict()
        for result in results[0]:
            cls_id = 0
            if result.size == 0:
                break
            score = result[4]
            if score > score_thr:
                xmin = int(result[0])
                ymin = int(result[1])
                xmax = int(result[2])
                ymax = int(result[3])
                r1 = result[5]
                r2 = result[6]
                yc = (ymin + ymax) / 2
                xc = (xmin + xmax) / 2
                h = ymax - ymin
                w = xmax - xmin
                l1 = w * r1
                l2 = h * r2
                wc = math.sqrt(l1 ** 2 + l2 ** 2)
                hc = math.sqrt((w - l1) ** 2 + (h - l2) ** 2)
                angle = math.atan(l1 / l2)
                if wc < hc:
                   W = hc
                   H = wc
                   angle = angle*180/3.1415926
                else :
                    W = wc
                    H = hc
                    angle = 90 + angle*180/3.1415926




                rec = ((xc, yc), (W, H), angle)
                box = cv2.cv.Boxpoints() if imutils.is_cv2()else cv2.boxPoints(rec)
                box = np.int0(box)
                cv2.drawContours(img, [box], -1, (255, 255, 0), 2)
                s = 'score:%.3f' % (score)
                p = (int(xmin - 20), int(ymin - 20))
                #cv2.putText(img, s, p, cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 1)
        plt.imshow(img)
        plt.show()

