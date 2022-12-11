#!/usr/bin/env python3

# Usage:
# python3
# > import puzzle
# > importlib.reload(puzzle)
# > importlib.reload(puzzle)

import matplotlib.pyplot as plt
import numpy as np
import cv2
import importlib
import math
import statistics


from math import atan2, degrees

#def angle_between(p1, p2, p3):
#    x1, y1 = p1
#    x2, y2 = p2
#    x3, y3 = p3
#    deg1 = degrees(atan2(x2 - x1, y2 - y1)) % 360
#    deg2 = degrees(atan2(x3 - x2, y3 - y2)) % 360
#    angle = (deg2 - deg1) % 360
#    return angle if 0 <= angle <= 180 else angle - 360


class Display():
    def __init__(self):
        self.images = []

    def add(self, data, title):
        image = (data, title)
        self.images.append(image)

    def show(self):
        nb_rows = round(math.sqrt(len(self.images)))
        nb_cols = math.ceil(len(self.images) / nb_rows)
        fig = plt.figure(figsize=(7, 7), tight_layout=True)
        axis = None
        for idx, image in enumerate(self.images):
            data, title = image
            axis = fig.add_subplot(nb_rows, nb_cols, idx+1, sharex = axis, sharey = axis)
            plt.imshow(data)
            plt.axis('off')
            plt.title(title)
        plt.show()


class Shape():
    def __init__(self, img_orig, contour):
        self.contour = contour
        self.straight_rect = cv2.boundingRect(self.contour)
        (x, y, w, h) = self.straight_rect
        self.straight_img = img_orig[y:y+h, x:x+w]
        self.perimeter = cv2.arcLength(self.contour, True)
        self.rotated_rect = cv2.minAreaRect(self.contour)
        self.rotated_box = np.int0(cv2.boxPoints(self.rotated_rect))
        self.area = self.rotated_rect[1][0] * self.rotated_rect[1][1]
        cx, cy = self.rotated_rect[0]
        tx, ty, _ = img_orig.shape
        i = int(13 * cx // (tx-40))  # account for the border
        j = int(13 * cy // (ty-40))  # account for the border
        self.id = chr(ord('A') + i - 1) + str(j)

    def draw_text(self, img, text, dy = 0):
        cx, cy = self.rotated_rect[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 1.6 # 0.5
        sx, sy = cv2.getTextSize(text, font, size, 1)[0]
        cv2.putText(img, text, [int(cx - sx/2), int(cy + dy + sy/2)], font, size, (255, 0, 255), 3, cv2.LINE_AA)

    def draw_contour(self, img):
        cv2.drawContours(img, [self.contour], -1, (0, 255, 0), 2)

    def draw_contour_approx(self, img):
        epsilon = 0.02 * self.perimeter
        contour = cv2.approxPolyDP(self.contour, epsilon, True)
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

    def draw_contour_convex(self, img):
        contour = cv2.convexHull(self.contour)
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

    def draw_straight_rect(self, img):
        (x, y, w, h) = self.straight_rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.line(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.line(img, (x+w, y), (x, y+h), (0, 255, 0), 1)

    def draw_rotated_rect(self, img):
        cv2.drawContours(img, [self.rotated_box], 0, (0, 255, 0), 2)

    def draw_oriented(self, img):
        (x, y, w, h) = self.straight_rect
        (cx, cy), (sx, sy), angle = self.rotated_rect
        cx = int(cx)
        cy = int(cy)
        d = 40
        matrix = cv2.getRotationMatrix2D((d, d), angle, 1.0)
        cv2.warpAffine(img[cy-d:cy+d, cx-d:cx+d], matrix, (2*d, 2*d), img[cy-d:cy+d, cx-d:cx+d])
        cv2.rectangle(img, (int(cx-sx/2), int(cy-sy/2)), (int(cx+sx/2), int(cy+sy/2)), (0, 255, 0), 2)
        cv2.rectangle(img, (int(cx-sx/2), int(cy-0.3*sy)), (int(cx-0.3*sx), int(cy+0.3*sy)), (255, 0, 0), 1)

    def draw_polar_line(self, img, rho, theta):
        (x, y, w, h) = self.straight_rect
        cos = math.cos(theta)
        sin = math.sin(theta)
        x0 = x + cos * rho
        y0 = y + sin * rho
        p0 = (int(x0 - 10*sin), int(y0 + 10*cos))
        p1 = (int(x0 + 10*sin), int(y0 - 10*cos))
        cv2.line(img, p0, p1, (0, 255, 0), 1, cv2.LINE_AA)

    def draw_lines(self, img, img_edges):
        (x, y, w, h) = self.straight_rect
        img_edges = img_edges[y:y+h, x:x+w]
        lines = cv2.HoughLines(img_edges, 1, np.pi / 180, 30, None, 0, 0)
        if lines is not None:
            theta0 = None
            for line in lines[:2]:
                rho, theta = line[0]
                self.draw_polar_line(img, rho, theta)
                if theta0 is not None:
                    angle = degrees(theta - theta0) % 180
                    isSquare = 85 < abs(angle) < 95
                    if isSquare:
                        print(f"{self.id} is a square")
                    break
                theta0 = theta
                
        #lines = cv2.HoughLinesP(img_edges, 2, np.pi / 180, 15, None, 15, 2)
        #if lines is not None:
        #    for line in lines:
        #        x0, y0, x1, y1 = line[0]
        #        cv2.line(img, (x+x0, y+y0), (x+x1, y+y1), (0, 255, 0), 2, cv2.LINE_AA)

    #def log_contour(self):
    #    contour = self.contour[-1] + self.contour + self.contour[0]
    #    print(contour[0][0])
    #    for idx in range(1, len(contour) - 2):
    #        p1, p2, p3 = map(lambda p: p[0], contour[idx-1:idx+2])
    #        angle = angle_between(p1, p2, p3)
    #        print(p2, angle)
    #    print(contour[-1][0])


class Inventory():
    def __init__(self, img_orig, contours):
        self.img_orig = img_orig
        self.shapes = [Shape(img_orig, contour) for contour in contours]
        median_area = statistics.median([shape.area for shape in self.shapes])
        print(f"Number of detected shapes: {len(self.shapes)}")
        print(f"Ignore too small or too big shapes")
        low_area = 0.5 * median_area
        high_area = 2 * median_area
        self.shapes = list(filter(lambda shape: low_area < shape.area < high_area, self.shapes))
        print(f"Number of remaining shapes: {len(self.shapes)}")

    def draw_contours(self):
        img = self.img_orig.copy()
        for idx, shape in enumerate(self.shapes):
            shape.draw_contour(img)
        return img

    def draw_contours_approx(self):
        img = self.img_orig.copy()
        for idx, shape in enumerate(self.shapes):
            shape.draw_contour_approx(img)
        return img

    def draw_contours_convex(self):
        img = self.img_orig.copy()
        for idx, shape in enumerate(self.shapes):
            shape.draw_contour_convex(img)
        return img

    def draw_straight_rects(self):
        img = self.img_orig.copy()
        for idx, shape in enumerate(self.shapes):
            shape.draw_straight_rect(img)
        return img

    def draw_lines(self, img_edges):
        img = self.img_orig.copy()
        for shape in self.shapes:
            shape.draw_lines(img, img_edges)
        return img

    def draw_rotated_rects(self):
        img = self.img_orig.copy()
        for idx, shape in enumerate(self.shapes):
            shape.draw_rotated_rect(img)
        return img

    def draw_oriented(self):
        img = self.img_orig.copy()
        for idx, shape in enumerate(self.shapes):
            shape.draw_oriented(img)
            # shape.draw_text(img, str(shape.id))
        return img

    #def draw_corners(self):
    #    img = self.img_orig.copy()
    #    for shape in self.shapes[0:]:
    #        shape.log_contour()
    #        (x, y, w, h) = shape.rect
    #        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #        break
    #    return img

img_orig = cv2.imread('jigsawsqr.png')  # full image
border = 20
img_border = cv2.copyMakeBorder(img_orig, border, border, border, border, cv2.BORDER_CONSTANT)
img_gray = cv2.cvtColor(img_border, cv2.COLOR_BGR2GRAY)
# img_blur = cv2.blur(img_gray, (3, 3))  # Produces some wrong contours
img_edges = cv2.Canny(image=img_gray, threshold1=100, threshold2=200)

(contours, _) = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
inventory = Inventory(img_border, contours)
img_contours = inventory.draw_contours()
img_contours_approx = inventory.draw_contours_approx()
img_contours_convex = inventory.draw_contours_convex()
img_straight_rects = inventory.draw_straight_rects()
img_rotated_rects = inventory.draw_rotated_rects()
img_lines = inventory.draw_lines(img_edges)
#img_oriented = inventory.draw_oriented()
#img_corners = inventory.draw_corners()
#img_rotate = inventory.draw_rotate()
#img_corners = cv2.cornerHarris(np.float32(cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)), 2, 3, 0.04)
#img_corners = cv2.cornerHarris(np.float32(img_gray), 2, 3, 0.04)
#img_corners = cv2.dilate(img_corners, None)

d = Display()

#d.add(img_border, 'border')
#d.add(img_gray, 'gray')
#d.add(img_edges, 'edges')
#d.add(img_contours, 'contours')
#d.add(img_contours_approx, 'contours approx')
#d.add(img_contours_convex, 'contours convex')
#d.add(img_straight_rects, 'straight rects')
#d.add(img_rotated_rects, 'rotated rects')
d.add(img_lines, 'lines')
#d.add(img_oriented, 'oriented')
#d.add(img_rotate, 'rotate')
#d.add(img_corners, 'corners')
#d.add(img_sum, 'sum')
d.show()
