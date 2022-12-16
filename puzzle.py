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
from functools import cached_property


## 180: left, 90: down, 0: right, 90: up
#def angle_vector(v):
#    x, y = v
#    angle = math.degrees(math.atan2(y, x)) % 360
#    return angle if 0 <= angle <= 180 else angle - 360
#
#
#def angle3p(p1, p2, p3):
#    angle1 = angle2p(p1, p2)
#    angle2 = angle2p(p2, p3)
#    angle = (angle2 - angle1) % 360
#    return angle if 0 <= angle <= 180 else angle - 360


def sigmoid(k, x, x0):
    return 1 / (1 + math.exp(-k * (x-x0)))

def bell(k, x, x0):
    return math.exp(-k * (x-x0)**2)

def diff_angle(angle, angle_ref):
    diff = (angle - angle_ref) % 360
    if diff > 180:
        diff -= 360
    return diff


class Piece():
    def __init__(self, idx, img_orig, img_gray, img_edges, contour):
        self.idx = idx
        x, y, w, h = cv2.boundingRect(contour)
        self.img_orig = img_orig[y:y+h, x:x+w]
        self.img_gray = img_gray[y:y+h, x:x+w]
        self.img_edges = img_edges[y:y+h, x:x+w]
        self.img_corners = cv2.cornerHarris(np.float32(self.img_gray), 2, 3, 0.04)
        self.contour = contour - [x, y]
        self.rotated_rect = cv2.minAreaRect(self.contour)
        self.rotated_box = np.int0(cv2.boxPoints(self.rotated_rect))
        (cx, cy), (sx, sy), angle = self.rotated_rect
        self.area = sx * sy
        self.lines = cv2.HoughLines(self.img_edges, 1, np.pi / 180, 10, None, 0, 0)
        if self.lines is None:
            self.lines = []

    def rotate(self):
        img_orig = cv2.copyMakeBorder(self.img_orig, 20, 20, 20, 20, cv2.BORDER_CONSTANT)
        (cx, cy), (sx, sy), angle = self.rotated_rect
        matrix = cv2.getRotationMatrix2D((20 + cx, 20 + cy), angle, 1.0)
        h, w, _ = img_orig.shape
        cv2.warpAffine(img_orig, matrix, (w, h), img_orig)
        if sx > sy:
            img_orig = cv2.rotate(img_orig, cv2.ROTATE_90_CLOCKWISE)
        img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
        img_edges = cv2.Canny(image=img_gray, threshold1=100, threshold2=200)
        (contours, _) = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return Piece(self.idx, img_orig, img_gray, img_edges, contours[0])

#        if len(self.lines) > 0:
#            rho, thetaRadians = self.lines[0][0]
#            theta = math.degrees(thetaRadians)
#            borderTop = pRho(rho/h, 0) * pTheta(theta, 90)
#            borderBottom = pRho(rho/h, 1) * pTheta(theta, 90)
#            borderLeft = pRho(rho/w, 0) * pTheta(theta, 0)
#            borderRight = pRho(rho/w, 1) * pTheta(theta, 0)
#            print(f"T {borderTop}, B {borderBottom}, L {borderLeft}, R {borderRight}")
            # if abs(theta) < math.radians(20) and rho < 20
            #print(rho/h, rho/w, math.degrees(theta))

# compute borders:
# crop image to detect minarearect => not robust, difficult to set dx & dy
# analyze the contour to detect lines => difficult
# get largest continuous countour points near hough lines

    def compute_borders(self):
        h, w, _ = self.img_orig.shape
        pTop= 0
        pBottom= 0
        pLeft= 0
        pRight= 0
        for line in self.lines[:2]:
            rho, thetaRadians = line[0]
            theta = math.degrees(thetaRadians)
            pVertical = bell(0.01, diff_angle((theta+90) % 180, 0), 90)
            pHorizontal = bell(0.01, diff_angle(theta % 180, 90), 0)
            pTop = max(pTop, pHorizontal * bell(100, rho/h, 0))
            pBottom = max(pBottom, pHorizontal * bell(100, rho/h, 1))
            pLeft = max(pLeft, pVertical * bell(100, abs(rho/w), 0))
            pRight = max(pRight, pVertical * bell(100, abs(rho/w), 1))
        threshold = 0.2
        borders = {
            'top': pTop > threshold,
            'bottom': pBottom > threshold,
            'left': pLeft > threshold,
            'right': pRight > threshold
        }
        self.borders = set([k for k, v in borders.items() if v])
        return self


#    THIS IS NOT ROBUST: many false positives AND false negatives
#    def compute_borders(self):
#        def filter_array(a, x0, y0, x1, y1):
#            return np.array([p for p in a if x0 <= p[0][0] <= x1 and y0 <= p[0][1] <= y1])
#
#        self.topBorder = None
#        self.bottomBorder = None
#        self.leftBorder = None
#        self.rightBorder = None
#
#        h, w, _ = self.img_orig.shape
#        dx = int(0.2 * w)
#        dy = int(0.1 * h)
#        (cx, cy), (sx, sy), angle = cv2.minAreaRect(filter_array(self.contour, dx, 0, w-dx, dy))
#        if sx < 2 or sy < 2:
#            self.topBorder = (cx, cy), angle
#        (cx, cy), (sx, sy), angle = cv2.minAreaRect(filter_array(self.contour, dx, h-dy, w-dx, h))
#        if sx < 2 or sy < 2:
#            self.bottomBorder = (cx, cy), angle
#        dx = int(0.1 * w)
#        dy = int(0.3 * h)
#        (cx, cy), (sx, sy), angle = cv2.minAreaRect(filter_array(self.contour, 0, dy, dx, h-dy))
#        if (sx > 0 or sy > 0) and (sx < 2 or sy < 2):
#            self.leftBorder = (cx, cy), angle
#        (cx, cy), (sx, sy), angle = cv2.minAreaRect(filter_array(self.contour, w-dx, dy, w, h-dy))
#        if (sx > 0 or sy > 0) and (sx < 2 or sy < 2):
#            self.rightBorder = (cx, cy), angle
#        self.hasBorder = self.topBorder or self.bottomBorder or self.leftBorder or self.rightBorder
#        return self

    def compute_corners(self):
        h, w, _ = self.img_orig.shape
        contour = np.concatenate((self.contour[-1:], self.contour, self.contour[:1]))
        #print(contour[0][0])
        pTopLeft = []  # list of (proba, (x, y))
        pTopRight = []  # list of (proba, (x, y))
        pBottomLeft = []  # list of (proba, (x, y))
        pBottomRight = []  # list of (proba, (x, y))
        for idx in range(1, len(contour) - 1):
            p1, p2, p3 = map(lambda p: p[0], contour[idx-1:idx+2])
            angle1 = angle_vector(p2 - p1)
            angle2 = angle_vector(p3 - p2)
            angle = (angle2 - angle1) % 360
            angle = angle if 0 <= angle <= 180 else angle - 360
            x, y = p2

            pTop = sigmoid(-6, y/h, 0.2)
            pBottom = sigmoid(6, y/h, 0.8)
            pLeft = sigmoid(-10, x/w, 0.1)
            pRight = sigmoid(10, x/w, 0.9)
            pCorner = p_angle(angle, -90)

            pTopLeft.append((pTop * pLeft * p_angle(angle1, 180) * p_angle(angle2, 90) * pCorner, (x, y)))
            pTopRight.append((pTop * pRight * p_angle(angle1, -90) * p_angle(angle2, 180) * pCorner, (x, y)))
            pBottomLeft.append((pBottom * pLeft * p_angle(angle1, 90) * p_angle(angle2, 0) * pCorner, (x, y)))
            pBottomRight.append((pBottom * pRight * p_angle(angle1, 0) * p_angle(angle2, -90) * pCorner, (x, y)))

        self.topLeft = max(pTopLeft)[1]
        self.topRight = max(pTopRight)[1]
        self.bottomLeft = max(pBottomLeft)[1]
        self.bottomRight = max(pBottomRight)[1]
        return self


class Solver():
    def __init__(self):
        self.pieces = []

    def read_pieces(self, filename):
        print("Detect contours...")
        img_orig = cv2.imread(filename)
        img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
        # img_blur = cv2.blur(img_gray, (3, 3))  # Produces some wrong contours
        img_edges = cv2.Canny(image=img_gray, threshold1=100, threshold2=200)
        (contours, _) = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print(f"Number of detected contours: {len(contours)}")
        pieces = [Piece(idx, img_orig, img_gray, img_edges, contour) for idx, contour in enumerate(contours)]

        print("Ignore very big and very small contours...")
        median_area = statistics.median([piece.area for piece in pieces])
        self.pieces = [piece for piece in pieces if 0.5 < piece.area / median_area < 2]
        print(f"Contour median area: {median_area}")
        print(f"Number of detected pieces: {len(pieces)}")

    def rotate_pieces(self):
        self.pieces = [piece.rotate() for piece in self.pieces]

    def analyze_pieces(self):
        print("Analyze pieces...")
        self.pieces = [piece.compute_borders() for piece in self.pieces]

        nb_corners = len([piece for piece in self.pieces if len(piece.borders) == 2])
        print(f"Number of detected corner pieces: {nb_corners}")
        perimeter = sum([len(piece.borders) for piece in self.pieces])
        print(f"Perimeter: {perimeter}")
        # assert len(corner_pieces) == 4
        # assert perimeter & nb total pieces
        # compute puzzle shape


class Display():
    def show(self, pieces):
        nb_rows = round(math.sqrt(len(pieces)))
        nb_cols = math.ceil(len(pieces) / nb_rows)
        fig = plt.figure(figsize=(7, 7), tight_layout=True)
        axis = None
        for idx, piece in enumerate(pieces):
            axis = fig.add_subplot(nb_rows, nb_cols, idx+1)
            plt.imshow(self.draw(piece))
            plt.axis('off')
        plt.show()

    def draw(self, piece):
        img = piece.img_orig
        for line in piece.lines[:2]:
            rho, theta = line[0]
            self.draw_polar_line(img, rho, theta)
        #cv2.drawMarker(img, piece.topLeft, (0, 255, 0), markerSize=2)
        #cv2.drawMarker(img, piece.topRight, (0, 255, 0), markerSize=2)
        #cv2.drawMarker(img, piece.bottomLeft, (0, 255, 0), markerSize=2)
        #cv2.drawMarker(img, piece.bottomRight, (0, 255, 0), markerSize=2)
        #for p in piece.contour:
        #    cv2.drawMarker(img, p[0], (0, 255, 0), markerSize=1)
        #cv2.drawContours(img, [cv2.approxPolyDP(piece.contour, 6, True)], 0, (0, 255, 0), 1)
        #cv2.drawContours(img, [piece.rotated_box], 0, (0, 255, 0), 1)
        #cv2.drawContours(img, [piece.contour], 0, (255, 0, 0), 1)

        #h, w, _ = img.shape
        #dx = int(0.2 * w)
        #dy = int(0.1 * h)
        #cv2.rectangle(img, (dx, 0), (w-dx, dy), (0, 255, 0), 1)
        self.draw_text(img, str(piece.idx))
        return img

    def draw_text(self, img, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        img_h, img_w, _ = img.shape
        size = 1
        weight = 2
        text_w, text_h = cv2.getTextSize(text, font, size, weight)[0]
        cv2.putText(img, text, ((img_w-text_w)//2, (img_h+text_h)//2), font, size, (255, 100, 0), weight)

    def draw_polar_line(self, img, rho, theta):
        cos = math.cos(theta)
        sin = math.sin(theta)
        x0 = cos * rho
        y0 = sin * rho
        p0 = (int(x0 - 100*sin), int(y0 + 100*cos))
        p1 = (int(x0 + 100*sin), int(y0 - 100*cos))
        cv2.line(img, p0, p1, (0, 255, 0), 2, cv2.LINE_AA)

solver = Solver()
solver.read_pieces('jigsawsqr.png')
solver.rotate_pieces()
solver.analyze_pieces()

#display = Display()
#display.show(pieces)
