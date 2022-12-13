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


#from math import atan2, degrees
#def angle_between(p1, p2, p3):
#    x1, y1 = p1
#    x2, y2 = p2
#    x3, y3 = p3
#    deg1 = degrees(atan2(x2 - x1, y2 - y1)) % 360
#    deg2 = degrees(atan2(x3 - x2, y3 - y2)) % 360
#    angle = (deg2 - deg1) % 360
#    return angle if 0 <= angle <= 180 else angle - 360


class Piece():
    def __init__(self, img_orig, img_gray, img_edges, contour):
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
        #self.lines = cv2.HoughLines(self.img_edges, 1, np.pi / 180, 30, None, 0, 0)
        #if self.lines is None:   
        #    self.lines = []

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
        return Piece(img_orig, img_gray, img_edges, contours[0])

    #def log_contour(self):
    #    contour = self.contour[-1] + self.contour + self.contour[0]
    #    print(contour[0][0])
    #    for idx in range(1, len(contour) - 2):
    #        p1, p2, p3 = map(lambda p: p[0], contour[idx-1:idx+2])
    #        angle = angle_between(p1, p2, p3)
    #        print(p2, angle)
    #    print(contour[-1][0])


class Solver():
    def read_pieces(self, filename):
        img_orig = cv2.imread(filename)
        img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
        # img_blur = cv2.blur(img_gray, (3, 3))  # Produces some wrong contours
        img_edges = cv2.Canny(image=img_gray, threshold1=100, threshold2=200)
        (contours, _) = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print(f"Number of detected contours: {len(contours)}")

        pieces = [Piece(img_orig, img_gray, img_edges, contour) for contour in contours]
        median_area = statistics.median([piece.area for piece in pieces])
        pieces = [piece for piece in pieces if 0.5 < piece.area / median_area < 2]
        print(f"Contour median area={median_area}, Ignore too small or too big contours")
        print(f"Number of detected pieces: {len(pieces)}")
        return pieces

    def rotate_pieces(self, pieces):
        return [piece.rotate() for piece in pieces]


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
        #cv2.drawContours(img, [piece.rotated_box], 0, (0, 255, 0), 1)
        #cv2.drawContours(img, [piece.contour], 0, (255, 0, 0), 1)
        for line in piece.lines[:10]:
            rho, theta = line[0]
            self.draw_polar_line(img, rho, theta)
        return img

    #def draw_polar_line(self, img, rho, theta):
    #    cos = math.cos(theta)
    #    sin = math.sin(theta)
    #    x0 = cos * rho
    #    y0 = sin * rho
    #    p0 = (int(x0 - 100*sin), int(y0 + 100*cos))
    #    p1 = (int(x0 + 100*sin), int(y0 - 100*cos))
    #    cv2.line(img, p0, p1, (0, 255, 0), 1, cv2.LINE_AA)


solver = Solver()
pieces = solver.read_pieces('jigsawsqr.png')
pieces = solver.rotate_pieces(pieces)

display = Display()
display.show(pieces[-16:])
