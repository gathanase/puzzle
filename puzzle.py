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
from scipy.signal import find_peaks
from functools import cached_property


# 180: left, 90: down, 0: right, 90: up
def angle_vector(v):
    x, y = v
    angle = math.degrees(math.atan2(y, x)) % 360
    return angle if 0 <= angle <= 180 else angle - 360

def bell(k, x, x0):
    return math.exp(-k * (x-x0)**2)

def diff_angle(angle, angle_ref):
    diff = (angle - angle_ref) % 360
    if diff > 180:
        diff -= 360
    return diff

def distance(p0, p1):
    return (p0[0] - p1[0])**2 + (p0[1] - p1[1])**2


class Piece():
    def __init__(self, idx, img_orig, img_gray, img_edges, contour):
        self.idx = idx
        x, y, w, h = cv2.boundingRect(contour)
        self.img_orig = img_orig[y:y+h, x:x+w]
        self.img_gray = img_gray[y:y+h, x:x+w]
        self.img_edges = img_edges[y:y+h, x:x+w]
        self.img_corners = cv2.cornerHarris(np.float32(self.img_gray), 2, 3, 0.04)
        self.w = w
        self.h = h
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


    def compute_borders(self):
        # This is quite accurate but there are some false positives along the long edge
        # We could compute the longest streak to remove these false positives
        self.is_border_top = False
        self.is_border_bottom = False
        self.is_likely_border_left = False
        self.is_likely_border_right = False

        for line in self.lines[:2]:
            rho, thetaRadians = line[0]
            theta = math.degrees(thetaRadians)
            pVertical = bell(0.01, diff_angle((theta+90) % 180, 0), 90)
            pHorizontal = bell(0.01, diff_angle(theta % 180, 90), 0)

            if pHorizontal * bell(100, abs(rho/self.h), 0) > 0.2:
                self.is_border_top = True
            if pHorizontal * bell(100, abs(rho/self.h), 1) > 0.2:
                self.is_border_bottom = True
            if pVertical * bell(100, abs(rho/self.w), 0) > 0.2:
                self.is_likely_border_left = True
            if pVertical * bell(100, abs(rho/self.w), 1) > 0.2:
                self.is_likely_border_right = True


    def compute_corners(self):
        (cx, cy), (sx, sy), angle = self.rotated_rect
        # Fix corner detection for pieces with top or bottom borders
        if self.is_border_top:
            cy = cy - 0.1 * sy
        if self.is_border_bottom:
            cy = cy + 0.1 * sy

        # Compute the distance from cx, cy to each degree
        # This is a simple solution to get the max distance for each angle
        angle2dist = {}  # key=int angle degrees, value=distance from cx, cy
        old_theta = None
        old_distance = None
        for p in np.concatenate((self.contour, self.contour[:1]), axis=0):   # one more point for the linear interpolation
            x, y = p[0]
            distance = math.sqrt((x-cx)**2 + (y-cy)**2)
            theta = angle_vector((x-cx, y-cy)) % 360
            if old_theta is not None and theta != old_theta:
                # linear interpolation for intermediate angles
                for t in range(math.ceil(min(theta, old_theta)), math.floor(max(theta, old_theta) + 1)):
                    dt = old_distance + (t-old_theta)/(theta-old_theta) * (distance-old_distance)
                    angle2dist[t] = max(angle2dist.get(t, 0), dt)
            old_theta = theta
            old_distance = distance
        angle_dist = sorted(angle2dist.items())
        peaks = find_peaks([distance for theta, distance in angle_dist], prominence=2)

        peak_points = [] # (x, y)
        for idx in peaks[0]:
            theta, distance = angle_dist[idx]
            x = cx + distance * math.cos(math.radians(theta))
            y = cy + distance * math.sin(math.radians(theta))
            peak_points.append((int(x), int(y)))

        def distance_cy(p):
            return abs(p[1] - cy)

        self.corner_top_left = min([(x, y) for x, y in peak_points if x < sx/2 and y < sy/2], key=distance_cy)
        self.corner_top_right = min([(x, y) for x, y in peak_points if x > sx/2 and y < sy/2], key=distance_cy)
        self.corner_bottom_left = min([(x, y) for x, y in peak_points if x < sx/2 and y > sy/2], key=distance_cy)
        self.corner_bottom_right = min([(x, y) for x, y in peak_points if x > sx/2 and y > sy/2], key=distance_cy)


    def compute_edges(self):
        # find the contour point closest to each detected corners
        index_top_left = min([(distance(p[0], self.corner_top_left), idx) for idx, p in enumerate(self.contour)])[1]
        index_top_right = min([(distance(p[0], self.corner_top_right), idx) for idx, p in enumerate(self.contour)])[1]
        index_bottom_left = min([(distance(p[0], self.corner_bottom_left), idx) for idx, p in enumerate(self.contour)])[1]
        index_bottom_right = min([(distance(p[0], self.corner_bottom_right), idx) for idx, p in enumerate(self.contour)])[1]

        # we always have index_top_left < index_bottom_left < index_bottom_right < index_top_right
        self.points_left = self.contour[index_top_left:index_bottom_left]
        self.points_bottom = self.contour[index_bottom_left:index_bottom_right]
        self.points_right = self.contour[index_bottom_right:index_top_right]
        self.points_top = np.concatenate([self.contour[index_top_right:], self.contour[:index_top_left]])

    def analyze(self):
        self.compute_borders()
        self.compute_corners()
        self.compute_edges()


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
        for piece in self.pieces:
            piece.analyze()

        # nb_corners = len([piece for piece in self.pieces if len(piece.borders) == 2])
        # print(f"Number of detected corner pieces: {nb_corners}")
        # perimeter = sum([len(piece.borders) for piece in self.pieces])
        # print(f"Perimeter: {perimeter}")
        # assert len(corner_pieces) == 4
        # assert perimeter & nb total pieces
        # compute puzzle shape


class Display():
    def show(self, pieces):
        nb_tiles = len(pieces) * 1
        nb_cols = round(math.sqrt(nb_tiles))
        nb_cols = nb_cols + nb_cols % 2
        nb_rows = math.ceil(nb_tiles / nb_cols)
        fig = plt.figure(figsize=(7, 7), tight_layout=True)
        for idx, piece in enumerate(pieces):
            fig.add_subplot(nb_rows, nb_cols, 1*idx+1)
            self.draw(plt, piece)
            plt.axis('off')
        plt.show()

    def draw(self, plt, piece):
        img = piece.img_orig
        # (cx, cy), (sx, sy), angle = piece.rotated_rect
        # cv2.drawMarker(img, (int(cx), int(cy)), (255, 0, 0))
        # for line in piece.lines[:1]:
        #     rho, thetaRadians = line[0]
        #     self.draw_polar_line(img, rho, thetaRadians)
        #cv2.drawContours(img, [piece.rotated_box], 0, (0, 255, 0), 1)
        #cv2.drawContours(img, [piece.contour], 0, (255, 0, 0), 1)
        #cv2.line(img, piece.corner_top_left, piece.corner_bottom_right, (0, 255, 0), 1, cv2.LINE_AA)
        #cv2.line(img, piece.corner_top_right, piece.corner_bottom_left, (0, 255, 0), 1, cv2.LINE_AA)
        for p in piece.points_top:
            cv2.drawMarker(img, p[0], (255, 0, 0), markerSize=1)
        for p in piece.points_bottom:
            cv2.drawMarker(img, p[0], (0, 255, 0), markerSize=1)
        for p in piece.points_left:
            cv2.drawMarker(img, p[0], (255, 255, 0), markerSize=1)
        for p in piece.points_right:
            cv2.drawMarker(img, p[0], (0, 0, 255), markerSize=1)
        self.draw_text(img, str(piece.idx))
        plt.imshow(img)

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
        cv2.line(img, p0, p1, (0, 255, 0), 1, cv2.LINE_AA)


solver = Solver()
solver.read_pieces('jigsawsqr.png')
solver.rotate_pieces()
solver.analyze_pieces()

pieces = [piece for piece in solver.pieces if piece.is_border_top or piece.is_border_bottom]

display = Display()
display.show(pieces)
