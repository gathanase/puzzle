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


    def compute_borders(self):
        borders = []  # (direction, probability, rho, thetaRadians)]
        for line in self.lines[:2]:
            rho, thetaRadians = line[0]
            theta = math.degrees(thetaRadians)
            pVertical = bell(0.01, diff_angle((theta+90) % 180, 0), 90)
            pHorizontal = bell(0.01, diff_angle(theta % 180, 90), 0)

            borders.append(('top', pHorizontal * bell(100, abs(rho/self.h), 0), rho, thetaRadians))
            borders.append(('bottom', pHorizontal * bell(100, abs(rho/self.h), 1), rho, thetaRadians))
            borders.append(('left', pVertical * bell(100, abs(rho/self.w), 0), rho, thetaRadians))
            borders.append(('right', pVertical * bell(100, abs(rho/self.w), 1), rho, thetaRadians))

        borders = [border for border in borders if border[1] > 0.2] 
        if self.idx == 25:
            print(borders)

        #for border in borders:
        #    ranges = []  # ((x0, y0), (x1, y1))
        #    line_start = None
        #    for idx in range(len(self.contour)):
        #        p = self.contour[idx][0]
        #        if p

        self.borders = borders
        return self


    def compute_corners(self):
        points = np.zeros((len(self.contour), 2))
        (cx, cy), (sx, sy), angle = self.rotated_rect
        for idx, p in enumerate(self.contour):
            x, y = p[0]
            d = math.sqrt((x-cx)**2 + (y-cy)**2)
            points[idx] = (angle_vector((x-cx, y-cy)), d)
        self.corners = [points[0], points[100]]
        return self


    # def compute_corners(self):
    #     def p_angle(k, angle, angle_ref):
    #         diff = diff_angle(angle, angle_ref)
    #         return {-45: k, 0: 1, 45: k}.get(diff, 0)

    #     h, w, _ = self.img_orig.shape
    #     contour = np.concatenate((self.contour[-2:], self.contour, self.contour[:2]))
    #     pTopLeft = []  # list of (proba, (x, y))
    #     pTopRight = []  # list of (proba, (x, y))
    #     pBottomLeft = []  # list of (proba, (x, y))
    #     pBottomRight = []  # list of (proba, (x, y))
    #     for idx in range(2, len(contour) - 2):
    #         p1, p2, p3 = map(lambda p: p[0], contour[idx-1:idx+2])
    #         angle1 = angle_vector(p2 - p1)
    #         angle2 = angle_vector(p3 - p2)
    #         angle = (angle2 - angle1) % 360
    #         angle = angle if 0 <= angle <= 180 else angle - 360
    #         x, y = p2

    #         pTop = sigmoid(-10, y/h, 0.21)
    #         pBottom = sigmoid(10, y/h, 0.79)
    #         pLeft = sigmoid(-10, x/w, 0.06)
    #         pRight = sigmoid(10, x/w, 0.94)
    #         pCorner = p_angle(0.6, angle, -90)

    #         pTopLeft.append((pTop * pLeft * p_angle(0.6, angle1, 180) * pCorner, (x, y)))
    #         pTopRight.append((pTop * pRight * p_angle(0.6, angle1, -90) * pCorner, (x, y)))
    #         pBottomLeft.append((pBottom * pLeft * p_angle(0.6, angle1, 90) * pCorner, (x, y)))
    #         pBottomRight.append((pBottom * pRight * p_angle(0.6, angle1, 0) * pCorner, (x, y)))

    #     self.topLeft = max(pTopLeft)[1]
    #     self.topRight = max(pTopRight)[1]
    #     self.bottomLeft = max(pBottomLeft)[1]
    #     self.bottomRight = max(pBottomRight)[1]
    #     return self


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
        self.pieces = [piece.compute_corners() for piece in self.pieces]

        nb_corners = len([piece for piece in self.pieces if len(piece.borders) == 2])
        print(f"Number of detected corner pieces: {nb_corners}")
        perimeter = sum([len(piece.borders) for piece in self.pieces])
        print(f"Perimeter: {perimeter}")
        # assert len(corner_pieces) == 4
        # assert perimeter & nb total pieces
        # compute puzzle shape


class Display():
    def show(self, pieces):
        nb_tiles = len(pieces) * 2
        nb_cols = round(math.sqrt(nb_tiles))
        nb_cols = nb_cols + nb_cols % 2
        nb_rows = math.ceil(nb_tiles / nb_cols)
        fig = plt.figure(figsize=(7, 7), tight_layout=True)
        for idx, piece in enumerate(pieces):
            fig.add_subplot(nb_rows, nb_cols, 2*idx+1)
            self.draw(plt, piece)
            plt.axis('off')
            fig.add_subplot(nb_rows, nb_cols, 2*idx+2)
            self.plot(plt, piece)
            plt.axis('off')
        plt.show()

    def plot(self, plt, piece):
        (cx, cy), (sx, sy), angle = piece.rotated_rect
        angle2dist = {}  # key=int angle degrees, value=distance from cx, cy
        old_theta = None
        old_distance = None
        for p in np.concatenate((piece.contour, piece.contour[:1]), axis=0):
            x, y = p[0]
            distance = math.sqrt((x-cx)**2 + (y-cy)**2)
            theta = int(angle_vector((x-cx, y-cy)) % 360)
            if old_theta is not None:
                for t in range(math.ceil(min(theta, old_theta)), math.floor(max(theta, old_theta))):
                    dt = old_distance + (t-old_theta)/(theta-old_theta) * (distance-old_distance)
                    angle2dist[t] = max(angle2dist.get(t, 0), dt)
            old_theta = theta
            old_distance = distance
        points = list(angle2dist.items())
        points.sort()
        
        plt.plot([x for x, y in points], [y for x, y in points])
        peaks = find_peaks([y for x, y in points])
        plt.plot(peaks[0], [points[x][1] for x in peaks[0]], 'rx')
        # plt.plot([[x, points[x]] for x in peaks[0]], 'rx')
        # plt.axvline(-90, 0, 100, c='r')
        # plt.axvline(90, 0, 100, c='r')

    def draw(self, plt, piece):
        img = piece.img_orig
        # cv2.drawMarker(img, piece.contour[0][0], (0, 255, 0))
        (cx, cy), (sx, sy), angle = piece.rotated_rect
        cv2.drawMarker(img, (int(cx), int(cy)), (255, 0, 0))
        #for point in piece.corners:
        #    x, y = point
        #    print(x, y, cx, cy)
        #    cv2.drawMarker(img, (int(x), int(y)), (0, 255, 0))
        # for _, _, rho, theta in piece.borders:
        #     self.draw_polar_line(img, rho, theta)
        # cv2.drawMarker(img, piece.topLeft, (0, 255, 0), markerSize=1)
        # cv2.drawMarker(img, piece.topRight, (0, 255, 0), markerSize=1)
        # cv2.drawMarker(img, piece.bottomLeft, (0, 255, 0), markerSize=1)
        # cv2.drawMarker(img, piece.bottomRight, (0, 255, 0), markerSize=1)
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

pieces = solver.pieces[0:1]

# pieces_id = dict([(piece.idx, piece) for piece in pieces])
# pieces_id[22].topLeft = (0, 13)
# print(dict([(piece.idx, (piece.topRight, piece.bottomRight, piece.topLeft, piece.bottomLeft)) for piece in pieces]))
# print(sum([piece.topRight[0]/piece.w for piece in pieces])/len(pieces))
# print(sum([piece.topRight[1]/piece.h for piece in pieces])/len(pieces))
# print(sum([piece.topLeft[0]/piece.w for piece in pieces])/len(pieces))
# print(sum([piece.topLeft[1]/piece.h for piece in pieces])/len(pieces))
# print(sum([piece.bottomRight[0]/piece.w for piece in pieces])/len(pieces))
# print(sum([piece.bottomRight[1]/piece.h for piece in pieces])/len(pieces))
# print(sum([piece.bottomLeft[0]/piece.w for piece in pieces])/len(pieces))
# print(sum([piece.bottomLeft[1]/piece.h for piece in pieces])/len(pieces))

display = Display()
display.show(pieces)
