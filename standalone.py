import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

logging.info('This is a standalone notebook to solve a jigsaw puzzle')

logging.info('# Import dependencies')

import matplotlib.pyplot as plt
import ipyplot
import numpy as np
import cv2
import math
import statistics
from scipy.signal import find_peaks
from collections import Counter

logging.info('# Add utilities')

class Item():
    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

class LoopingList(list):
    def __getitem__(self, i):
        if isinstance(i, int):
            return super().__getitem__(i % len(self))
        else:
            return super().__getitem__(i)

logging.info('# Detect pieces')

img_rgb = cv2.imread("jigsawsqr.png")
h, w = img_rgb.shape[:2]
img_rgb = cv2.resize(img_rgb, (4*w, 4*h))

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

_, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
contours = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

regions = [Item(contour=contour, area=cv2.contourArea(contour), rect=cv2.boundingRect(contour)) for contour in contours]
median_area = statistics.median([region.area for region in regions])
min_area = 0.5 * median_area
max_area = 2 * median_area
logging.info(f"Ignore too big or small regions, median_area: {median_area}")

regions = [region for region in regions if 0.5 < region.area / median_area < 2]
logging.info(f"Number of remaining regions: {len(regions)}")

pieces = []
for region in regions:
    x, y, w, h = region.rect
    p = np.array([x, y])
    piece = Item(
        contour=region.contour - p,
        area=region.area,
        img_rgb=img_rgb[y:y+h, x:x+w],
        img_gray=img_gray[y:y+h, x:x+w],
        size=np.array([h, w])
    )
    pieces.append(piece)

logging.info('# Analyze pieces')

for piece in pieces:
    min_area_rect = cv2.minAreaRect(piece.contour)
    (cx, cy), (sx, sy), angle_degrees = min_area_rect
    if sy < sx:
        angle_degrees = (angle_degrees + 90) % 360
        sx, sy = sy, sx
    min_area_rect = ((cx, cy), (sx, sy), angle_degrees)
    piece.update(
        min_area_rect=min_area_rect,
        angle_degrees=angle_degrees
    )

for piece in pieces:
    h, w = piece.size
    img = cv2.linearPolar(piece.img_gray, (w/2, h/2), max(h, w), cv2.WARP_FILL_OUTLIERS)
    y0 = int(piece.angle_degrees * h / 360) % 360
    img = np.concatenate([img[y0:, :], img[:y0, :]])
    _, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    piece.update(img_polar=img_binary)

for piece in pieces:
    img_reversed = np.flip(piece.img_polar, axis=1)
    h, w = piece.size
    radius = w - np.argmax(img_reversed, axis=1)
    peak_indices = find_peaks(radius, prominence=5)[0]

    polar_peaks = [(radius[idx] * max(h, w) / w, idx * 360 / h) for idx in peak_indices]
    piece.update(polar_peaks=polar_peaks)   # list of [(peak_radius, peak_degrees)]

for piece in pieces:
    h, w = piece.size

    peaks = {}  # key=quarter, value=[(abs(angle), angle, distance)] where angles are relative to the piece_angle
    for peak_radius, peak_degrees in piece.polar_peaks:
        quarter = peak_degrees // 90
        diff_degrees = peak_degrees % 180
        if diff_degrees > 90:
            diff_degrees -= 180
        peaks.setdefault(quarter, []).append((abs(diff_degrees), peak_radius, peak_degrees + piece.angle_degrees))

    corners = LoopingList()
    for quarter in range(4):
        _, peak_radius, peak_degrees = min(peaks.get(3 - quarter, [(0, 0, 0)]))
        peak_radians = math.radians(peak_degrees)
        x = w/2 + peak_radius * math.cos(peak_radians)
        y = h/2 + peak_radius * math.sin(peak_radians)
        corner = Item(point=np.array([int(x), int(y)]))
        corners.append(corner)

    for idx, corner in enumerate(corners):
        corner.update(prev=corners[idx-1], next=corners[idx+1])
    piece.update(corners=corners)
    
def length(v):
    return v[0]**2 + v[1]**2

def sub_contour(c, idx0, idx1):
    if idx1 > idx0:
        return c[idx0:idx1]
    else:
        return np.concatenate([c[idx0:], c[:idx1]])

for piece in pieces:
    for corner in piece.corners:
        distances = [length(p[0] - corner.point) for p in piece.contour]
        corner.update(idx=distances.index(min(distances)))
    
    edges = LoopingList()
    for corner in piece.corners:
        corner0 = corner.prev
        corner1 = corner
        contour = sub_contour(piece.contour, corner0.idx, corner1.idx)
        edge = Item(
            contour=contour,
            corner0=corner0,
            corner1=corner1,
            p0=corner0.point,
            p1=corner1.point,
            idx0=corner0.idx,
            idx1=corner1.idx
        )
        edges.append(edge)
    
    piece.update(edges=edges)

    for edge in piece.edges:
        edge.corner0.update(edge1=edge)
        edge.corner1.update(edge0=edge)

for piece in pieces:
    nb_flats = 0
    for idx, edge in enumerate(piece.edges):
        dx, dy = edge.p1 - edge.p0
        angle_radians = math.atan2(dy, dx)
        sin = math.sin(angle_radians)
        cos = math.cos(angle_radians)
        matrix = np.array([[cos, -sin], [sin, cos]])
        normalized_contour = (edge.contour - edge.p0) @ matrix  # first point at (0, 0), last point at (X, 0)
        heights = normalized_contour[:,0,1]
        min_height = min(heights)
        max_height = max(heights)
        if abs(max_height) + abs(min_height) < 10:
            edge_type = 0
            nb_flats += 1
        elif abs(max_height) > abs(min_height):
            edge_type = 1
        else:
            edge_type = -1
        edge.update(
            idx=idx,
            arc_length=cv2.arcLength(edge.contour, closed=False),
            straight_length=math.sqrt(dx**2 + dy**2),
            height=max(abs(min_height), abs(max_height)),
            angle_degrees=math.degrees(angle_radians),
            type=edge_type,
            prev=piece.edges[idx-1],
            next=piece.edges[idx+1]
        )
        edge.update(
            normalized_prev_point=(edge.prev.p0 - edge.p0) @ matrix,
            normalized_next_point=(edge.next.p1 - edge.p1) @ matrix,
        )

    piece.update(nb_flats=nb_flats)

PAD = 50

for idx, piece in enumerate(pieces):
    h, w = piece.size
    for edge in piece.edges:
        matrix = cv2.getRotationMatrix2D((PAD + w/2, PAD + h/2), edge.angle_degrees, 1)
        contour = cv2.transform(edge.contour + PAD, matrix)
        p0 = contour[0][0]
        p1 = contour[-1][0]
        p_prev = p0 + edge.normalized_prev_point
        p_next = p1 + edge.normalized_next_point

logging.info('# Compute puzzle size')

nb_flats = Counter([piece.nb_flats for piece in pieces])

assert nb_flats[2] == 4
# H**2 - H*B/2 + I = 0
a = 1
b = - nb_flats[1] / 2
c = nb_flats[0]
delta = b**2 - 4*a*c
inner_height = int((-b - math.sqrt(delta)) / (2*a))
inner_width = int((-b + math.sqrt(delta)) / (2*a))
logging.info(f"Size of puzzle: {2 + inner_width}x{2 + inner_height}")
assert inner_height * inner_width == nb_flats[0]
assert 2 * (inner_height + inner_width) == nb_flats[1]

logging.info('# Compute the border')

before_flat_features = {}  # key=piece, value=(flat_edge, side_features)
after_flat_features = {}  # key=piece, value=(flat_edge, side_features)
for piece, edge in [(piece, edge) for piece in pieces for edge in piece.edges if edge.type == 0]:
    if edge.prev.type != 0:
        feature = np.array([
            edge.prev.straight_length,
            edge.prev.arc_length,
            edge.prev.height,
            edge.normalized_prev_point[0],
            edge.normalized_prev_point[1]
        ])
        before_flat_features[piece] = (edge, feature)
    if edge.next.type != 0:
        feature = np.array([
            edge.next.straight_length,
            edge.next.arc_length,
            edge.next.height,
            edge.normalized_next_point[0],
            edge.normalized_next_point[1]
        ])
        after_flat_features[piece] = (edge, feature)

male_after_flat_features = np.array([feature for piece, (edge, feature) in after_flat_features.items() if edge.next.type == 1])
female_after_flat_features = np.array([feature for piece, (edge, feature) in after_flat_features.items() if edge.next.type == -1])
male_before_flat_features = np.array([feature for piece, (edge, feature) in before_flat_features.items() if edge.prev.type == 1])
female_before_flat_features = np.array([feature for piece, (edge, feature) in before_flat_features.items() if edge.prev.type == -1])

def best_pieces_after_flat(piece):
    ref_edge, ref_features = after_flat_features[piece]
    results = [(sum((features - ref_features)**2), piece, edge) for piece, (edge, features) in before_flat_features.items() if edge.prev.type == -ref_edge.next.type]
    results.sort()
    return [(piece, edge) for score, piece, edge in results]

def best_pieces_before_flat(piece):
    ref_edge, ref_features = before_flat_features[piece]
    results = [(sum((features - ref_features)**2), piece, edge) for piece, (edge, features) in after_flat_features.items() if edge.next.type == -ref_edge.prev.type]
    results.sort()
    return [(piece, edge) for score, piece, edge in results]

def transform_idx(piece, contour_idx, transform):
    return cv2.transform(piece.contour[contour_idx:contour_idx+1], transform)[0][0]

def make_transform(piece, contour_idx, target_position, angle_degrees):
    """ Compute the affine transform of the piece that rotates by angle_degrees
    and set the point piece.contour[idx] at position"""
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle_degrees, 1)
    position = transform_idx(piece, contour_idx, rotation_matrix)
    dx, dy = target_position - position
    rotation_matrix33 = np.concatenate([rotation_matrix, [[0, 0, 1]]])
    translation_matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
    transform = translation_matrix @ rotation_matrix33
    return transform[:2]

solution = {}  # key=(i, j), value=(piece, top_edge_idx)
PAD = 30

def draw_solution():
    solution_rgb = np.zeros_like(img_rgb)
    for piece, top_edge_idx in solution.values():
        cv2.drawContours(solution_rgb, cv2.transform(piece.contour, piece.transform), -1, (255, 255, 255), 5)
    cv2.imwrite("solution.png", solution_rgb)

def place_top_left():
    piece = [piece for piece in pieces if piece.nb_flats == 2][2]
    edge = [edge for edge in piece.edges if edge.type == 0 and edge.prev.type != 0][0]
    corner = edge.corner1
    transform = make_transform(piece, corner.idx, np.array([PAD, PAD]), edge.angle_degrees + 180)
    piece.update(transform=transform)
    solution[(0, 0)] = (piece, edge.idx)

def place_border(pos, dpos, quarter):
    x, y = pos
    dx, dy = dpos
    for _ in range(11):
        piece, top_edge_idx = solution[(x, y)]
        edge = piece.edges[top_edge_idx - quarter]
        ref_point = transform_idx(piece, edge.corner0.idx, piece.transform)
        next_piece, next_edge = best_pieces_before_flat(piece)[0]
        next_transform = make_transform(next_piece, next_edge.idx1, ref_point, next_edge.angle_degrees - 90 * quarter + 180)
        next_piece.update(transform=next_transform)
        x += dx
        y += dy
        solution[(x, y)] = next_piece, next_edge.idx + quarter
        if next_piece.nb_flats == 2:
            break

place_top_left()
place_border((0, 0), (1, 0), 0)
place_border((11, 0), (0, 1), 1)
#place_border((11, 11), (-1, 0), 2)
#place_border((0, 11), (0, -1), 3)

logging.info("Write the image file")

draw_solution();

logging.info("Done")
