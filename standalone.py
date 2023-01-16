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
from functools import cache
from scipy.signal import savgol_filter
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
print(f"Ignore too big or small regions, median_area: {median_area}")

regions = [region for region in regions if 0.5 < region.area / median_area < 2]
print(f"Number of remaining regions: {len(regions)}")

pieces = []
for region in regions:
    x, y, w, h = region.rect
    p = np.array([x, y])
    col = int((x - w/2) * 13 / 4540)
    row = int(1 + (y - h/2) * 13 / 4450)
    name = chr(ord('A') + col) + str(row)
    piece = Item(
        contour=region.contour - p,
        area=region.area,
        img_rgb=img_rgb[y:y+h, x:x+w],
        img_gray=img_gray[y:y+h, x:x+w],
        size=np.array([h, w]),
        name=name
    )
    pieces.append(piece)

logging.info('# Analyze pieces')

logging.info('## Detect corners')

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

logging.info('## Analyze edges')

def length(v):
    return v[0]**2 + v[1]**2

def sub_contour(c, idx0, idx1):
    if idx1 > idx0:
        return c[idx0:idx1]
    else:
        return np.concatenate([c[idx0:], c[:idx1]])

for piece in pieces:
    for corner in piece.corners:
        results = [(length(p[0] - corner.point), idx, p[0]) for idx, p in enumerate(piece.contour)]
        distance, idx, point = min(results)
        corner.update(
            idx=idx,
            point=point
        )

    edges = LoopingList()
    for corner in piece.corners:
        corner0 = corner.prev
        corner1 = corner
        contour = sub_contour(piece.contour, corner0.idx, corner1.idx+1)
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
            normalized_contour=normalized_contour
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

logging.info('## Enhance contours')

NB_SAMPLES = 20

for piece in pieces:
    for edge in piece.edges:
        # smooth the curve
        smoothed = np.array(edge.normalized_contour)
        smoothed[:, 0, 0] = savgol_filter(smoothed[:, 0, 0], 21, 1)
        smoothed[:, 0, 1] = savgol_filter(smoothed[:, 0, 1], 21, 1)

        # compute the distance from the first point, this is not exactly edge.arc_length
        deltas = smoothed[1:] - smoothed[:-1]
        distances = np.cumsum(np.sqrt(np.sum(deltas ** 2, axis=2)))
        max_distance = distances[-1]
        # get N equidistant points
        sampled = []
        for i in range(NB_SAMPLES):
            distance = i * max_distance / (NB_SAMPLES - 1)
            idx = np.argmax(distances >= distance)
            sampled.append(smoothed[idx])
        sampled = np.array(sampled)
        edge.update(
            smoothed_contour=smoothed,
            sampled_contour=sampled
        )

logging.info('## Add piece utilities')

piece_by_name = dict([(piece.name, piece) for piece in pieces])

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

def first_flat_edge(piece):
    return [edge for edge in piece.edges if edge.type == 0 and edge.prev.type != 0][0]

def last_flat_edge(piece):
    return [edge for edge in piece.edges if edge.type == 0 and edge.next.type != 0][0]

def edge_after_flat(piece):
    return last_flat_edge(piece).next

def edge_before_flat(piece):
    return first_flat_edge(piece).prev

logging.info('# Compute puzzle size')

def compute_size(area, perimeter):
    # perimeter = 2 * (H+W)
    # area = H*W
    # H**2 - perimeter/2 * H + area = 0
    a = 1
    b = -perimeter/2
    c = area
    delta = b**2 - 4*a*c
    h = int((-b - math.sqrt(delta)) / (2*a))
    w = int((-b + math.sqrt(delta)) / (2*a))
    return (min(h, w), max(h, w))

solution = Item()

nb_flats = Counter([piece.nb_flats for piece in pieces])
assert nb_flats[2] == 4
area = len(pieces)
perimeter = nb_flats[1] + 2*nb_flats[2]
w, h = compute_size(area, perimeter)
print(f"Size of puzzle grid: {w} x {h}")
assert w * h == area
assert 2 * (w + h) == perimeter

solution.update(grid_size = (w, h))

area = sum([piece.area for piece in pieces])
perimeter = sum([edge.straight_length+1 for piece in pieces for edge in piece.edges if edge.type == 0])

w, h = compute_size(area, perimeter)
print(f"Approximate size of puzzle image: {w} x {h}")

solution.update(approximate_size = (w, h))

logging.info('# Add piece matchers')

logging.info('## Add border utilities')

border_pieces = [piece for piece in pieces if piece.nb_flats > 0]

logging.info('## Distance border match')

for piece in pieces:
    for edge in piece.edges:
        angle_degrees = edge.angle_degrees - edge.prev.angle_degrees
        matrix = cv2.getRotationMatrix2D((0, 0), angle_degrees, 1)
        samples = cv2.transform(edge.sampled_contour, matrix)
        edge.update(sampled_contour_after_edge=samples)

        angle_degrees = edge.angle_degrees - edge.next.angle_degrees
        matrix = cv2.getRotationMatrix2D((0, 0), angle_degrees, 1)
        samples = cv2.transform(edge.sampled_contour, matrix)
        edge.update(sampled_contour_before_edge=samples)

@cache
def distance_matches_after_flat(piece0):
    edge0 = edge_after_flat(piece0)
    contour0 = edge0.sampled_contour_after_edge[::-1]

    results = []
    for piece1 in pieces:
        if piece1.nb_flats > 0:
            edge1 = edge_before_flat(piece1)
            if edge1.type == -edge0.type:
                contour1 = edge1.sampled_contour_before_edge
                diff = contour0 - contour1
                offset = np.mean(diff, axis=0)
                score = np.sum((diff - offset)**2)
                results.append((score, piece1))
    results.sort()
    return [(score, piece) for score, piece in results]

logging.info('# Assert the border')

a1 = piece_by_name['A1']
ordered_border = [a1]
used_pieces = set()  # do not put A1, it will be used for the last piece
while True:
    next_piece = distance_matches_after_flat(ordered_border[-1])[0][1]
    if next_piece in used_pieces:
        break
    ordered_border.append(next_piece)
    used_pieces.add(next_piece)

print("Computed border pieces:", ' '.join([piece.name for piece in ordered_border]))
assert used_pieces == set(border_pieces)
assert ordered_border[-1] == ordered_border[0]  # loop on the A1 piece
ordered_border = ordered_border[:-1]  # remove the repeated A1 piece
h, w = solution.grid_size
if ordered_border[h-1].nb_flats == 1:
    h, w = w, h
    solution.grid_size = w, h
assert [ordered_border[i].nb_flats for i in [0, h-1, h+w-2, 2*h+w-3]] == [2, 2, 2, 2];

logging.info('# Place the border')

PAD = 30
size = max(solution.approximate_size) + 2*PAD

solution.update(
    img_rgb=np.zeros((size, size, 3), img_rgb.dtype),
    grid={} # key=(i, j), value=(piece, top_edge_idx)
)

def place_piece(pos, piece, top_edge_idx):
    x, y = pos
    c = int((x+y)%2 * 255)
    solution.grid[pos] = (piece, top_edge_idx)
    contour = cv2.transform(piece.contour, piece.transform)
    cv2.drawContours(solution.img_rgb, contour, -1, (c, 255 - c, 255), 5)

def place_top_left():
    piece = piece_by_name['A1']
    top_edge = first_flat_edge(piece)
    corner = top_edge.corner1
    transform = make_transform(piece, corner.idx, np.array([PAD, PAD]), top_edge.angle_degrees + 180)
    piece.update(transform=transform)
    place_piece((0, 0), piece, top_edge.idx)

def place_border(pos, dpos, quarter):
    pos = np.array(pos)
    dpos = np.array(dpos)
    for _ in range(11):
        piece, top_edge_idx = solution.grid[tuple(pos)]
        pos += dpos
        if tuple(pos) in solution.grid:
            break
        flat_edge = piece.edges[top_edge_idx - quarter]
        ref_point = transform_idx(piece, flat_edge.corner1.idx, piece.transform)
        score, piece = distance_matches_after_flat(piece)[0]
        flat_edge = first_flat_edge(piece)
        transform = make_transform(piece, flat_edge.idx0, ref_point, flat_edge.angle_degrees + 180 - 90 * quarter)
        piece.update(transform=transform)
        place_piece(tuple(pos), piece, flat_edge.idx + quarter)

place_top_left()
place_border((0, 0), (0, 1), 3)
place_border((0, 11), (1, 0), 2)
place_border((11, 11), (0, -1), 1)
place_border((11, 0), (-1, 0), 0)

logging.info("Write the image file")

cv2.imwrite("solution.png", solution.img_rgb)

logging.info("Done")
