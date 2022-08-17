from __future__ import annotations

from functools import total_ordering
from typing import *
from typing import Any

import cv2 as cv
import numpy as np
import numpy.typing as npt
from numpy import ndarray, dtype
from scipy.signal import argrelextrema
from skimage import io


def angle_between_arrays(p1s: npt.NDArray[Any], p2s: npt.NDArray[Any], p3s: npt.NDArray[Any]) -> npt.NDArray[np.float]:
    """
    Compute element-wise 3-point angles in anti-clockwise direction, with values between -pi and pi

    :param p1s: array of first points
    :param p2s: array of the middle points at the center of the angles
    :param p3s: array of last points
    :return: array of corresponding angles in rad
    """
    v21s = p1s - p2s
    v23s = p3s - p2s
    dot = np.empty(len(v21s))
    # seems to be the best solution based on https://stackoverflow.com/a/41443497/525036
    for i in range(len(v21s)):
        dot[i] = np.dot(v21s[i], v23s[i])
    det = np.cross(v21s, v23s)

    return np.arctan2(det, dot)


@total_ordering
class PuzzlePiece:
    center: npt.NDArray[np.float]
    id: Optional[str]

    def __init__(self, contour: npt.NDArray[np.int64]):
        self.contour = contour
        self.minX, self.minY = topLeft = contour.min(0, initial=None)[0]
        self.maxX, self.maxY = bottomRight = contour.max(0, initial=None)[0]
        self.center = (topLeft + bottomRight) / 2
        self.id = None

    def __eq__(self, other):
        return self.is_same_row(other) and self.is_same_col(other)

    def __lt__(self, other: PuzzlePiece):
        if self.minY > other.maxY:
            return False
        if self.maxY < other.minY:
            return True
        # both are on the same row
        return self.maxX < other.minX

    def is_same_row(self, other: PuzzlePiece) -> bool:
        return self.minY < other.maxY and self.maxY > other.minY

    def is_same_col(self, other: PuzzlePiece) -> bool:
        return self.minX < other.maxX and self.maxX > other.minX

    def set_coordinates(self, row: int, col: str):
        self.id = col + str(row)

    def __repr__(self):
        return "PuzzlePiece(%r)" % self.id

    def find_corners(self) -> (ndarray[Any, dtype[int]], ndarray[dtype[int]], ndarray[Any, dtype[int]]):
        # compute distance between center and contour to find the locally furthest points
        centered = self.contour[:, 0, :] - self.center
        cont_len = len(self.contour)
        norms = np.linalg.norm(centered, axis=1)
        candidates: npt.NDArray[int] = argrelextrema(norms, np.greater_equal, mode='wrap')[0]
        # enlarge lookup around candidates for potentially more acute angles
        # FIXME arbitrary range expansion should depend on pieces size
        lookaround = np.repeat(np.arange(-3, 3)[:, np.newaxis], len(candidates), axis=1)
        # remove duplicates – this also brings us back to a 1d array
        candidates = np.unique(np.mod(lookaround + candidates, cont_len))
        # split int sequences of continuous candidates...
        split_idx = np.flatnonzero(np.diff(candidates) != 1) + 1
        roll = candidates[-1] == cont_len - 1 and candidates[0] == 0
        if roll:
            # our candidates wrap around the contour, join end with start
            roll_shift = len(candidates) - split_idx[-1]
            candidates = np.roll(candidates, roll_shift)
            split_idx += roll_shift
        # ... but avoid too large ones
        # FIXME arbitrary LARGE_GAP should depend on pieces size
        LARGE_GAP = 10
        large_gaps = np.diff(split_idx) > LARGE_GAP
        if np.any(large_gaps):
            gaps_pos = np.flatnonzero(large_gaps)
            split_idx = np.insert(split_idx, gaps_pos + 1, (split_idx[gaps_pos] + split_idx[gaps_pos + 1]) / 2)
        if split_idx[0] > LARGE_GAP:
            split_idx = np.insert(split_idx, 0, split_idx[0] / 2)
        if roll:
            split_idx = split_idx[:-1]

        cont_cand = np.split(candidates, split_idx)
        shift = lambda cdts, s: self.contour[np.mod(cdts + s, cont_len), 0]
        # /!\ contours are counter-clockwise… with an inverted y-axis!
        # Thus take the points in original order to get inside angles at our potential corners
        # fixme arbitrary neighbour selection should depend on pieces size
        NEIB4ANGLES = 5
        angles = angle_between_arrays(
            shift(candidates, -NEIB4ANGLES),
            self.contour[candidates, 0],
            shift(candidates, NEIB4ANGLES))
        angles[angles < 0] += 2 * np.pi
        cont_angles = np.split(angles, split_idx)
        acutest = list(np.argmin(ca) for ca in cont_angles)
        good_cands = best_cands = np.fromiter(
            (c[a] for ca, a, c in zip(cont_angles, acutest, cont_cand) if np.pi / 4 < ca[a] < 3 * np.pi / 4), int)
        # check if candidate corners all face the center
        # check if center does not see previous neighbour left of candidate
        good_prev = angle_between_arrays(self.contour[best_cands, 0], self.center, shift(best_cands, -NEIB4ANGLES)) > 0
        if not np.all(good_prev):
            best_cands = best_cands[good_prev]
        # check if center does not see next neighbour right of candidate
        good_next = angle_between_arrays(shift(best_cands, NEIB4ANGLES), self.center, self.contour[best_cands, 0]) > 0
        if not np.all(good_next):
            best_cands = best_cands[good_next]
        return candidates, good_cands, best_cands


def main():
    # origimg = io.imread('https://vouwbad.nl/jigsaw/jigsawsqr.png')
    origimg = io.imread('jigsawsqr.png')
    displayed = origimg
    img = origimg[65:, 80:]
    ret, mask = cv.threshold(img[:, :, 1], 0, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    pieces: list[PuzzlePiece] = list(PuzzlePiece(c) for c in contours if len(c) > 10)
    pieces.sort()
    print('pieces:', len(pieces))

    row = 1
    col = ord('A')
    prev: Optional[PuzzlePiece] = None
    for i, piece in enumerate(pieces):
        if prev is not None:
            if piece.is_same_row(prev):
                col += 1
            else:
                row += 1
                col = ord('A')
        piece.set_coordinates(row, chr(col))
        prev = piece

        cv.drawContours(img, [piece.contour], 0, (255, 0, 0), 1)
        cv.circle(img, np.rint(piece.center).astype(int), 1, (255, 0, 0), -1)

        candidates, good_cands, best_cands = piece.find_corners()
        for c in candidates:
            cv.circle(img, piece.contour[c, 0], 0, (0, 255, 0), -1)
        for c in good_cands:
            cv.circle(img, piece.contour[c, 0], 0, (0, 196, 255), -1)
        print(piece.id, 'best_cands:', len(best_cands), 'nice!' if len(best_cands) == 4 else 'fixme…')
        for k, ext in enumerate(best_cands):
            cv.circle(img, piece.contour[ext, 0], 0, (0, 0, 255), -1)

    print('first piece:', pieces[0], 'last piece:', pieces[-1])

    cv.namedWindow("image", cv.WINDOW_NORMAL)
    cv.imshow("image", displayed)
    while cv.getWindowProperty("image", cv.WND_PROP_VISIBLE):
        key = cv.waitKey(100)
        if key != -1 and key != 233:
            break


if __name__ == '__main__':
    main()
