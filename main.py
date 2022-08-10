from __future__ import annotations

from colorsys import hsv_to_rgb
from functools import total_ordering
from typing import *

import cv2 as cv
import numpy as np
import numpy.typing as npt
from skimage import io


@total_ordering
class PuzzlePiece:
    id: Optional[str]

    def __init__(self, contour: npt.NDArray[np.int64]):
        self.contour = contour
        self.minX, self.minY = contour.min(0)[0]
        self.maxX, self.maxY = contour.max(0)[0]
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


if __name__ == '__main__':
    img = io.imread('https://vouwbad.nl/jigsaw/jigsawsqr.png')[70:, 90:]
    ret, mask = cv.threshold(img[:, :, 1], 0, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
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

        hue = float(i) / len(pieces)
        color = tuple(round(v * 255) for v in hsv_to_rgb(hue, 1, 1))
        cv.drawContours(img, [piece.contour], 0, color, 1)
    print('first piece:', pieces[0], 'last piece:', pieces[-1])

    cv.namedWindow("image", cv.WINDOW_NORMAL)
    cv.imshow("image", img)
    while cv.getWindowProperty("image", cv.WND_PROP_VISIBLE):
        key = cv.waitKey(100)
        if key != -1 and key != 233:
            break
