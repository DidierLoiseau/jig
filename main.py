from __future__ import annotations

import argparse
import logging.config
import re
from typing import *

import cv2 as cv
import numpy as np
import yaml
from numpy import ndarray, dtype
from skimage import io

from puzzle import PuzzlePiece

logger = logging.getLogger(__name__)


def main(piece1: Optional[str, int], piece2: Optional[str, int]):
    # origimg = io.imread('https://vouwbad.nl/jigsaw/jigsawsqr.png')
    origimg: ndarray[dtype[int]] = io.imread('jigsawsqr.png')
    displayed = origimg
    img = origimg[65:, 80:]
    ret, mask = cv.threshold(img[:, :, 1], 0, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    pieces: list[PuzzlePiece] = list(PuzzlePiece(c) for c in contours if len(c) > 10)
    pieces.sort()
    logger.info('Piece count %i', len(pieces))

    row = 1
    col = ord('A')
    prev: Optional[PuzzlePiece] = None
    min_y, min_x, max_y, max_x = img.shape[0], img.shape[1], 0, 0
    for i, piece in enumerate(pieces):
        if prev is not None:
            if piece.is_same_row(prev):
                col += 1
            else:
                row += 1
                col = ord('A')
        piece.set_coordinates(chr(col), row)
        prev = piece

        cv.drawContours(img, [piece.contour], 0, (255, 0, 0), 1)
        cv.circle(img, np.rint(piece.center).astype(int), 1, (255, 0, 0), -1)
        if (piece1 is None
                or piece1[0] <= piece.col <= piece2[0]
                and piece1[1] <= piece.row <= piece2[1]):
            min_y, min_x = min(min_y, piece.minY), min(min_x, piece.minX)
            max_y, max_x = max(max_y, piece.maxY), max(max_x, piece.maxX)

            candidates, good_cands, best_cands = piece.find_corners()
            for c in candidates:
                cv.circle(img, piece.contour[c, 0], 0, (0, 255, 0), -1)
            for c in good_cands:
                cv.circle(img, piece.contour[c, 0], 0, (0, 196, 255), -1)
            if len(best_cands) != 4:
                logger.warning('Piece %s has %i corners!', piece.id, len(best_cands))
            for k, ext in enumerate(best_cands):
                cv.circle(img, piece.contour[ext, 0], 0, (0, 0, 255), -1)

    logger.info('Top left: %s Bottom right: %s', pieces[0].id, pieces[-1].id)
    if piece1:
        if piece1 == piece2:
            logger.info('Showing %s%s', *piece1)
        else:
            logger.info('Showing pieces %s%s to %s%s', *piece1, *piece2)
        displayed = img[max(0, min_y - 10):min(max_y + 10, img.shape[0]),
                    max(0, min_x - 10):min(max_x + 10, img.shape[1])]

    cv.namedWindow("image", cv.WINDOW_NORMAL)
    cv.imshow("image", displayed)
    while cv.getWindowProperty("image", cv.WND_PROP_VISIBLE):
        key = cv.waitKey(100)
        if key != -1 and key != 233:
            break


if __name__ == '__main__':
    with open('logging.yml', 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(config)


    def parse_coord(value: str, pattern=re.compile(r'^([A-Za-z]+)(\d+)$')) -> (str, int):
        match = pattern.match(value.upper())
        if match:
            return match.group(1), int(match.group(2))
        raise argparse.ArgumentTypeError(f"invalid value '{value}'")


    parser = argparse.ArgumentParser()
    parser.add_argument('piece1', nargs='?', type=parse_coord, help='The top-left piece to display', )
    parser.add_argument('piece2', nargs='?', type=parse_coord,
                        help='The bottom-right piece to display, defaults to piece1 if it was provided')
    args = parser.parse_args()

    main(args.piece1, args.piece2 or args.piece1)
