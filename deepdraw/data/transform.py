import numpy as np
import PIL


class Sequencer(object):
    """Converts a drawing to sequence of state vectors.

    (dx, dy, p1, p2, p3)
    p1: Pen is down
    p2: Pen is up (next line won't be drawn)
    p3: End of drawing

    See https://arxiv.org/abs/1704.03477
    """

    def __init__(self):
        pass

    def __call__(self, drawing):
        """
        Args:
            drawing (dict): drawing to be sequenced
        Returns:
            list of tuples: (dx, dy, p1, p2, p3)
        """
        drawing = drawing['drawing']
        strokes = []
        for stroke in drawing:
            # stroke = [(x0, x1, ...), (y0, y1, ...)]
            # -> [(x0, y0, 1, 0, 0), (x1, y1, 1, 0, 0), ...]
            stroke = [*zip(*stroke)]
            stroke = [(x, y, 1, 0, 0) for x, y in stroke]
            # Mark last point as end of stroke
            stroke[-1] = stroke[-1][:2] + (0, 1, 0)
            strokes += stroke

        # Duplicate first point
        strokes = strokes[:1] + strokes
        # Pair adjacent points together: [(p0, p1), (p1, p2), ...]
        strokes = [*zip(strokes[:-1], strokes[1:])]

        res = []
        for a, b in strokes:
            (x0, y0), (x1, y1) = a[:2], b[:2]
            dx, dy = x1-x0, y1-y0

            res.append((dx, dy) + b[2:])

        # Add end-of-drawing marker
        res.append((0, 0, 0, 0, 1))
        return res


class Rasterizer(object):
    """Rasterize drawing using Bresenham's line algorithm.

    Args:
        center (bool): move drawing to center resulting image
    """

    def __init__(self, center=True):
        # TODO: resize image?
        self.center = center

    def __call__(self, drawing):
        """
        Args:
            drawing (dict): drawing to be rasterized
        Returns:
            PIL.Image: a 256x256 grayscale image
        """

        drawing = drawing['drawing']
        if self.center:
            x_max = max(x for X, _ in drawing for x in X)
            y_max = max(y for _, Y in drawing for y in Y)

        image = np.zeros((256, 256), dtype=np.bool)
        for stroke in drawing:
            # stroke = [(x0, x1, ...), (y0, y1, ...)]
            # -> [(x0, y0), (x1, y1), ...]
            stroke = [*zip(*stroke)]
            # -> [((x0, y0), (x1, y1)),  ((x1, y1), (x2, y2)), ...]
            stroke = zip(stroke[:-1], stroke[1:])

            for (x0, y0), (x1, y1) in stroke:
                if self.center:
                    dx = 256 - x_max
                    dy = 256 - y_max
                    x0 += dx // 2
                    x1 += dx // 2
                    y0 += dy // 2
                    y1 += dy // 2

                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                sx = -1 if x0 > x1 else 1
                sy = -1 if y0 > y1 else 1

                x, y = x0, y0
                if dx > dy:
                    err = dx / 2
                    while x != x1:
                        image[x, y] = True
                        err -= dy
                        if err < 0:
                            y += sy
                            err += dx
                        x += sx
                else:
                    err = dy / 2
                    while y != y1:
                        image[x, y] = True
                        err -= dx
                        if err < 0:
                            x += sx
                            err += dy
                        y += sy

                image[x, y] = True

        image = image.T  # Transpose to switch x and y axes
        image = image.astype(np.uint8) * 255
        image = PIL.Image.fromarray(image)
        return image
