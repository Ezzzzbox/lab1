"""Заготовки задач на NumPy."""

import numpy as np
import matplotlib.pyplot as plt
import unittest

from grader_contracts.numpy_tasks import (
    BinarizeInput,
    ChessInput,
    EllipseInput,
    MatrixInput,
    MatrixStatistics,
    MatrixVectorBatchInput,
    OneHotInput,
    RandomMatrixInput,
    RectangleInput,
    TimeSeriesInput,
    TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    if len(matrices) != len(vectors) or len(matrices) == 0:
        raise ValueError
    sum = np.zeros(len(vectors[0]))
    for i in range(len(matrices)):
        sum += matrices[i] @ vectors[i]
    return sum


class test_sum_prod(unittest.TestCase):
    def test_simple(self):
        mtrx = np.ones([2, 2])
        vec = np.ones(2)
        data = MatrixVectorBatchInput([mtrx], [vec])
        res = sum_prod(data)
        self.assertTrue(np.array_equal(res, np.array([2.0, 2.0])))

    def test_multiple_simple(self):
        mtrxs = []
        vecs = []
        for _ in range(5):
            mtrxs.append(np.ones([2, 2]))
            vecs.append(np.ones(2))
        res = compare(mtrxs, vecs, np.array([10.0, 10.0]))
        self.assertTrue(res)

    def test_two(self):
        mtrxs = [np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])]
        vecs = [np.array([5, 5]), np.array([5, 5])]
        res = compare(mtrxs, vecs, np.array([30, 70]))
        self.assertTrue(res)


# служебная функция для тестов
def compare(mtrxs, vecs, actual):
    data = MatrixVectorBatchInput(mtrxs, vecs)
    res = sum_prod(data)
    return np.array_equal(res, actual)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    if len(matrix.shape) == 1:
        return binarize_vector(matrix, threshold)
    rows = matrix.shape[0]
    columns = matrix.shape[1]
    for i in range(rows):
        for j in range(columns):
            if matrix[i, j] > threshold:
                matrix[i, j] = 1
            else:
                matrix[i, j] = 0
    return matrix


def binarize_vector(vector, threshold) -> np.ndarray:
    for i in range(len(vector)):
        if vector[i] > threshold:
            vector[i] = 1
        else:
            vector[i] = 0
    return vector


class test_binarize(unittest.TestCase):
    def compare(self, mtrx, tr, actual):
        data = BinarizeInput(mtrx, tr)
        res = binarize(data)
        return np.array_equal(actual, res)

    def test_vector(self):
        mtrx = np.array([1, 2, 3, 4, 5, 6])
        actual = np.array([0, 0, 0, 1, 1, 1])
        self.assertTrue(self.compare(mtrx, 3, actual))

    def test_matrix(self):
        mtrx = np.array([[1, 2, 3], [4, 5, 6]])
        actual = np.array([[0, 0, 0], [0, 1, 1]])
        self.assertTrue(self.compare(mtrx, 4.5, actual))

    def test_square_matrix(self):
        mtrx = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        actual = np.array([[0, 0, 0], [0, 1, 1], [1, 1, 1]])
        self.assertTrue(self.compare(mtrx, 4.5, actual))


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    if len(matrix.shape) == 1:
        return unique_rows_vector(matrix)
    unique_rows = []
    for i in range(matrix.shape[0]):
        unique_rows.append(unique_list(matrix[i]))
    return unique_rows


def unique_list(list) -> list[float]:
    dig_repeat = dict()
    for i in range(len(list)):
        if dig_repeat.get(list[i]) is None:
            dig_repeat[list[i]] = 1
        else:
            dig_repeat[list[i]] += 1
    res = []
    for num, repeat in dig_repeat.items():
        if repeat == 1:
            res.append(num)
    return res


def unique_rows_vector(vector):
    res = []
    for i in range(len(vector)):
        res.append([vector[i]])
    return res


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    if len(matrix.shape) == 1:
        return [unique_list(matrix)]
    res = []
    for j in range(matrix.shape[1]):
        res.append(unique_list(matrix[:, j]))
    return res


class test_unique_rows(unittest.TestCase):
    def run_test(self, mtrx, expect):
        data = MatrixInput(mtrx)
        return self.compare_lists(unique_rows(data), expect)

    def compare_lists(self, mtrx, expect):
        if len(mtrx) != len(expect):
            return False
        for i in range(len(mtrx)):
            if len(mtrx[i]) != len(expect[i]):
                return False
            set_mtrx = set(mtrx[i])
            for num in expect[i]:
                set_mtrx.add(num)
            if len(set_mtrx) > len(mtrx[i]):
                return False
        return True

    def test_vector(self):
        vec = np.array([1, 2, 3, 3, 4, 4])
        expect = [[1], [2], [3], [3], [4], [4]]
        self.assertTrue(self.run_test(vec, expect))

    def test_4x4(self):
        mtrx = np.array([[1, 2, 3, 4], [1, 1, 2, 2], [-5, -5, 5, 1], [0, 1, 3, 0]])
        expect = [[1, 2, 3, 4], [], [5, 1], [1, 3]]
        self.assertTrue(self.run_test(mtrx, expect))

    def test_4x2(self):
        mtrx = np.array([[1, 2, 3, 4], [1, 1, 0, 2]])
        expect = [[1, 2, 3, 4], [0, 2]]
        self.assertTrue(self.run_test(mtrx, expect))

    def test_2x4(self):
        mtrx = np.array([[1, 2], [1.3, 1.3], [0, 0], [-1, -1]])
        expect = [[1.0, 2.0], [], [], []]
        self.assertTrue(self.run_test(mtrx, expect))


class test_unique_columns(unittest.TestCase):
    def run_test(self, mtrx, expect):
        data = MatrixInput(mtrx)
        return self.compare_lists(unique_columns(data), expect)

    def compare_lists(self, mtrx, expect):
        if len(mtrx) != len(expect):
            return False
        for i in range(len(mtrx)):
            if len(mtrx[i]) != len(expect[i]):
                return False
            set_mtrx = set(mtrx[i])
            for num in expect[i]:
                set_mtrx.add(num)
            if len(set_mtrx) > len(mtrx[i]):
                return False
        return True

    def test_vector(self):
        vec = np.array([1, 2, 3, 3, 4, 4])
        expect = [[1, 2]]
        self.assertTrue(self.run_test(vec, expect))

    def test_4x4(self):
        mtrx = np.array([[1, 2, 3, 4], [1, 1, 2, 2], [-5, -5, 5, 1], [0, 1, 3, 0]])
        expect = [[-5, 0], [2, -5], [2, 5], [4, 2, 1, 0]]
        self.assertTrue(self.run_test(mtrx, expect))

    def test_4x2(self):
        mtrx = np.array([[1, 2, 3, 4], [1, 1, 0, 2]])
        expect = [[], [2, 1], [3, 0], [4, 2]]
        self.assertTrue(self.run_test(mtrx, expect))

    def test_2x4(self):
        mtrx = np.array([[1, 2], [0, 1.3], [0, 1.3], [-1, -1]])
        expect = [[1, -1], [2, -1]]
        self.assertTrue(self.run_test(mtrx, expect))


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = (
        data.rows,
        data.columns,
        data.mean,
        data.std,
        data.seed,
    )
    np.random.seed(seed)
    mtrx = np.random.normal(mean, std, (rows, columns))
    rows_avg = [sum(row) / len(row) for row in mtrx]
    columns_avg = [sum(clm) / len(clm) for clm in mtrx[:,]]
    rows_dis = []
    columns_dis = []
    for i in range(len(mtrx)):
        rows_dis.append(count_dispersion(mtrx[i], rows_avg[i]))
    for j in range(mtrx.shape[1]):
        columns_dis.append(count_dispersion(mtrx[:, j], columns_avg[j]))
    return MatrixStatistics(mtrx, rows_avg, columns_avg, rows_dis, columns_dis)


def count_dispersion(ls, avg):
    mean_square = sum([i**2 for i in ls]) / len(ls)
    avg_square = avg * avg
    return mean_square - avg_square


def draw_histogram(matrix):
    plt.hist(matrix)
    plt.show()


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    chessboard = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            if (i + j) % 2 == 0:
                chessboard[i, j] = first
            else:
                chessboard[i, j] = second
    return chessboard


class test_chess(unittest.TestCase):
    def run_test(self, rows, clms, first, second):
        chessboard = chess(ChessInput(rows, clms, first, second))
        if len(chessboard) != rows:
            return False
        if chessboard.shape[1] != clms:
            return False
        for i in range(rows):
            for j in range(clms):
                if (i + j) % 2 == 0 and chessboard[i, j] == second:
                    return False
                elif (i + j) % 2 == 1 and chessboard[i, j] == first:
                    return False
        return True

    def test_0x0(self):
        self.assertTrue(self.run_test(0, 0, 5, 8))

    def test_1x1(self):
        self.assertTrue(self.run_test(1, 1, 5, 8))

    def test_1x2(self):
        self.assertTrue(self.run_test(1, 2, 5, 8))

    def test_2x2(self):
        self.assertTrue(self.run_test(2, 2, 5, 8))

    def test_3x2(self):
        self.assertTrue(self.run_test(3, 2, 5, 8))

    def test_7x7(self):
        self.assertTrue(self.run_test(1, 2, 5, 8))


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    img = make_rgb_img(image_height, image_width, background_color)
    start = [image_height // 2 - height // 2, image_width // 2 - width // 2]
    for i in range(width):
        img[start[0], start[1] + i] = shape_color
        img[start[0] + height - 1, start[1] + i] = shape_color
    for j in range(height):
        img[start[0] + j, start[1]] = shape_color
        img[start[0] + j, start[1] + width - 1] = shape_color
    return img


def make_rgb_img(height, width, background_color):
    img = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            img[i, j] = background_color
    return img


"""
class test_draw_rectangle(unittest.TestCase):
    def test_visual(self):
        data = RectangleInput(30, 30, 100, 100, (255, 0, 0), (120, 120, 120))
        img = draw_rectangle(data)
        plt.imshow(img)
        plt.show()
        self.assertTrue(True)
"""


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    img = make_rgb_img(image_height, image_width, background_color)
    img_center = (image_height // 2, image_width // 2)
    for i in range(image_height):
        for j in range(image_width):
            if (i - img_center[0]) ** 2 / semi_axis_y**2 + (
                j - img_center[1]
            ) ** 2 / semi_axis_x**2 <= 1:
                img[i, j] = shape_color
    return img


"""
class test_draw_ellipse(unittest.TestCase):
    def test_visual(self):
        data = EllipseInput(10, 5, 100, 100, (120, 120, 120), (0, 0, 0))
        img = draw_ellipse(data)
        plt.imshow(img)
        plt.show()
        self.assertTrue(True)
"""


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    math_expect = sum(values) / len(values)
    avg_squares = sum([i**2 for i in values]) / len(values)
    dispersion = avg_squares - math_expect**2
    square_deviation = dispersion**0.5
    loc_maxs = []
    loc_mins = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            loc_maxs.append(i)
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            loc_mins.append(i)
    slide_avg = []
    # breakpoint()
    for i in range(len(values) - window + 1):
        slide_avg.append(sum(values[i : window + i]) / window)
    res = TimeSeriesStatistics(
        math_expect, dispersion, square_deviation, loc_maxs, loc_mins, slide_avg
    )
    return res


class test_analyze_time_series(unittest.TestCase):
    def run_test(self, res, expect):
        # breakpoint()
        if res.mean - expect.mean > 0.01:
            return False
        if res.variance - expect.variance > 0.01:
            return False
        if res.std - expect.std > 0.01:
            return False
        if not np.array_equal(res.local_maxima_indices, expect.local_maxima_indices):
            return False
        if not np.array_equal(res.local_minima_indices, expect.local_minima_indices):
            return False
        if not np.array_equal(res.moving_average, expect.moving_average):
            return False
        return True

    def test_simple(self):
        data = TimeSeriesInput([1, 2, 3, 4, 5, 6], 2)
        res = analyze_time_series(data)
        expect = TimeSeriesStatistics(
            3.5, 2.917, 1.708, [], [], [1.5, 2.5, 3.5, 4.5, 5.5]
        )
        self.assertTrue(self.run_test(res, expect))


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    if len(labels) == 0:
        return [[]]
    if class_count is None:
        class_count = max(labels) + 1
    encode_mtrx = np.zeros((len(labels), class_count))
    for i in range(len(labels)):
        encode_mtrx[i, labels[i]] = 1
    return encode_mtrx


class test_one_hot(unittest.TestCase):
    def test_0x0(self):
        res = one_hot(OneHotInput([]))
        expect = [[]]
        self.assertTrue(np.array_equal(res, expect))

    def test_1x1(self):
        res = one_hot(OneHotInput([0]))
        expect = [[1]]
        self.assertTrue(np.array_equal(res, expect))

    def test_2x2(self):
        res = one_hot(OneHotInput([0, 1]))
        expect = [[1, 0], [0, 1]]
        self.assertTrue(np.array_equal(res, expect))

    def test_3x2(self):
        res = one_hot(OneHotInput([0, 0, 1]))
        expect = [[1, 0], [1, 0], [0, 1]]
        self.assertTrue(np.array_equal(res, expect))

    def test_3x3(self):
        res = one_hot(OneHotInput([1, 1, 2]))
        expect = [[0, 1, 0], [0, 1, 0], [0, 0, 1]]
        self.assertTrue(np.array_equal(res, expect))

    def test_4x3(self):
        res = one_hot(OneHotInput([0, 1, 2, 2]))
        expect = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
        self.assertTrue(np.array_equal(res, expect))

    def test_4x4(self):
        res = one_hot(OneHotInput([0, 2, 3, 0]))
        expect = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
        self.assertTrue(np.array_equal(res, expect))

    def test_0x0_count(self):
        res = one_hot(OneHotInput([], 0))
        expect = [[]]
        self.assertTrue(np.array_equal(res, expect))

    def test_1x1_count(self):
        res = one_hot(OneHotInput([0], 1))
        expect = [[1]]
        self.assertTrue(np.array_equal(res, expect))

    def test_2x2_count(self):
        res = one_hot(OneHotInput([0, 1], 2))
        expect = [[1, 0], [0, 1]]
        self.assertTrue(np.array_equal(res, expect))

    def test_3x2_count(self):
        res = one_hot(OneHotInput([0, 0, 1], 2))
        expect = [[1, 0], [1, 0], [0, 1]]
        self.assertTrue(np.array_equal(res, expect))

    def test_3x3_count(self):
        res = one_hot(OneHotInput([1, 1, 2], 3))
        expect = [[0, 1, 0], [0, 1, 0], [0, 0, 1]]
        self.assertTrue(np.array_equal(res, expect))

    def test_4x3_count(self):
        res = one_hot(OneHotInput([0, 1, 2, 2], 3))
        expect = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
        self.assertTrue(np.array_equal(res, expect))

    def test_4x4_count(self):
        res = one_hot(OneHotInput([0, 2, 3, 0], 4))
        expect = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
        self.assertTrue(np.array_equal(res, expect))
