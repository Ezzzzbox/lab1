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
    breakpoint()
    for i in range(len(mtrx)):
        rows_dis.append(count_dispersion(mtrx[i], rows_avg[i]))
    for j in range(mtrx.shape[1]):
        columns_dis.append(count_dispersion(mtrx[:, j], columns_avg[j]))
    return MatrixStatistics(mtrx, rows_avg, columns_avg, rows_dis, columns_dis)


def draw_histogram(matrix):
    plt.hist(matrix)
    plt.show()


temp = RandomMatrixInput(10, 10, 5, 0.5, 1)
draw_histogram(matrix_statistics(temp).matrix)


def count_dispersion(ls, avg):
    mean_square = sum([i**2 for i in ls]) / len(ls)
    avg_square = avg * avg
    return mean_square - avg_square


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    raise NotImplementedError  # TODO


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    raise NotImplementedError  # TODO


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    raise NotImplementedError  # TODO


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    raise NotImplementedError  # TODO


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    raise NotImplementedError  # TODO
