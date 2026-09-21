"""Заготовки задач на NumPy."""

import numpy as np

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
    raise NotImplementedError  # TODO


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    raise NotImplementedError  # TODO


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    raise NotImplementedError  # TODO


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = (
        data.rows,
        data.columns,
        data.mean,
        data.std,
        data.seed,
    )
    raise NotImplementedError  # TODO


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
