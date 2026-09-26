"""Заготовки задач на NumPy."""

import numpy as np

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
    sum = np.zeros((len(vectors[0]), 1))
    for i in range(len(matrices)):
        sum += matrices[i] @ vectors[i]
    return sum


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


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    if len(matrix.shape) == 1:
        return unique_rows_vector(matrix)
    unique_rows = []
    for i in range(matrix.shape[0]):
        unique_rows.append(unique_list(matrix[i]))
    for i in range(len(unique_rows)):
        unique_rows[i].sort()
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
    for i in range(len(res)):
        res[i].sort()
    return res


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = (
        data.rows,
        data.columns,
        data.mean,
        data.std,
        data.seed,
    )

    generator = np.random.default_rng(seed)
    mtrx = generator.normal(mean, std, size=(rows, columns))
    return MatrixStatistics(
        matrix=mtrx,
        row_means=np.mean(mtrx, axis=1),
        column_means=np.mean(mtrx, axis=0),
        row_variances=np.var(mtrx, axis=1),
        column_variances=np.var(mtrx, axis=0),
    )


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


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    img = make_rgb_img(image_height, image_width, background_color)
    y0 = (image_width - width) // 2
    x0 = (image_height - height) // 2
    for i in range(height):
        for j in range(width):
            img[x0 + i, y0 + j] = shape_color
    return img


def make_rgb_img(width, height, background_color):
    img = np.zeros((width, height, 3), dtype=np.int64)
    for i in range(width):
        for j in range(height):
            img[i, j] = background_color
    return img


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    img = make_rgb_img(image_height, image_width, background_color)

    y0 = (image_height - 1) / 2
    x0 = (image_width - 1) / 2

    for i in range(image_height):
        for j in range(image_width):
            y_part = (i - y0) / semi_axis_y
            x_part = (j - x0) / semi_axis_x

            if y_part**2 + x_part**2 <= 1:
                img[i, j] = shape_color
    return img


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


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    if len(labels) == 0:
        return [[]]
    if class_count is None:
        class_count = max(labels) + 1
    encode_mtrx = np.zeros((len(labels), class_count), dtype=np.int64)
    for i in range(len(labels)):
        encode_mtrx[i, labels[i]] = 1
    return encode_mtrx
