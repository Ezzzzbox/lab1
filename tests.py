import numpy as np
import unittest
from numpy_tasks import *
from python_basics import *


class test_is_balanced_number(unittest.TestCase):
    def test_zero(self):
        num = PositiveIntegerInput(0)
        res = is_balanced_number(num)
        self.assertTrue(res)  # а что делать с цифрами?

    def test_balanced_not_even_len(self):
        num = PositiveIntegerInput(123404321)
        res = is_balanced_number(num)
        self.assertTrue(res)

    def test_balanced_even_len(self):
        num = PositiveIntegerInput(1234004321)
        res = is_balanced_number(num)
        self.assertTrue(res)

    def test_unbalanced_not_even_len(self):
        num = PositiveIntegerInput(123406789)
        res = is_balanced_number(num)
        self.assertFalse(res)

    def test_unbalanced_even_len(self):
        num = PositiveIntegerInput(1234006789)
        res = is_balanced_number(num)
        self.assertFalse(res)


class test_pyramid(unittest.TestCase):
    def test_one(self):
        num = PositiveIntegerInput(1)
        res = pyramid(num)
        self.assertEqual(res, "k")

    def test_multiple(self):
        num = 0
        err_nums = []
        for i in range(1, 100):
            num += i * i
            inp_num = PositiveIntegerInput(num)
            res = pyramid(inp_num)
            if res != "k":
                err_nums.append(i)
        self.assertEqual([], err_nums)

    def test_incorrect_simle(self):
        num = PositiveIntegerInput(7)
        res = pyramid(num)
        self.assertEqual(res, "It is impossible")

    def test_incorrect(self):
        num = PositiveIntegerInput(500)
        res = pyramid(num)
        self.assertEqual(res, "It is impossible")


class test_prime_factorization(unittest.TestCase):
    def test_zero(self):
        num = PositiveIntegerInput(0)
        res = prime_factorization(num)
        self.assertEqual(res, "")

    def test_simple(self):
        num = PositiveIntegerInput(100)
        res = prime_factorization(num)
        self.assertEqual(res, "(2**2)(5**2)")

    def tet_prime_number(self):
        num = PositiveIntegerInput(219)
        res = prime_factorization(num)
        self.assertEqual(res, "(217)")


class test_mse(unittest.TestCase):
    def test_empty(self):
        pair = VectorPairInput([], [])
        with self.assertRaises(ValueError):
            mse(pair)

    def test_zero_mse(self):
        pair = VectorPairInput([1, 2, 3], [1, 2, 3])
        res = mse(pair)
        self.assertEqual(res, 0)

    def test_simple(self):
        pair = VectorPairInput([1, 2], [0, 0])
        res = mse(pair)
        self.assertEqual(res, 2.5)


class test_multiplicative_persistance(unittest.TestCase):
    def test_digit(self):
        allpassed = True
        for i in range(10):
            num = PositiveIntegerInput(i)
            if multiplicative_persistence(num) != 0:
                allpassed = False
        self.assertTrue(allpassed)

    def test_number(self):
        num = PositiveIntegerInput(555)
        res = multiplicative_persistence(num)
        self.assertEqual(res, 3)


class test_count_bin_ones(unittest.TestCase):
    def test_zero(self):
        num = PositiveIntegerInput(0)
        res = count_one_bits(num)
        self.assertEqual(res, 0)

    def test_simple(self):
        num = PositiveIntegerInput(1023)
        res = count_one_bits(num)
        self.assertEqual(res, 10)

    def test_negative(self):
        num = PositiveIntegerInput(-1)
        with self.assertRaises(ValueError):
            count_one_bits(num)


class test_symbols_unique(unittest.TestCase):
    def test_ascii(self):
        s = ""
        for i in range(255):
            s += chr(i)
        res = has_unique_characters(TextInput(s))
        self.assertEqual(res, True)

    def test_empty(self):
        res = has_unique_characters(TextInput(""))
        self.assertEqual(res, True)

    def test_repetition(self):
        s = TextInput("abcd11")
        res = has_unique_characters(s)
        self.assertEqual(res, False)


class test_vovels_count(unittest.TestCase):
    def test_all_letters(self):
        res = count_vowels(TextInput("aeiouAEIOU"))
        self.assertEqual(res, 10)

    def test_empty(self):
        res = count_vowels(TextInput(""))
        self.assertEqual(res, 0)

    def test_ascii(self):
        sequence = ""
        for i in range(255):
            sequence += chr(i)
        res = count_vowels(TextInput(sequence))
        self.assertEqual(res, 10)


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


"""
def draw_histogram(matrix):
    plt.hist(matrix)
    plt.show()
"""


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


"""
class test_draw_rectangle(unittest.TestCase):
    def test_visual(self):
        data = RectangleInput(30, 30, 100, 100, (255, 0, 0), (120, 120, 120))
        img = draw_rectangle(data)
        plt.imshow(img)
        plt.show()
        self.assertTrue(True)
"""


"""
class test_draw_ellipse(unittest.TestCase):
    def test_visual(self):
        data = EllipseInput(10, 5, 100, 100, (120, 120, 120), (0, 0, 0))
        img = draw_ellipse(data)
        plt.imshow(img)
        plt.show()
        self.assertTrue(True)
"""


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
