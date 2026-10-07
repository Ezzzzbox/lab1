from python_basics import prime_factorization
from grader_contracts.python_basics import PositiveIntegerInput
import unittest


class test_fact(unittest.TestCase):
    def run_tests(self, num, expec):
        tmp = PositiveIntegerInput(num)
        res = prime_factorization(tmp)
        self.assertEqual(res, expec)

    def test_neg_simple(self):
        self.run_tests(-1, "")

    def test_neg_big(self):
        self.run_tests(-111111, "")

    def test_zero(self):
        self.run_tests(0, "")

    def test_one(self):
        self.run_tests(1, "")

    def test_two(self):
        self.run_tests(2, "(2)")

    def test_prime(self):
        self.run_tests(17, "(17)")

    def test_big_prime(self):
        self.run_tests(11117, "(11117)")

    def test_square2(self):
        self.run_tests(4, "(2**2)")

    def test_square17(self):
        self.run_tests(289, "(17**2)")

    def test_six(self):
        self.run_tests(6, "(2)(3)")

    def test_210(self):
        self.run_tests(210, "(2)(3)(5)(7)")

    def test_big(self):
        self.run_tests(86240, "(2**5)(5)(7**2)(11)")
