"""Заготовки задач на базовый Python."""

from typing import Sequence

from grader_contracts.python_basics import (
    PositiveIntegerInput,
    TextInput,
    VectorPairInput,
)

import unittest


def count_vowels(data: TextInput) -> int:
    text = data.value
    vovels = ["a", "e", "i", "o", "u", "A", "E", "I", "O", "U"]
    count = 0
    for i in range(len(text)):
        for vovel in vovels:
            if text[i] == vovel:
                count += 1
                break
    return count


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


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    symbols_set = set()
    for i in range(len(text)):
        symbols_set.add(text[i])
        if i + 1 > len(symbols_set):
            return False
    return True


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


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    if number < 0:  # разумно ли считать что может быть
        raise ValueError()  # на входе отрицательное число?
        # наверно это должен класс гарантировать

    bin_str = str(bin(number)[2:])
    count = 0
    for i in range(len(bin_str)):
        if bin_str[i] == "1":
            count += 1
    return count


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


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    count = 0
    while number % 10 != number:
        temp = 1
        while number != 0:
            temp = temp * (number % 10)
            number = number // 10
        number = temp
        count += 1
    return count


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


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    mse = 0
    if len(predicted) == 0:
        raise ValueError()
    for i in range(len(predicted)):
        mse += (predicted[i] + expected[i]) ** 2
    mse = mse / len(predicted)
    return mse


class test_mse(unittest.TestCase):
    def test_empty(self):
        pair = VectorPairInput([], [])
        with self.assertRaises(ValueError):
            mse(pair)

    def test_zeromse(self):
        pair = VectorPairInput([1, 2, 3], [1, 2, 3])
        res = mse(pair)
        self.assertEqual(res, 0)

    def test_simple(self):
        pair = VectorPairInput([1, 2], [0, 0])
        res = mse(pair)
        self.assertEqual(res, 2.5)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    res = ""
    for i in range(number + 1):
        count = 0
        while number % i == 0:
            count += 1
        if count > 1:
            res += "(" + str(i) + "**" + str(count) + ")"
        if count == 1:
            res += "(" + str(i) + ")"
    return res


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    raise NotImplementedError  # TODO


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    raise NotImplementedError  # TODO
