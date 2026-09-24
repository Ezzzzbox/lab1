"""Заготовки задач на базовый Python."""

from typing import Sequence

from grader_contracts.python_basics import (
    PositiveIntegerInput,
    TextInput,
    VectorPairInput,
)


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


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    symbols_set = set()
    for i in range(len(text)):
        symbols_set.add(text[i])
        if i + 1 > len(symbols_set):
            return False
    return True


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


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    mse = 0
    if len(predicted) == 0:
        raise ValueError()
    for i in range(len(predicted)):
        mse += (predicted[i] - expected[i]) ** 2
    mse = mse / len(predicted)
    return mse


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    res = ""
    for i in range(2, number + 1):
        count = 0
        while number % i == 0:
            count += 1
            number = number // i
        if count > 1:
            res += "(" + str(i) + "**" + str(count) + ")"
        if count == 1:
            res += "(" + str(i) + ")"
    return res


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    if cube_count <= 0:
        return "It is impossible"
    count = 0
    i = 0
    while count < cube_count:
        i += 1
        count += i * i
    if count == cube_count:
        return i
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = []
    while number != 0:
        digits.append(number % 10)
        number = number // 10
    left = 0
    right = 0
    is_even_len = len(digits) % 2 == 0
    # breakpoint()
    if is_even_len:
        for i in range(len(digits) // 2 - 1):
            left += digits[i]
        for i in range(len(digits) // 2 + 1, len(digits)):
            right += digits[i]
    else:
        for i in range(len(digits) // 2):
            left += digits[i]
        for i in range(len(digits) // 2 + 1, len(digits)):
            right += digits[i]
    # breakpoint()
    return left == right
