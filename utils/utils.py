import numpy as np


def generate_series(w1, w2, Q, N):
    """
    Генерирует ряд с разладкой в точке Q и частотами до и после w1 и w2 соответственно.
    :param w1: Частота ряда до разладки
    :param w2: Частота ряда после разладки
    :param Q: Точка разладки
    :param N: Длина ряда
    :return:
    """
    series = lambda n: np.sin(2 * np.pi * w1 * n) if n < Q - 1 else np.sin(2 * np.pi * w2 * n)
    return [series(i) for i in range(N)]
