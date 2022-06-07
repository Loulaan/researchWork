import numpy as np
from scipy.stats import norm


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


def generate_series_amp(w1, w2, Q, N, c1, c2):
    """
    Генерирует ряд с разладкой в точке Q и частотами до и после w1 и w2 соответственно.
    :param w1: Частота ряда до разладки
    :param w2: Частота ряда после разладки
    :param Q: Точка разладки
    :param N: Длина ряда
    :return:
    """
    series = lambda n: c1 * np.sin(2 * np.pi * w1 * n) if n < Q - 1 else c2 * np.sin(2 * np.pi * w2 * n)

    return [series(i) for i in range(N)]


def find_fpr_corr_to_thresh(threshes, fprs, analytical_thresh):
    array = np.asarray(threshes)
    idx = (np.abs(array - analytical_thresh)).argmin()
    return fprs[idx]


def find_Q_hat(series, thresh):
    for idx, val in enumerate(series):
        if val >= thresh:
            return idx
    return None


def get_confidence_interval(statistics, iters):
    return np.array(norm.interval(0.95, loc=np.mean(statistics, axis=0), scale=np.std(statistics, axis=0) / iters))
