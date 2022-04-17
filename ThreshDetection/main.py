import time

import matplotlib.pyplot as plt
import numpy as np

from utils.hmatr import Hmatr
from ThreshDetection.thresh import ThreshAnalytical
from utils.utils import generate_series


def find_Q_hat(series, thresh):
    for idx, val in enumerate(series):
        if val >= thresh:
            return idx
    return None


def main():
    N = 700  # Длина ряда
    w1 = 1 / 10  # Начальная частота
    w_min = w1 + 1/100  # Минимальная разница в частотах для обнаружения неоднородности
    s = 10
    w2 = 1 / 5
    C = 1
    phi1 = 0
    phi2 = 0
    Q = 301
    B = 100
    T_ = 100
    L = 50
    r = 2
    method = "svd"

    start_t = time.time()
    an = ThreshAnalytical(w1, w_min, L, s)
    time_analytical = time.time() - start_t

    print(f"Analytical thresh: {round(an.thresh, 5)}, elapsed {round(time_analytical, 5)}s | ")
    print()

    original_series = generate_series(w1, w2, Q, N)

    for noiseSD in np.arange(0, 1, 0.1):
        print("---------------------------------------------------------------------------------")
        eps = np.random.randn(N) * noiseSD ** 2
        print(f"Series with noise (sd = {noiseSD})")
        original_series_noised = original_series + eps
        hm = Hmatr(f=original_series_noised, B=B, T=T_, L=L, neig=r, svdMethod=method)
        row = hm.getRow(sync=True)
        plt.plot(row)
        plt.title(f"NoiseSD = {round(noiseSD, 4)}")
        plt.show()

        Q_hat_an = find_Q_hat(row, an.thresh)
        print(f"Q_hat using analytical thresh: {Q_hat_an}")
        if Q_hat_an is None:
            max_row_val = round(np.max(row), 5)

            print(f"Неоднородность по порогу {an.thresh} не обнаружена: \n"
                  "(1) либо задана слишком большая минимальная разница в частотах для "
                  "обнаружения неоднородности (параметр w_min);")

            if max_row_val > 0.8:
                print(f"Так как максимальное значение индекса неоднородности близко к 1 ({max_row_val}), "
                      f"неоднородность в ряде присутствует (возможно обусловлена шумом).")
            else:
                print("(2) либо неоднородности действительно нет, \n"
                      f"Максимальное значение индекса неоднородности для рассмотренного ряда {max_row_val}")
        elif Q_hat_an == 0:
            print("Начальное значение индекса неоднородности соответствует более сильному изменению в частотах ряда: \n"
                  "(1) Либо шум рассматриваемого ряда слишком сильный; \n"
                  "(2) либо надо увеличить минимальную разницу в частотах для обнаружения неоднородности "
                  "(параметр w_min); \n"
                  "(3) либо алгоритм попал на переходный интервал, т.к. не удается определить начальную структуру "
                  "ряда, а следовательно момент возмущения был ранее.")
        else:
            print(f"Момент возмущения в рассматриваемом ряде - {Q_hat_an}")
            print(f"Значения функции обнаружения: Row[Q_hat-1]: {round(row[Q_hat_an-1], 4)}, "
                  f"Row[Q_hat]: {round(row[Q_hat_an], 4)}, Row[Q_hat+1]: {round(row[Q_hat_an+1], 4)}")


def test():
    N = 700  # Длина ряда
    w1 = 1 / 10  # Начальная частота
    w_min = w1 + 1 / 200  # Минимальная разница в частотах для обнаружения неоднородности
    k = 30  # Кол-во точек, за которые нужно обнаружить разладку
    w2 = 1 / 10
    C = 1
    phi1 = 0
    phi2 = 0
    Q = 301
    B = 100
    T_ = 100
    L = 80
    r = 2
    method = "svd"
    print(f"Params: w1 = {w1}, w2 = {w2}, w_min = {round(w_min, 5)}, L = {L}, k = {k}")

    g_analytical = ThreshAnalytical(w1, w_min, L, T_, k)
    print(f"Analytical thresh: {round(g_analytical.thresh, 5)}, "
          f"value after heterogeneity {round(g_analytical.value_after_heterogeneity, 5)}")
    print()

    original_series = generate_series(w1, w2, Q, N)
    hm = Hmatr(f=original_series, B=B, T=T_, L=L, neig=r, svdMethod=method)
    row = hm.getRow(sync=True)
    Q_hat = find_Q_hat(row, g_analytical.thresh)
    if Q_hat is None:
        print("Неоднородность не обнаружена")
    else:
        print(f"Q_hat using analytical thresh: {Q_hat}, found using {Q_hat - Q} points, k = {k}")

        # Generate analytical approximation to row function
        approx = [0 for i in range(Q-1)]
        approx = [*approx, *g_analytical.transition_interval.tolist()]
        approx = [*approx, *[g_analytical.value_after_heterogeneity for i in range(len(row) - len(approx))]]
        assert len(approx) == len(row), f"Length are different: {len(approx)}, {len(row)}"

        plt.figure(figsize=(7, 5))
        plt.plot(row, label='Row')
        plt.plot(approx, label='Approximation')
        plt.plot(np.arange(len(row)), [row[Q_hat]]*len(row), '--', label='Thresh')
        plt.plot(Q_hat, row[Q_hat], marker='o')
        plt.title(f"w1 = {w1}, w2 = {w2}, w_min = {round(w_min, 5)}, L = {L}, k = {k}")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    test()
