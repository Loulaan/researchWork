import time

import matplotlib.pyplot as plt
import numpy as np

from utils.hmatr import Hmatr
from Confirence.thresh import ThreshAnalytical, ThreshExact
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

    start_t = time.time()
    ex = ThreshExact(w1, w_min, L, B, T_, s, r, method)
    time_exact = time.time() - start_t
    print(f"Analytical thresh: {round(an.thresh, 5)}, elapsed {round(time_analytical, 5)}s | "
          f"Exact thresh: {round(ex.thresh, 5)}, elapsed {round(time_exact, 5)}s")
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
        Q_hat_ex = find_Q_hat(row, ex.thresh)
        print(f"Q_hat using analytical thresh: {Q_hat_an},| Q_hat using exact thresh: {Q_hat_ex}")
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


if __name__ == "__main__":
    main()
