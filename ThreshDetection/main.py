import time

import matplotlib.pyplot as plt
import numpy as np

from utils.hmatr import Hmatr
from ThreshDetection.thresh import ThreshAnalytical
from utils.utils import generate_series, find_Q_hat


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
    w_min = w1 + 1/10  # Минимальная разница в частотах для обнаружения неоднородности
    k = 30  # Кол-во точек, за которые нужно обнаружить разладку
    w2 = w_min
    C = 1
    phi1 = 0
    phi2 = 0
    Q = 301
    r = 2
    method = "svd"

    B = 100
    T_ = 100
    L = 60
    print(f"Params: w1 = {w1}, w2 = {w2}, w_min = {round(w_min, 5)}, L = {L}, k = {k}, T = {T_}")

    original_series = generate_series(w1, w2, Q, N)
    noise = np.random.randn(N) * 0.5 ** 2
    # original_series += noise
    hm = Hmatr(f=original_series, B=B, T=T_, L=L, neig=r, svdMethod=method)
    row = hm.getRow(sync=True)
    initial_value = np.quantile(row[:Q], 0.95)

    # Generate analytical approximation to row function
    g_analytical = ThreshAnalytical(w1, w_min, L, T_, k)
    print(f"Analytical thresh: {round(g_analytical.thresh, 5)}, "
          f"value after heterogeneity {round(g_analytical.value_after_heterogeneity, 5)}")
    print()

    approx = [0 for i in range(Q)]
    approx = [*approx, *g_analytical.transition_interval.tolist()]
    approx = [*approx, *[g_analytical.value_after_heterogeneity for i in range(len(row) - len(approx))]]
    assert len(approx) == len(row), f"Length are different: {len(approx)}, {len(row)}"

    Q_hat = find_Q_hat(row, g_analytical.thresh)
    if Q_hat is None:
        print("Неоднородность не обнаружена")
    else:
        print(f"Q_hat using analytical thresh: {Q_hat}, found using {Q_hat - Q} points, k = {k}")

        # Generate analytical approximation to row function
        g_analytical_correct = ThreshAnalytical(w1, w_min, L, T_, k, initial_value)
        print(f"Analytical thresh correct: {round(g_analytical_correct.thresh, 5)}, "
              f"value after heterogeneity {round(g_analytical_correct.value_after_heterogeneity, 5)}")
        print()
        approx_correct = [initial_value for i in range(Q)]
        approx_correct = [*approx_correct, *g_analytical_correct.transition_interval.tolist()]
        approx_correct = [*approx_correct, *[g_analytical_correct.value_after_heterogeneity for i in range(len(row) - len(approx_correct))]]
        assert len(approx) == len(row), f"Length are different: {len(approx)}, {len(row)}"
        Q_hat_correct = find_Q_hat(row, g_analytical_correct.thresh)
        print(f"Q_hat using analytical thresh: {Q_hat_correct}, found using {Q_hat_correct - Q} points, k = {k}")
        row = Hmatr(f=original_series, B=B, T=T_, L=L, neig=r, svdMethod=method).getRow(sync=True)
        w_min_str = r'$\omega_{min}$'

        original_series = generate_series(w1, w_min, Q, N)
        noise = np.random.randn(N) * 0.5 ** 2
        original_series += noise
        thresh = Hmatr(f=original_series, B=B, T=T_, L=L, neig=r, svdMethod=method).getRow(sync=True)[Q + k]

        plt.figure(figsize=(10, 5))
        plt.plot(row, label='Row')
        # plt.plot(approx, label='Approximation')
        plt.plot(approx, label='Approximation')
        # plt.plot(np.arange(len(row)), [thresh]*len(row), '--', label='Thresh classic')
        # plt.plot(np.arange(len(row)), [row[Q_hat_correct]] * len(row), '--', label='Thresh analytical')
        # plt.plot(Q_hat, row[Q_hat], marker='o')
        # plt.plot(Q+k, approx[Q+k], marker='o')
        # plt.title(f"w1 = {w1}, w2 = {w2}, w_min = {round(w_min, 5)}, L = {L}, k = {k}, T = {T_}")
        # plt.title(fr"$\omega_1$={w1}, $\omega_2$={1/5}, {w_min_str}={round(w2, 4)}, L={L}, B={B}, T={T_}")
        plt.title(fr"$\omega_1$={w1}, $\omega_2$={w2}, L={L}")
        plt.legend(loc='upper left')
        plt.show()


if __name__ == "__main__":
    test()
