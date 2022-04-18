import time

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils.utils import find_Q_hat
from ThreshDetection.thresh import ThreshAnalytical
from utils.hmatr import Hmatr
from utils.utils import generate_series, find_fpr_corr_to_thresh

# Parameters
# Fixing: N = 700, w1 = 1 / 10, w_min = w1 + 1 / 100, k = 30, w2 = 1 / 5, Q = 301, noise_sd = 0.5, iter_num = 200
N = 700  # Длина ряда
w1 = 1 / 10  # Начальная частота
w_min = w1 + 1 / 100  # Минимальная разница в частотах для обнаружения неоднородности
k = 30  # Кол-во точек, за которые нужно обнаружить разладку
w2 = w_min
Q = 301
r = 2
method = "svd"
noise_sd = 0.5
B = 100
T_ = 100
L = 80
iter_num = 200


def modelling_series():
    np.random.seed(12345)
    series = []
    rows = []
    time_start = time.time()
    for i in tqdm(range(iter_num)):
        original_series = generate_series(w1, w2, Q, N)
        noise = np.random.randn(N) * noise_sd ** 2
        original_series += noise
        hm = Hmatr(f=original_series, B=B, T=T_, L=L, neig=r, svdMethod=method)
        rows.append(hm.getRow(sync=True))
        series.append(original_series)
    print(f"Modelling took {round(time.time() - time_start, 4)} s")
    return series, rows


def main():
    print(f"Params: w1 = {w1}, w2 = {w2}, w_min = {round(w_min, 5)}, L = {L}, k = {k}, T = {T_}")

    modelled_series, modelled_rows = modelling_series()
    threshes = np.arange(0, 1.01, 0.01)

    fpr = []  # Детекции раньше настоящего момента возмущения
    tpr = []  # Детекции в или позже момента возмущения
    ttpr = []  # Детекции в или позже момента возмущения, но не позже момента k

    time_start = time.time()
    for thresh in threshes:
        fp = 0
        tp = 0
        ttp = 0
        for row in modelled_rows:
            Q_hat = find_Q_hat(row, thresh)
            if Q_hat is None:
                # Не превзошли порог, пропускаем значение. При подсчете долей кривые будут уходить в 0
                continue
            if Q_hat < Q:
                fp += 1
            if Q_hat >= Q:
                tp += 1
                if Q_hat <= Q + k:
                    ttp += 1
        fpr.append(round(fp/iter_num, 4))
        ttpr.append(round(ttp/iter_num, 4))
        tpr.append(round(tp/iter_num, 4))

    g_analytical = ThreshAnalytical(w1, w_min, L, T_, k)

    print(f"Calculating statistics took {round(time.time() - time_start, 4)} s")
    plt.figure(figsize=(20, 10))
    plt.plot([g_analytical.thresh]*len(threshes), threshes, '--', label='Analytical thresh')
    plt.plot(threshes, fpr, label='FPR')
    plt.plot(threshes, tpr, label='TPR')
    plt.plot(threshes, ttpr, label='TTPR')
    plt.title(f"w1={w1}, w2={w2}, w_min={round(w_min, 4)}, k={k}, L={L}, T={T_}, iters={iter_num}, noise_sd={noise_sd}")
    plt.xlabel('Thresh')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid()
    plt.show()

    fpr_to_thresh = find_fpr_corr_to_thresh(fpr, g_analytical.thresh)
    print(f"FPR corresponding to analytical thresh: {fpr_to_thresh}")


if __name__ == "__main__":
    main()
