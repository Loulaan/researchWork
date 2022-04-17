from typing import List

import numpy as np
from numpy import sin, cos, pi
import rpy2.robjects as robjects
from utils.utils import generate_series
from rpy2.robjects.packages import importr

rssa = importr('Rssa')


class ThreshAnalytical:
    """
    Класс аналитической строковой функции разладки, вычисленной для ряда, заданного синусом или косинусом
    """

    def __init__(self, omega_1, omega_2, L=0, s=5):
        """
        :param omega_1: Начальная частота ряда
        :param omega_2: Минимальная частота для детекции разладки
        :param L: Длина окна
        :param s: Количество точек для определения неоднородности, s < L
        """
        assert s < L, "Parameter s is too large"
        self.s = s

        self.omega_1 = omega_1
        self.omega_2 = omega_2
        self.L = L
        self.N = int(10 * self.L)
        self.K = self.N - self.L

        self.Q = int(self.N / 2)
        self.omegas = [[omega_1, omega_2] if it < self.Q - 1 else [omega_1, omega_2] for it in range(self.N)]
        self.f = generate_series(self.omega_1, self.omega_2, self.Q, self.N)

        self.row = self.compute_row_analytical()

    def compute_val_analytical(self, w1, w2):
        a = w1 + w2
        b = w1 - w2
        if w1 == w2:
            first = (self.L / 2 - sin(4 * pi * self.L * w1) / (8 * pi * w1)) ** 2
            second = (sin(2 * pi * self.L * w1) ** 2 / (4 * pi * w1)) ** 2
        else:
            first = (sin(2 * pi * self.L * b) / (4 * pi * b) - sin(2 * pi * self.L * a) / (4 * pi * a)) ** 2
            second = ((cos(2 * pi * self.L * b) - 1) / (4 * pi * b) - (cos(2 * pi * self.L * a) - 1) / (
                    4 * pi * a)) ** 2
        numerator = first + second
        denom = np.square(self.L / 2)
        ratio = numerator / denom
        return 1 - ratio

    def compute_row_analytical(self):
        # print("omegas: ", self.omegas)
        return [self.compute_val_analytical(*ws) for ws in self.omegas]

    @property
    def thresh(self):
        return self.row[self.Q + self.s]


class ThreshExact:
    def __init__(self, omega_1, omega_2, L=0, B=0, T=0, s=5, neig=0, svdMethod='propack'):
        """
        :param omega_1:
        :param omega_2:
        :param L:
        :param B:
        :param T:
        :param s:
        :param neig:
        :param svdMethod:
        """
        assert s < L, "Parameter s is too large"
        self.s = s

        self.omega_1 = omega_1
        self.omega_2 = omega_2
        self.L = L if L != 0 else B // 4
        self.N = int(5 * self.L)
        self.K = self.N - self.L

        self.Q = int(self.N / 2)
        self.omegas = [omega_1 if it < self.Q - 1 else omega_2 for it in range(self.N)]
        self.f = generate_series(self.omega_1, self.omega_2, self.Q, self.N)

        self.B = B if B != 0 else self.N // 4
        self.T = T if T != 0 else self.N // 4

        self.NT = self.N - self.T
        self.KT = self.T - self.L

        self.NB = self.N - self.B
        self.KB = self.B - self.L

        self.neig = neig if neig != 0 else 10
        self.svdMethod = svdMethod

        self.th = np.transpose(np.array(rssa.hankel(robjects.FloatVector(self.f), L=self.L)))
        self.U = None
        self.row = self.compute_single_row(0)

    def compute_single_row(self, row_id):
        """
        Подсчет строковой функции разладки номер row_id. Более медленный метод, но зато намного нагляднее.
        :param row_id: int
        :return:
        """

        def _compute_ratio(idx, numer, denom):
            numerator = np.sum(numer[idx:(idx + self.KT)])
            denominator = np.sum(denom[idx:(idx + self.KT)])
            return np.round(numerator / denominator, 8)

        Fb = self.f[row_id:(row_id + self.B)]
        s = rssa.ssa(robjects.FloatVector(Fb), L=self.L, neig=min(2 * self.neig, 50), svd_method=self.svdMethod)
        self.U = np.array(rssa._U_ssa(s))[:, :self.neig]
        data_for_numerator = np.square(np.dot(self.th, self.U))
        data_for_denominator = np.square(np.linalg.norm(self.th, axis=1))
        ratio = np.array([_compute_ratio(idx, data_for_numerator, data_for_denominator) for idx in range(self.NT)])
        return 1 - ratio

    @property
    def thresh(self):
        return self.row[self.Q + self.s]
