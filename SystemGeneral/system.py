import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
import warnings

from System.thresh import ThreshAnalytical
from utils.hmatr import Hmatr
from utils.utils import find_Q_hat

warnings.filterwarnings('ignore')

rpy2.robjects.numpy2ri.activate()
utils = importr('utils')
utils.chooseCRANmirror(ind=1)
utils.install_packages('Rssa')
rssa = importr('Rssa')


class System:
    def __init__(self, f, k, delta_min):
        self.series = f
        self.k = k
        self.delta_min = delta_min

        self.w1 = self._estimate_omega_1()
        self.w_min = self.w1 + self.delta_min
        self.r = 2

        self.B = int(len(self.series) / 10)
        self.T = self.B
        self.L = int(0.4 * self.T)
        self.initial_value = self._estimate_gamma_min()
        self.thresh = ThreshAnalytical(omega_1=self.w1, omega_2=self.w_min, L=self.L, T_=self.T, k=self.k,
                                       initial_value=self.initial_value)
        self.row = Hmatr(f=self.series, B=self.B, T=self.T, L=self.L, neig=2, svdMethod='svd').getRow()
        self.Q_hat = find_Q_hat(self.row, self.thresh.thresh)

    def _estimate_omega_1(self):
        autogrouping = robjects.r('''
            install.packages("Rssa")
            library(Rssa)
            autogrouping = grouping.auto.wcor
        ''')
        frequency = None
        ssa = rssa.ssa(robjects.FloatVector(self.series[:len(self.series)/4]), L=self.L, neig=10, svd_method='svd')
        groups = autogrouping(ssa, nclust=3, groups=robjects.r('''1:10'''), method="complete")
        g_form = ""
        for group in groups:
            g_form += "c(" + ''.join(str(list(group)))[1:-1] + '), '
        g_form = g_form[:-2]

        groups_est = rssa.parestimate(ssa, groups=robjects.r(f'list({g_form})'))
        for i, g_est in enumerate(groups_est):
            # estimate rates
            periods = {}
            max_period = 0
            number = -1
            for num, (period, mod) in enumerate(zip(g_est[1], g_est[-2])):
                period = np.abs(period)
                if period != np.inf and np.round(mod, 1) == 1:
                    if periods.get(np.round(period, 5)) is not None:
                        periods[np.round(period, 5)] += 1
                        if periods[np.round(period, 5)] == 2 and max_period < np.round(period, 5) < len(self.series)/4:
                            max_period = np.round(period, 5)
                            number = num
                    else:
                        periods[np.round(np.abs(period), 5)] = 1
            if number == -1:
                continue
            estimated_freq = np.abs(np.round(g_est[2][number], 3))
            if estimated_freq == 0:
                continue
            frequency = estimated_freq
            break
        return frequency

    def _estimate_gamma_min(self):
        return np.max(
            Hmatr(f=self.series[:len(self.series)/4], B=self.B, T=self.T, L=self.L, neig=2, svdMethod='svd').getRow()
        )

    def __call__(self, *args, **kwargs):
        return self.Q_hat + self.T
