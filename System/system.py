from System.thresh import ThreshAnalytical
from utils.hmatr import Hmatr
from utils.utils import find_Q_hat


class System:
    """
    Система для обнаружения неоднородности. Выводим момент нарушения однородности.
    """
    def __init__(self):
        self.F = None
        self.k = None
        self.w1 = None
        self.delta_min = None
        self.w_min = None
        self.sigma_sq = None

        self.initial_value = None
        self.B = None
        self.T = None
        self.L = None
        self.thresh = None

        self.row = None
        self.Q_hat = None

    def __call__(self, *args, **kwargs):
        self.F = kwargs.get('F')
        self.k = kwargs.get('k')
        self.w1 = kwargs.get('w1')
        self.delta_min = kwargs.get('delta_min')
        self.sigma_sq = kwargs.get('sigma_sq')

        self.w_min = self.w1 + self.delta_min
        self.initial_value = self.sigma_sq / (0.5 + self.sigma_sq)
        self.B = int(len(self.F) / 10)
        self.T = self.B
        self.L = int(0.4 * self.T)
        self.thresh = ThreshAnalytical(omega_1=self.w1, omega_2=self.w_min, L=self.L, T_=self.T, k=self.k,
                                       initial_value=self.initial_value)
        self.row = Hmatr(f=self.F, B=self.B, T=self.T, L=self.L, neig=2, svdMethod='svd').getRow()
        self.Q_hat = find_Q_hat(self.row, self.thresh.thresh)
        return self.Q_hat + self.T