import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

rpy2.robjects.numpy2ri.activate()
utils = importr('utils')
utils.chooseCRANmirror(ind=1)
utils.install_packages('Rssa')
rssa = importr('Rssa')


class Hmatr:
    """
    Main class of heterogeneity matrix. Able to use in real-time (method update_hmatr).
    """
    def __init__(self, f, B=0, T=0, L=0, neig=0, svdMethod='propack'):
        self.f = f
        self.N = len(f)
        self.L = L if L != 0 else B // 4
        self.K = self.N - self.L + 1
        self.B = B if B != 0 else self.N // 4
        self.T = T if T != 0 else self.N // 4

        self.NT = self.N - self.T + 1
        self.KT = self.T - self.L + 1

        self.neig = neig if neig != 0 else 10
        self.svdMethod = svdMethod

        self.th = None
        self.cth2_cumsum = None
        self.U = []
        self.hmatr = self.__compute_hmatr()

    def __call__(self):
        return self.hmatr

    def __compute_cXU2(self, i):
        return np.r_[0, np.cumsum(np.sum((np.matmul(self.th, self.U[i])) ** 2, axis=1))]

    def __compute_hmatr(self):
        self.th = np.transpose(np.array(rssa.hankel(robjects.FloatVector(self.f), L=self.L)))
        self.cth2_cumsum = np.r_[0, np.cumsum(np.sum(self.th ** 2, axis=1))]
        cth2 = (self.cth2_cumsum[self.KT:self.K] - self.cth2_cumsum[:self.NT - 1])

        def hc(idx):
            Fb = self.f[idx:(idx + self.B)]
            s = rssa.ssa(robjects.FloatVector(Fb), L=self.L, neig=min(2 * self.neig, 50), svd_method=self.svdMethod)
            self.U.append(np.array(rssa._U_ssa(s))[:, :self.neig])
            cXU2 = self.__compute_cXU2(-1)
            sumDist = (cXU2[self.KT:(self.N - self.L + 1)] - cXU2[:self.NT - 1])
            return 1 - sumDist / cth2

        h = np.r_[[hc(i) for i in range(0, self.N - self.B)]]
        return h

    def getNewRow(self):
        cth2 = self.cth2_cumsum[(self.T - self.L + 1):(self.N - self.L + 1)] - self.cth2_cumsum[:(self.N - self.T)]
        s = rssa.ssa(robjects.FloatVector(self.f[(self.N - self.B):self.N]), L=self.L, neig=min(2 * self.neig, 50),
                     svd_method=self.svdMethod)
        self.U.append(np.array(rssa._U_ssa(s))[:, :self.neig])
        cXU2 = self.__compute_cXU2(-1)
        sumDist = cXU2[self.KT:(self.N - self.L + 1)] - cXU2[:self.N - self.T]
        return 1 - sumDist / cth2

    def getNewCol(self):
        newCol = np.array(())
        for i in range(self.N - self.T - 1):
            cXU2 = self.__compute_cXU2(i)
            sumDist = cXU2[self.K] - cXU2[self.N - self.T]
            cth2 = self.cth2_cumsum[self.K] - self.cth2_cumsum[self.N - self.T]
            newCol = np.r_[newCol, (1 - sumDist / cth2)]
        return newCol.reshape(-1, 1)

    def update_hmatr(self, newValueOfSeries):
        self.K += 1
        self.N += 1
        self.NT += 1
        self.f.append(newValueOfSeries)
        self.th = np.vstack(
            [self.th, np.array(self.f[self.K - 1:self.N]).reshape(1, -1)])  # ?????????????????? ?? ?????????????????? ?????????????? ?????????? ????????????
        self.cth2_cumsum = np.r_[self.cth2_cumsum, self.cth2_cumsum[-1] + np.sum(
            self.th[-1] ** 2)]  # ?????????????????? ?????????? ?????????????? ?? ???????????????????????? ??????????
        row = self.getNewRow()
        col = self.getNewCol()
        self.hmatr = np.vstack([np.hstack([self.hmatr, col]), row])  # ?????????????????? ?????????????? ???? ???????? ???????????? ?? ??????????????

    def getRow(self, n=0, sync=False):
        if sync:
            return np.r_[np.zeros(self.T) + np.mean(self.hmatr[:self.T, 0]), self.hmatr[n, :]]
        return self.hmatr[n, :]

    def getCol(self, n=0, sync=False):
        if sync:
            return np.r_[np.zeros(self.B) + np.mean(self.hmatr[:self.B, 0]), self.hmatr[:, n]]
        return self.hmatr[:, n]

    def getSym(self, sync=False):
        if sync:
            return np.r_[
                np.zeros(self.N - len(self.hmatr.diagonal())) + np.mean(self.hmatr[:self.B, 0]), self.hmatr.diagonal()]
        return self.hmatr.diagonal()

    def getDiag(self, sync=False):
        if sync:
            return np.r_[np.zeros(self.N - len(self.hmatr.diagonal(self.B + 1))) + np.mean(
                self.hmatr.diagonal(self.B + 1)[:self.B]), self.hmatr.diagonal(self.B + 1)]
        return self.hmatr.diagonal(self.B + 1)

    def plotHeterFunc(self, title='Heterogenety Functions', w=16, h=4):
        plt.figure(figsize=(w, h))
        plt.title(title)
        plt.plot(np.arange(self.T, self.N), self.getRow(0), label='row')
        plt.plot(np.arange(self.B, self.N), self.getCol(0), label='col')
        plt.plot(np.arange(self.T, self.N), self.getSym(), label='symmetric')  # self.T - seems to be wrong.
        plt.plot(np.arange(self.B + self.T + 1, self.N), self.getDiag(), label='diag')
        plt.legend()
        plt.show()

    def plotHm(self, title='HM', w=4, h=4):
        plt.figure(figsize=(w, h))
        plt.title(title)
        plt.xlim([0, self.N - self.T + 1])
        plt.ylim([0, self.N - self.B + 1])
        plt.imshow(self.hmatr)
