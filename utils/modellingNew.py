import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
import openpyxl
import pandas as pd

from utils.hmatr import Hmatr

rpy2.robjects.numpy2ri.activate()
utils = importr('utils')
utils.chooseCRANmirror(ind=1)
# utils.install_packages('Rssa')
rssa = importr('Rssa')



def create_dataframe(name):
    pass


def measureStatistics(func, Q, tail):
        return np.max(func[:(Q-tail)]), np.quantile(func[:(Q-tail)], 0.95)




def modellingNoiseStatistics(dictSeries:dict, iterNum:int, N:int, B:int, T:int, Q:int, L:int, r:int, method:str, vareps:float):
    '''
    Моделирование статистик ряда (средний 95й процентиль и средний максимум) при различных реализациях шума до момента разладки методом Монте-Карло.
    Внимание, шум добавляется внутри метода!
    :param dict dictSeries: The dictionary where key is the type of series and value is a series. Example: { 'Permanent': [x_1, ..., x_N] }.
    :param int iterNum: Number of iterations for modelling.
    :param int N: The len of series.
    :param int B: The len of base subseries.
    :param int T: The len of test subseries.
    :param int Q: The point of perturbation.
    :param int L: The window len.
    :param int r: Number of eigen vectors.
    :param str method: SVD method.
    :param float vareps: Variance of the noise.
    :return: Pandas DataFrame
    '''
    ds = pd.DataFrame(columns=["HeterType", "StatType", "row", "col", "sym", "diag"])
    ds["HeterType"] = ['Permanent', 'Permanent', 'Temporary', 'Temporary', 'Shifted', 'Shifted', 'Outlier', 'Outlier']
    ds["StatType"] = ["meanMax", "mean95Procentile"]*4
    
    row, col, sym, diag = [], [], [], []
    for num, typeV in enumerate(dictSeries.keys()):
        statsRow, statsCol, statsSym, statsDiag = [], [], [], []
        
        
        for i in range(iterNum):
            eps = np.random.randn(N) * vareps**2
            if typeV == 'Temporary':
                eps[:Q] = eps[:Q]/2
            
            seriesNoise = dictSeries[typeV] + eps
            hm = Hmatr(seriesNoise, B, T, L, neig=r, svdMethod=method)
            statsRow.append(measureStatistics(hm.getRow(), Q, hm.T))
            statsCol.append(measureStatistics(hm.getCol(), Q, hm.B))
            statsSym.append(measureStatistics(hm.getSym(), Q, hm.T))
            statsDiag.append(measureStatistics(hm.getDiag(), Q, hm.B + hm.T + 1))
        
        print("StatsRow:\n", statsRow,)
        print("StatsRow:\n", np.mean(statsRow, axis=1))
        row.append(np.mean(statsRow, axis=1))
        col.append(np.mean(statsCol, axis=1))
        sym.append(np.mean(statsSym, axis=1))
        diag.append(np.mean(statsDiag, axis=1))
    
    print(*row)

    ds["row"] = np.array(row).reshape(-1, 1)
    ds["col"] = np.array(col).reshape(-1, 1)
    ds["sym"] = np.array(sym).reshape(-1, 1)
    ds["diag"] = np.array(diag).reshape(-1, 1)
    
    return ds