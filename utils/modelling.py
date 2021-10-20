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


def measureStatistics(func, Q, tail):
        return np.max(func[:(Q-tail)]), np.quantile(func[:(Q-tail)], 0.95)


def modellingNoiseStatistics(dictSeries:dict, iterNum:int, N:int, B:int, T:int, Q:int, L:int, r:int, method:str, destFile:str, vareps:float):
    '''
    Моделирование статистик ряда (95й процентиль и средний максимум) при различных реализациях шума до момента разладки методом Монте-Карло.
    :param dict dictSeries: The dictionary where key is the type of series and value is a series. Example: { 'Permanent': [x_1, ..., x_N] }.
    :param int iterNum: Number of iterations for modelling.
    :param int N: The len of series.
    :param int B: The len of base subseries.
    :param int T: The len of test subseries.
    :param int Q: The point of perturbation.
    :param int L: The window len.
    :param int r: Number of eigen vectors.
    :param str method: SVD method.
    :param str destFile: Name of the file for saving results.
    :param float vareps: Variance of the noise.
    '''

    try:
        wb = openpyxl.load_workbook(filename = destFile)
        sheet = wb['Modelling']
    except FileNotFoundError:
        wb = openpyxl.Workbook()
        sheet = wb.active
        sheet.title = "Modelling"
        sheet = wb['Modelling']

    for num, typeV in enumerate(dictSeries.keys()):
        statsRow = []
        statsCol = []
        statsSym = []
        statsDiag = []
        
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

        sheet.cell(row=num*4 + 1, column=1).value = typeV    
        sheet.cell(row=num*4 + 2, column=1).value = 'meanMax'
        sheet.cell(row=num*4 + 3, column=1).value = '95 procentile'

        insert_cell(sheet=sheet, func_type='row', row_num=num*4 + 1, col_num=2, stats=statsRow)
        insert_cell(sheet=sheet, func_type='col', row_num=num*4 + 1, col_num=3, stats=statsCol)
        insert_cell(sheet=sheet, func_type='sym', row_num=num*4 + 1, col_num=4, stats=statsSym)
        insert_cell(sheet=sheet, func_type='diag', row_num=num*4 + 1, col_num=5, stats=statsDiag)

    wb.save(filename = destFile)

def insert_cell(sheet, func_type, row_num, col_num, stats):
    # Вставляем значения статистик в таблицу
    sheet.cell(row=row_num, column=col_num).value = func_type
    i = 2
    for rec in np.mean(stats, axis=0):
        sheet.cell(row=row_num + i, column=col_num).value = rec
        i += 1

def findOvercomingMeanMax(ser, Q, tail, num, sheet, typeF='row'):
    # Ищем точку преодоления среднего максимума (моделируемая характеристика при реализациях шума)
    maxVal = sheet[typeF][num*4 + 0]
    breakNum = None
    for i in range(Q-tail, len(ser)):
        if round(ser[i], 10) > round(maxVal, 10):
            breakNum = i + tail
            return [breakNum, ser[Q-tail], ser[Q-tail+10], ser[Q-tail+20], ser[Q-tail+30]]
    return [None, None, None, None, None]
        
def findOvercoming95Procentile(ser, Q, tail, num, sheet, typeF='row'):
    # Ищем точку преодоления среднего 95го процентиля (моделируемая характеристика при реализациях шума)
    maxVal = sheet[typeF][num*4 + 1]
    breakNum = None
    for i in range(Q-tail, len(ser)):
        if round(ser[i], 10) > round(maxVal, 10):
            breakNum = i + tail
            return [breakNum, ser[Q-tail], ser[Q-tail+10], ser[Q-tail+20], ser[Q-tail+30]]
    return [None, None, None, None, None]
        
        
def rateOfIncrease(hm, Q, num, sheet, typeInc='meanMax'):
    if typeInc == 'meanMax':
        res = {
            'Row': findOvercomingMeanMax(hm.getRow(), Q, hm.T, num, sheet, 'row'),
            'Col': findOvercomingMeanMax(hm.getCol(), Q, hm.B, num, sheet, 'col'),
            'Sym': findOvercomingMeanMax(hm.getSym(), Q, hm.T, num, sheet, 'sym'),
            'Diag': findOvercomingMeanMax(hm.getDiag(), Q, hm.B + hm.T + 1, num, sheet, 'diag')
        }
    
    if typeInc == '95':
        res = {
            'Row': findOvercoming95Procentile(hm.getRow(), Q, hm.T, num, sheet, 'row'),
            'Col': findOvercoming95Procentile(hm.getCol(), Q, hm.B, num, sheet, 'col'),
            'Sym': findOvercoming95Procentile(hm.getSym(), Q, hm.T, num, sheet, 'sym'),
            'Diag': findOvercoming95Procentile(hm.getDiag(), Q, hm.B + hm.T + 1, num, sheet, 'diag')
        }

    return res



def insertRecord(sheet, num, valueMeanMax, value95, noise=True):

    multipRow = 10
    multipCol = 5

    for j, typeV in enumerate(valueMeanMax.keys()):
        # Insert info cells
        sheet.cell(row=num*multipRow + 2, column=j*multipCol+1).value = typeV
        sheet.cell(row=num*multipRow + 2, column=j*multipCol+2).value = 'meanMax'
        sheet.cell(row=num*multipRow + 2, column=j*multipCol+3).value = '95 procentile'

        if noise:
            sheet.cell(row=num*multipRow + 3, column=j*multipCol+1).value = 'Num Points of overcoming'
        sheet.cell(row=num*multipRow + 4 - int(not noise), column=j*multipCol+1).value = 'Point of overcoming'

        sheet.cell(row=num*multipRow + 5 - int(not noise), column=j*multipCol+1).value = 'X[Q]'
        sheet.cell(row=num*multipRow + 6 - int(not noise), column=j*multipCol+1).value = 'X[Q+10]'
        sheet.cell(row=num*multipRow + 7 - int(not noise), column=j*multipCol+1).value = 'X[Q+20]'
        sheet.cell(row=num*multipRow + 8 - int(not noise), column=j*multipCol+1).value = 'X[Q+30]'

        # Process the functions with meanMax
        i = 3
        for rec in valueMeanMax[typeV]:
            sheet.cell(row=num*multipRow + i, column=j*multipCol+2).value = rec
            i += 1

        # Process the functions with 95 procentile
        i = 3
        for rec in value95[typeV]:
            sheet.cell(row=num*multipRow + i, column=j*multipCol+3).value = rec
            i += 1




def modellingSeriesStatistics(dictSeries:dict, iterNum:int, N:int, B:int, T:int, Q:int, L:int, r:int, method:str, destFile:str, modellingResultsPath:str, title:str, vareps:float):
    '''
    Modelling for series with noise
    :param dict dictSeries: The dictionary where key is the type of series and value is a series. Example: { 'Permanent': [x_1, ..., x_N] }.
    :param int iterNum: Number of iterations for modelling.
    :param int N: The len of original series.
    :param int B: The len of base subseries.
    :param int T: The len of test subseries.
    :param int Q: The point of perturbation.
    :param int L: The window len.
    :param int r: Number of eigen vectors.
    :param str method: SVD method.
    :param str destFile: Name of the file for saving results.
    :param str modellingResultsPath: Name of the file with modelling results.
    :param str title: Sheet title where results will be stored.
    :param float vareps: Variance of the noise.
    '''

    try:
        wb = openpyxl.load_workbook(filename = destFile)
        sheet = wb[title]
    except FileNotFoundError:
        wb = openpyxl.Workbook()
        sheet = wb.active
        sheet.title = title
        sheet = wb[title]
    except KeyError:
        wb = openpyxl.load_workbook(filename = destFile)
        sheet = wb.create_sheet(title=title)

    modellingResults = pd.read_excel(modellingResultsPath, sheet_name='Modelling', engine='openpyxl')

    for num, typeV in enumerate(dictSeries.keys()):

        statsMeanMax = []
        stats95 = []

        for i in range(iterNum):
            eps = np.random.randn(N) * vareps**2
            if typeV == 'Temporary':
                eps[:Q] = eps[:Q]/2

            seriesNoise = dictSeries[typeV] + eps
            hm = Hmatr(seriesNoise, B, T, L, neig=r, svdMethod=method)
            statsMeanMax.append(list(rateOfIncrease(hm, Q, num, modellingResults, 'meanMax').values()))
            stats95.append(list(rateOfIncrease(hm, Q, num, modellingResults, '95').values()))

        statsMeanMax = np.array(statsMeanMax)
        stats95 = np.array(stats95)

        # Process MeanMax
        resMeanMaxArr = []
        # Get mean values of points of overcome of detection functions, num that points and values of [Q, Q+10, Q+20, Q+30]
        for i in range(4):
            resMeanMaxArr.append(get_statistics_for_detection_function_for_series_with_noise(stats=statsMeanMax, col_num=i))
        resMeanMax = {
            'Row': resMeanMaxArr[0],
            'Col': resMeanMaxArr[1],
            'Sym': resMeanMaxArr[2],
            'Diag': resMeanMaxArr[3]
        }

       # Process 95 Procentile
        res95Arr = []
        # Get mean values of points of overcome of detection functions, num that points and values of [Q, Q+10, Q+20, Q+30]
        for i in range(4):
            res95Arr.append(get_statistics_for_detection_function_for_series_with_noise(stats=stats95, col_num=i))
        res95 = {
            'Row': res95Arr[0],
            'Col': res95Arr[1],
            'Sym': res95Arr[2],
            'Diag': res95Arr[3]
        }
        sheet.cell(row=num*10 + 1, column=1).value = typeV
        insertRecord(sheet, num, resMeanMax, res95)

    wb.save(filename = destFile)


def get_statistics_for_detection_function_for_series_with_noise(stats, col_num):
    # Вычленяем из общего набора статистик нужную нам функцию обнаружения и формируем средний результат для генерации таблицы.
    ans = []
    for i in range(5):
        if i==0:
            tmp = stats[:, col_num, i]
            tmp = tmp[tmp != np.array(None)]
            ans.append(len(tmp))
            ans.append(round(np.mean(tmp)))
        else:
            tmp = stats[:, col_num, i]
            tmp = np.mean(tmp[tmp != np.array(None)])
            ans.append(tmp)
    return ans



def fixSeriesStatistics(dictSeries:dict, B:int, T:int, Q:int, L:int, r:int, method:str, destFile:str, modellingResultsPath:str, title:str):
    '''
    Save results (statistics) for series without noise
    :param dict dictSeries: The dictionary where key is the type of series and value is a series. Example: { 'Permanent': [x_1, ..., x_N] }.
    :param int B: The len of base subseries.
    :param int T: The len of test subseries.
    :param int Q: The point of perturbation.
    :param int L: The window len.
    :param int r: Number of eigen vectors.
    :param str method: SVD method.
    :param str destFile: Name of the file for saving results.
    :param str modellingResultsPath: Name of the file with modelling results.
    :param str title: Sheet title where results will be stored.
    '''

    try:
        wb = openpyxl.load_workbook(filename = destFile)
        sheet = wb[title]
    except FileNotFoundError:
        wb = openpyxl.Workbook()
        sheet = wb.active
        sheet.title = title
        sheet = wb[title]
    except KeyError:
        wb = openpyxl.load_workbook(filename = destFile)
        sheet = wb.create_sheet(title=title)

    modellingResults = pd.read_excel(modellingResultsPath, sheet_name='Modelling', engine='openpyxl')

    for num, typeV in enumerate(dictSeries.keys()):
        series = dictSeries[typeV]
        hm = Hmatr(series, B, T, L, neig=r, svdMethod=method)
        
        resMeanMax = rateOfIncrease(hm, Q, num, modellingResults, 'meanMax')
        res95 = rateOfIncrease(hm, Q, num, modellingResults, '95')

        sheet.cell(row=num*10 + 1, column=1).value = typeV
        insertRecord(sheet, num, resMeanMax, res95, False)

    
    wb.save(filename = destFile)
