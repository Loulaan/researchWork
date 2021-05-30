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


        sheet.cell(row=num*4 + 1, column=2).value = 'row'
        i = 2
        for rec in np.mean(statsRow, axis=0):
            sheet.cell(row=num*4 + i, column=2).value = rec
            i += 1


        sheet.cell(row=num*4 + 1, column=3).value = 'col'
        i = 2
        for rec in np.mean(statsCol, axis=0):
            sheet.cell(row=num*4 + i, column=3).value = rec
            i += 1


        sheet.cell(row=num*4 + 1, column=4).value = 'sym'
        i = 2
        for rec in np.mean(statsSym, axis=0):
            sheet.cell(row=num*4 + i, column=4).value = rec
            i += 1


        sheet.cell(row=num*4 + 1, column=5).value = 'diag'
        i = 2
        for rec in np.mean(statsDiag, axis=0):
            sheet.cell(row=num*4 + i, column=5).value = rec
            i += 1


    wb.save(filename = destFile)



def findOvercomingMeanMax(ser, Q, tail, num, sheet, typeF='row', noise=True):
    if noise:
        maxVal = sheet[typeF][num*4 + 0]
    else:
        maxVal = 1e-10
    breakNum = None
    for i in range(Q-tail, len(ser)):
        if round(ser[i], 10) > round(maxVal, 10):
            breakNum = i + tail
            return [breakNum, ser[Q-tail], ser[Q-tail+10], ser[Q-tail+20], ser[Q-tail+30]]
    return [None, None, None, None, None]
        
def findOvercoming95Procentile(ser, Q, tail, num, sheet, typeF='row', noise=True):
    if noise:
        maxVal = sheet[typeF][num*4 + 1]
    else:
        maxVal = 1e-10
    breakNum = None
    for i in range(Q-tail, len(ser)):
        if round(ser[i], 10) > round(maxVal, 10):
            breakNum = i + tail
            return [breakNum, ser[Q-tail], ser[Q-tail+10], ser[Q-tail+20], ser[Q-tail+30]]
    return [None, None, None, None, None]
        
        
def rateOfIncrease(hm, Q, num, sheet, typeInc='meanMax', noise=True):
    if typeInc == 'meanMax':
        res = {
            'Row': findOvercomingMeanMax(hm.getRow(), Q, hm.T, num, sheet, 'row', noise),
            'Col': findOvercomingMeanMax(hm.getCol(), Q, hm.B, num, sheet, 'col', noise),
            'Sym': findOvercomingMeanMax(hm.getSym(), Q, hm.T, num, sheet, 'sym', noise),
            'Diag': findOvercomingMeanMax(hm.getDiag(), Q, hm.B + hm.T + 1, num, sheet, 'diag', noise)
        }
    
    if typeInc == '95':
        res = {
            'Row': findOvercoming95Procentile(hm.getRow(), Q, hm.T, num, sheet, 'row', noise),
            'Col': findOvercoming95Procentile(hm.getCol(), Q, hm.B, num, sheet, 'col', noise),
            'Sym': findOvercoming95Procentile(hm.getSym(), Q, hm.T, num, sheet, 'sym', noise),
            'Diag': findOvercoming95Procentile(hm.getDiag(), Q, hm.B + hm.T + 1, num, sheet, 'diag', noise)
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
    :param int N: The len of series.
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

    # TODO: optimize homogeneous operations

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
        
        # Get mean values of row function point of overcome, num that points and values of [Q, Q+10, Q+20, Q+30] points
        valuesMeanMaxRow = []
        for i in range(5):
            if i==0:
                tmp = statsMeanMax[:, 0, i]
                tmp = tmp[tmp != np.array(None)]
                valuesMeanMaxRow.append(len(tmp))
                valuesMeanMaxRow.append(round(np.mean(tmp)))
            else:
                tmp = statsMeanMax[:, 0, i]
                tmp = np.mean(tmp[tmp != np.array(None)])
                valuesMeanMaxRow.append(tmp)


        # Get mean values of col function point of overcome, num that points and values of [Q, Q+10, Q+20, Q+30] points
        valuesMeanMaxCol = []
        for i in range(5):
            if i==0:
                tmp = statsMeanMax[:, 1, i]
                tmp = tmp[tmp != np.array(None)]
                valuesMeanMaxCol.append(len(tmp))
                valuesMeanMaxCol.append(round(np.mean(tmp)))
            else:
                tmp = statsMeanMax[:, 1, i]
                tmp = np.mean(tmp[tmp != np.array(None)])
                valuesMeanMaxCol.append(tmp)

        # Get mean values of sym function point of overcome, num that points and values of [Q, Q+10, Q+20, Q+30] points
        valuesMeanMaxSym = []
        for i in range(5):
            if i==0:
                tmp = statsMeanMax[:, 2, i]
                tmp = tmp[tmp != np.array(None)]
                valuesMeanMaxSym.append(len(tmp))
                valuesMeanMaxSym.append(round(np.mean(tmp)))
            else:
                tmp = statsMeanMax[:, 2, i]
                tmp = np.mean(tmp[tmp != np.array(None)])
                valuesMeanMaxSym.append(tmp)

        # Get mean values of diag function point of overcome, num that points and values of [Q, Q+10, Q+20, Q+30] points
        valuesMeanMaxDiag = []
        for i in range(5):
            if i==0:
                tmp = statsMeanMax[:, 3, i]
                tmp = tmp[tmp != np.array(None)]
                valuesMeanMaxDiag.append(len(tmp))
                valuesMeanMaxDiag.append(round(np.mean(tmp)))
            else:
                tmp = statsMeanMax[:, 3, i]
                tmp = np.mean(tmp[tmp != np.array(None)])
                valuesMeanMaxDiag.append(tmp)

        resMeanMax = {
            'Row': valuesMeanMaxRow,
            'Col': valuesMeanMaxCol,
            'Sym': valuesMeanMaxSym,
            'Diag': valuesMeanMaxDiag
        }

       # Process 95 Procentile
       # Get mean values of row function point of overcome, num that points and values of [Q, Q+10, Q+20, Q+30] points
        values95Row = []
        for i in range(5):
            if i==0:
                tmp = stats95[:, 0, i]
                tmp = tmp[tmp != np.array(None)]
                values95Row.append(len(tmp))
                values95Row.append(round(np.mean(tmp)))
            else:
                tmp = stats95[:, 0, i]
                tmp = np.mean(tmp[tmp != np.array(None)])
                values95Row.append(tmp)


        # Get mean values of col function point of overcome, num that points and values of [Q, Q+10, Q+20, Q+30] points
        values95Col = []
        for i in range(5):
            if i==0:
                tmp = stats95[:, 1, i]
                tmp = tmp[tmp != np.array(None)]
                values95Col.append(len(tmp))
                values95Col.append(round(np.mean(tmp)))
            else:
                tmp = stats95[:, 1, i]
                tmp = np.mean(tmp[tmp != np.array(None)])
                values95Col.append(tmp)

        # Get mean values of sym function point of overcome, num that points and values of [Q, Q+10, Q+20, Q+30] points
        values95Sym = []
        for i in range(5):
            if i==0:
                tmp = stats95[:, 2, i]
                tmp = tmp[tmp != np.array(None)]
                values95Sym.append(len(tmp))
                values95Sym.append(round(np.mean(tmp)))
            else:
                tmp = stats95[:, 2, i]
                tmp = np.mean(tmp[tmp != np.array(None)])
                values95Sym.append(tmp)

        # Get mean values of diag function point of overcome, num that points and values of [Q, Q+10, Q+20, Q+30] points
        values95Diag = []
        for i in range(5):
            if i==0:
                tmp = stats95[:, 3, i]
                tmp = tmp[tmp != np.array(None)]
                values95Diag.append(len(tmp))
                values95Diag.append(round(np.mean(tmp)))
            else:
                tmp = stats95[:, 3, i]
                tmp = np.mean(tmp[tmp != np.array(None)])
                values95Diag.append(tmp)


        res95 = {
            'Row': values95Row,
            'Col': values95Col,
            'Sym': values95Sym,
            'Diag': values95Diag
        }


        sheet.cell(row=num*9 + 1, column=1).value = typeV

        insertRecord(sheet, num, resMeanMax, res95)


    wb.save(filename = destFile)



def fixSeriesStatistics(dictSeries:dict, N:int, B:int, T:int, Q:int, L:int, r:int, method:str, destFile:str, modellingResultsPath:str, title:str):
    '''
    Save results for series without noise
    :param dict dictSeries: The dictionary where key is the type of series and value is a series. Example: { 'Permanent': [x_1, ..., x_N] }.
    :param int N: The len of series.
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

    for num, typeV in enumerate(dictSeries.keys()):
        series = dictSeries[typeV]
        hm = Hmatr(series, B, T, L, neig=r, svdMethod=method)
        
        resMeanMax = rateOfIncrease(hm, Q, num, modellingResultsPath, 'meanMax', False)
        res95 = rateOfIncrease(hm, Q, num, modellingResultsPath, '95', False)

        sheet.cell(row=num*10 + 1, column=1).value = typeV
        insertRecord(sheet, num, resMeanMax, res95, False)

    
    wb.save(filename = destFile)
