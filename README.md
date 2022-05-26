# Change-point detection by the SSA method
Change-point detection by the SSA method


# Installation (Windows)

* Create new virtual enviroment in repo folder: `python -m venv {name}`. For example: `python -m venv env`.
* Activate it: `{name}\Scripts\activate`. For example: `env\Scripts\activate`.
* Install all dependencies: `pip install -r requirements.txt`.
* Launch it: `jupyter lab`.

# Files desctiption

## `utils`
* `utils/hmatr.py` - содержит класс матрицы разладки. Базовая имплементация переписана из пакета RSSA. Добавлен функционал для достроения матрицы разладки (итеративно, при появлении новых значений ряда), сохраняя вычисления на предыдущем шаге.
* `utils/hmatrStatistics.py` - копия класса из `hmatr.py` с дополнительными методами для исследований, связанных со строковой функции обнаружения неоднородности.
* `utils/modelling.py` - численные эксперименты для сравнений качества функций неоднородности.
* `utils/coop_modelling.py` - частично повторяет функционал из `utils/modelling.py`, за исключением функции `modelling_series_statistics` - шум добавляется к ряду на каждой итерации моделирования и статистики считаются совместно. В данной версии функции в файле `utils/modelling.py`, статистики посчитаны заранее и от реализации шума не зависят.
* `utils/utils.py` - дополнительные функции.

## `tables`
```Таблицы с результатами тестирований, логи.``` 

## `System`
* `thresh.py` - классы порога срабатывания системы.
* `main.py` - содержит класс системы и пример использования. Также есть код для тестирования и отображения графиков.
* `images.ipynb` - тетрадка со всеми изображениями главы 5.


## `SystemSimplified`
* `system.py` - система, описанная алгоритмом 5.
* `estimate_freq.ipynb` - код для оценки частоты преиодики временного ряда.
* `system_tests.ipynb` - тестирование системы.

## `root`
* `Analytical.ipynb` - тесты и информация об аналитической аппроксимации строковой функции неоднородности.
* `timeTests.ipynb` - тесты для достроения матрицы неоднородности для данных, поступающих в режиме реального времени.
* `heterFuncTest.ipynb` - сравнение качества функций обнаружения неоднородности.
* `freqTests.ipynb` - тесты функции обнаружения неоднородности при разных частотах ряда. (А также попытки численного описания переходного интервала).



# Key words
```
Heterogeneity, heterogeneity detection, heterogeneity analysis, change-point,  change-point detection, change-point estimation, approximation, 
time series, SSA, RSAA, heterogeneity detection function, heterogeneity index, heterogeneity matrix, trajectory matrix, 
SVD, embedded vectors.
```

```
Разладка, определение разладки, анализ разладки, определение момента изменения структуры ряда, оценка момента изменения структуры ряда,
аппроксимация, временные ряды, анализ сингулярного спектра, метод гусеница, пакет RSSA, функции обнаружения разладки,
индекс неоднородности, матрица неоднородности, траекторная матрица, сингулярное разложение, вектора вложений.
```

# Annotation
```
Временной ряд, управляемый некоторой линейной рекуррентной формулой является однородным. При внешнем воздействии ряд 
подвергается мгновенному возмущению и перестает управляться данной формулой. Однородность ряда нарушается
и возникает проблема изучения этой неоднородности. В данной работе рассматриваются типы неоднородности, заданные
изменением частоты, амплитуды, фазовым сдвигом периодики и выбросом; численно сравниваются 
функции обнаружения неоднородностей; упрощается, аппроксимируется и анализируется поведение строковой функции обнаружения, и,
на основе полученных результатов, проектируется и тестируется система оценки момента возмущения. 
```

```
We shall call a time series homogeneous if it is governed by some linear recurrent relation. Assume that due to an 
outside action a homogeneous time series is being exposed to an instant perturbation; that is, it stops following the 
original formulae and the problem of studying this heterogeneity arises. In this paper heterogeneity types defined by 
change of frequency, amplitude, phase shift of seasonal component of signal and outlier are considered; heterogeneity 
detection functions are compared numerically; the row detection function is simplified, approximated and it's behaviour
is analyzed. Additionally, based on the results, the system of perturbation moment estimation is designed and tested.
```