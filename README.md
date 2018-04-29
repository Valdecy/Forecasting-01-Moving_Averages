# Forecasting-01-Moving_Averages

A total of 3 functions are given to calculate the Centered Moving Average of a time series: Weighted Moving Average, Simple Moving Average and Exponential Moving Average.

A) Weighted Moving Average

* timeseries = The dataset in a Time Series format.

* w = A list of weigths. The default values are [0.2, 0.3, 0.5].

* graph = If True then the original dataset and the moving average curves will be plotted. The default value is True.

* horizon = Calculates the prediction h steps ahead. The default value is 0.

B) Simple Moving Average

* timeseries = The dataset in a Time Series format.

* n = Indicates the considered number of periods to calculate the moving average. The default value is 2.

* graph = If True then the original dataset and the moving average curves will be plotted. The default value is True.

* horizon = Calculates the prediction h steps ahead. The default value is 0.

C) Exponential Moving Average

* timeseries = The dataset in a Time Series format.

* alpha =  Exponential smoothing parameter. The default value is 0.5.

* graph = If True then the original dataset and the moving average curves will be plotted. The default value is True.

* horizon = Calculates the prediction h steps ahead. The default value is 0.

* optimize = If True then best "alpha" (exponential smoothing parameter) is calculated by brute force. The default value is False.
