# Combinatorial Component Day-Ahead Load Forecasting through Unanchored Time Series Chain Evaluation
Day-Ahead Load Forecasting model introducing a combinatorial decomposition method and a pattern conservation quality evaluation method.

Research project within the scope of my postdoctoral studies on "Hybridization of Forecasting Models in the Energy Sector Towards Structural Transparency and Resilient Estimation"
This project introduces:
- A novel combinatorial decomposition method merging STL, SSA and EMD under an importance-based XGBoost feature selector
- A novel evolutionary pattern evaluation method based on unanchored time series chains for daily and weekly chain conservation
This work was tested on several different types of estimators including LR, XGBoost, DNN, LSTM, Attention-LSTM, denoting substantial improvements on the performance the neural network structrures.
The experiments and evaluation of the proposed model were carried out through a case study on the Greek power system utilizing electricity load and renewable generation sequences.
The dataset utilized in this work is publicly available on: https://doi.org/10.25832/time_series/2020-10-06
