# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 04/11/2026


### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.

### DATASET
DailyDElhiClimate

### SOFTWARE REQUIRED
Jupyter Notebook

### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Load training and testing datasets
train = pd.read_csv('DailyDelhiClimateTrain.csv')
test = pd.read_csv('DailyDelhiClimateTest.csv')

# Convert date column into datetime format
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

# Sort data according to date
train = train.sort_values('date')
test = test.sort_values('date')

# Select Mean Temperature
X = train['meantemp'].dropna()
N = 1000

plt.rcParams['figure.figsize'] = [12, 6]

# Plot original data
plt.plot(train['date'], X)

plt.title('Original DailyDelhiClimate Data')
plt.xlabel('Date')
plt.ylabel('Mean Temperature')
plt.show()
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plot_acf(X, lags=40, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=40, ax=plt.gca(), method='ywm')
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()
```
<img width="1104" height="717" alt="image" src="https://github.com/user-attachments/assets/f0d15ef1-7abf-4929-b9d7-badfb4a81d73" />

```python
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plot_acf(X_diff, lags=40, ax=plt.gca())
plt.title('Differenced Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X_diff, lags=40, ax=plt.gca(), method='ywm')
plt.title('Differenced Data PACF')

plt.tight_layout()
plt.show()
```
<img width="1111" height="741" alt="image" src="https://github.com/user-attachments/assets/9768c821-e8da-485b-badd-f6a43484b5e9" />

#### SIMULATED ARMA(1,1) PROCESS:

```python
# Fit ARMA(1,1) using ARIMA
arma11_model = ARIMA(X, order=(1, 1, 1)).fit()

# Extract model parameters
params11 = dict(zip(arma11_model.param_names, arma11_model.params))

phi1_arma11 = params11['ar.L1']
theta1_arma11 = params11['ma.L1']

print("ARMA(1,1) Model Parameters")
print("AR coefficient:", phi1_arma11)
print("MA coefficient:", theta1_arma11)
# Define AR and MA coefficients
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

# Generate simulated ARMA(1,1) process
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

# Plot simulated process
plt.figure(figsize=(12, 6))

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlabel('Observations')
plt.ylabel('Values')
plt.xlim([0, 500])
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_acf(ARMA_1, lags=40, ax=plt.gca())
plt.title('ARMA(1,1) ACF')

plt.subplot(1, 2, 2)
plot_pacf(ARMA_1, lags=40, ax=plt.gca(), method='ywm')
plt.title('ARMA(1,1) PACF')

plt.tight_layout()
plt.show()
```
#### SIMULATED ARMA(2,2) PROCESS:<br>

```python
# Fit ARMA(2,2) using ARIMA
arma22_model = ARIMA(X, order=(2, 1, 2)).fit()

# Extract model parameters
params22 = dict(zip(arma22_model.param_names, arma22_model.params))

phi1_arma22 = params22['ar.L1']
phi2_arma22 = params22['ar.L2']

theta1_arma22 = params22['ma.L1']
theta2_arma22 = params22['ma.L2']

print("ARMA(2,2) Model Parameters")
print("AR coefficient 1:", phi1_arma22)
print("AR coefficient 2:", phi2_arma22)
print("MA coefficient 1:", theta1_arma22)
print("MA coefficient 2:", theta2_arma22)

# Fit ARMA(2,2) using ARIMA
arma22_model = ARIMA(X, order=(2, 1, 2)).fit()

# Extract model parameters
params22 = dict(zip(arma22_model.param_names, arma22_model.params))

phi1_arma22 = params22['ar.L1']
phi2_arma22 = params22['ar.L2']

theta1_arma22 = params22['ma.L1']
theta2_arma22 = params22['ma.L2']

print("ARMA(2,2) Model Parameters")
print("AR coefficient 1:", phi1_arma22)
print("AR coefficient 2:", phi2_arma22)
print("MA coefficient 1:", theta1_arma22)
print("MA coefficient 2:", theta2_arma22)
# Define AR and MA coefficients
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

# Generate simulated ARMA(2,2) process
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

# Plot simulated process
plt.figure(figsize=(12, 6))

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlabel('Observations')
plt.ylabel('Values')
plt.xlim([0, 500])
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_acf(ARMA_2, lags=40, ax=plt.gca())
plt.title('ARMA(2,2) ACF')

plt.subplot(1, 2, 2)
plot_pacf(ARMA_2, lags=40, ax=plt.gca(), method='ywm')
plt.title('ARMA(2,2) PACF')

plt.tight_layout()
plt.show()
```

<br>

```python
# Calculate RMSE and MAE
rmse11 = np.sqrt(mean_squared_error(y_test, forecast11))
rmse22 = np.sqrt(mean_squared_error(y_test, forecast22))

mae11 = mean_absolute_error(y_test, forecast11)
mae22 = mean_absolute_error(y_test, forecast22)

print("ARMA(1,1) Performance")
print("RMSE:", rmse11)
print("MAE:", mae11)

print("\nARMA(2,2) Performance")
print("RMSE:", rmse22)
print("MAE:", mae22)
```

### OUTPUT:
#### SIMULATED ARMA(1,1) PROCESS:



##### Partial Autocorrelation

##### Autocorrelation
<img width="569" height="454" alt="image" src="https://github.com/user-attachments/assets/ba2e65f4-169e-4381-a56b-51d97cb2feaf" />



#### SIMULATED ARMA(2,2) PROCESS:
<img width="569" height="454" alt="image" src="https://github.com/user-attachments/assets/2d4a3225-e735-4567-9986-4d668f534cce" />

##### Partial Autocorrelation
<img width="555" height="450" alt="image" src="https://github.com/user-attachments/assets/5aeebac0-a33d-4f77-bdb6-4fbf7e030064" />


##### Autocorrelation
<img width="555" height="450" alt="image" src="https://github.com/user-attachments/assets/81b06286-2077-4639-80f0-e9a467769471" />

### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
