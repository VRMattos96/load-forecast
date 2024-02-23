
# Energy Load Forecasting 
![Teste](https://s3-us-west-2.amazonaws.com/transmountain-craftcms/images/_1200x630_crop_center-center_82_none/transmission-lines-1030x515.jpg?mtime=1582829191)
## Overview

This project focuses on forecasting the Energy Load in Brazil using a combination of Variational Mode Decomposition (VMD) for time series decomposition and Bidirectional Long Short-Term Memory (Bi-LSTM) neural networks for forecasting. The goal is to provide accurate predictions of energy consumption for better resource planning.

## Requirements

- Python 3.9
- TensorFlow
- Keras
- Matplotlib
- Seaborn
- Pandas
- Numpy
- VMDpy

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/VRMattos96/load-forecast.git
    ```

2. Navigate to the project directory:

    ```bash
    cd energy-load-forecasting
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

The project is organized as following:

load-forecasting/
│
├── 01.data/ # Store your dataset and any other data files
│   ├── raw/ # Raw data files
│   ├── processed/ # Processed data files
├── 02.codes/ # Source code
│   ├── data_processing/ # Code for data preprocessing
│   ├── modelling/ # Implementation of the forecasting model
│   ├── validation/ # Code for model validation
├── 03.results/ # Directory for storing results
├── README.md # Project README file
└── requirements.txt # List of Python dependencies


## Feature Engineering: Variational Mode Decomposition (VMD)

In this project, one of the key feature engineering techniques employed is Variational Mode Decomposition (VMD). VMD is a data-driven approach used for time series decomposition, which aims to decompose a time series signal into a set of oscillatory modes.

### What is Variational Mode Decomposition?

Variational Mode Decomposition is a signal processing technique that decomposes a given time series signal into a set of modes, each representing a different oscillatory pattern within the signal. The decomposition is achieved through an optimization process that minimizes a defined cost function, leading to the extraction of intrinsic mode functions (IMFs).

### How VMD is Applied in this Project

1. **Decomposition of Time Series:**
   - The original time series, representing energy load data, is decomposed into its constituent oscillatory modes using VMD.

2. **Identification of Patterns:**
   - Each mode extracted by VMD corresponds to a specific oscillatory pattern within the time series. These patterns can represent various underlying factors affecting energy load, such as seasonality, trends, and anomalies.

3. **Feature Extraction:**
   - The extracted modes serve as features for the subsequent steps in the modeling process. These features aim to capture the relevant information within the energy load data, providing a basis for more accurate forecasting.

### VMD Equations

#### Objective Function:

The VMD optimization problem is formulated as follows:

\[ \min_{u_k, \omega_k} \sum_{k=1}^{K} \left\| x - \sum_{k=1}^{K} u_k \cos(\omega_k t + \phi_k) \right\|_2^2 + \lambda \sum_{k=1}^{K-1} \| \omega_{k+1} - \omega_k \|_2^2 \]

Where:
- \( x(t) \) is the original time series.
- \( u_k \) and \( \omega_k \) are the amplitude and frequency of the \( k \)-th mode, respectively.
- \( \phi_k \) is the phase of the \( k \)-th mode.
- \( \lambda \) is a regularization parameter.
- \( K \) is the number of modes.

#### Update Rules:

The optimization is typically performed using an iterative scheme. The update rules for \( u_k \) and \( \omega_k \) are given by:

\[ u_k = \frac{\mathcal{H}_\lambda(x - \sum_{j \neq k} u_j \cos(\omega_j t + \phi_j))}{\cos(\omega_k t + \phi_k)} \]
\[ \omega_k = \frac{\sum_{t=1}^{T} t u_k \sin(\omega_k t + \phi_k) + \lambda \sum_{k=1}^{K-1} (\omega_{k+1} - 2\omega_k + \omega_{k-1})}{\sum_{t=1}^{T} t u_k \cos(\omega_k t + \phi_k)} \]

### Summation to Reconstruct Original Signal

It's important to note that the sum of all decomposed nodes, or intrinsic mode functions (IMFs), obtained through VMD reconstruction, reproduces the original signal. Mathematically, if \(x(t)\) is the original time series and \(c_k(t)\) are the individual IMFs obtained through VMD, the reconstruction is given by:

\[ x(t) = \sum_{k=1}^{N} c_k(t) \]

This property ensures that the information contained in the decomposed modes is exhaustive and can be combined to recreate the original energy load signal.

#### Demonstration:

![VMD Process](vmd.png)

### Why VMD?

- VMD is particularly useful in scenarios where a time series exhibits complex and non-linear patterns.
- It allows for the separation of different frequency components, making it suitable for capturing various temporal aspects of energy load data.

By employing VMD as part of the feature engineering process, we aim to enhance the effectiveness of our forecasting model by capturing and utilizing the inherent patterns present in the energy load time series data.


