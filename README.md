# Reinforcement Learning in FX Trading

This is a reinforcement learning approach to trading in the foreign exchange market. The trained algorithm interprets market data and estimates the optimal actions base on the learnt strategy. The details are provided in this [explanation paper]() 

The learning algorithm is the Tensorflow 2 implementation of [Twin Delayed Deep Determinsitic Policy Gradients (TD3), (Fujimoto et al., 2018)](https://arxiv.org/abs/1802.09477). It trades in a forex environment with the goal of trading strategy construction and optimisation of the spreadbetting task (other tasks/instruments, such as CFDs, can also be accommodated through updating the reward function).

The main prupose is to allow the learning algorithm to learn trading the securities the user is interested in and to formulate a systemetic trading strategy for implementation.

## Quick Example
This repository contains a pre-trained model that trades EUR/USD, learnt from historical data (March/2012 - Mid-March/2019). The networks are trained using Tensorflow 2.6.0 and Python 3.8.
The trading result is:
![](samples/Sample_Trained_Model_Performance.png)

The simpliest way the trained model can be used is by opening the [One Step Implementation]() notbook and providing the daily historical EUR/USD price data of the past 55 days or more, including the current day, in a .csv format. E.g.

```
dataset_path = 'EURUSD_price_data.csv'
```

After running all the cells, the output is the estimated optimal current action given by the pre-trained model.

## Usage
Users can either (I) use the sample pre-trained model for trading EUR/USD, (II) train a new model for other security or currency, or (III) train a new model to trade multiple currencies at once. The technical instructions are detailed in the IPython notebook.

Regardless of the usage, the Bid-Ask price data from the user's broker must be provided in locations marked ***(User-input required)***. The Bid-Ask price data is generally expected to be in the .csv format with index and columns as shown:

| Index | Bid Open | Bid High | Bid Low | Bid Close | Ask Open | Ask High | Ask Low | Ask Close |
| :---: | :------: | :------: | :-----: | :-------: | :------: | :------: | :-----: | :-------: |
| Timestamp_0 | ... | ... | ... | ... | ... | ... | ... | ... |
| Timestamp_1 | ... | ... | ... | ... | ... | ... | ... | ... |
| Timestamp_2 | ... | ... | ... | ... | ... | ... | ... | ... |
|     ...     | ... | ... | ... | ... | ... | ... | ... | ... |
| Timestamp_n | ... | ... | ... | ... | ... | ... | ... | ... |

