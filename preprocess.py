# Extracting price indicators for building an agent-observable dataset
import numpy as np
import pandas as pd


def ma(df, ma_ranges=[10, 21, 50]):
    """
    Simple Moving Average

    Parameters
    ----------
    df : pandas.DataFrame, must include columns ['Close']
     Dataframe where the ma is extracted from

    ma_ranges: list, default [10, 21, 50]
     List of periods of Simple Moving Average to be extracted

    Return
    ------
    df : pandas.DataFrame
     DataFrame with Simple Moving Average
    """

    df = df.copy()
    for period in ma_ranges:
        df[f"MA{period}"] = df['Close'].rolling(window=period).mean()

    return df


def macd(df, short_long_signals=[(12, 26, 9)]):
    """
    Moving Average Convergence Divergence

    Parameters
    ----------
    df : pandas.DataFrame, must include columns ['Close']
     Dataframe where the macd is extracted from

    short_long_signals : list, default [(12, 26, 9)]
     List of periods of (short_ema, long_ema, signal) of Moving Average Convergence Divergence to be extracted

    Return
    ------
    df : pandas.DataFrame
     Dataframe with Moving Average Convergence Divergence
    """

    df = df.copy()
    for (short, long, signal) in short_long_signals:
        df[f"EMA{short}"] = df["Close"].ewm(span=short, adjust=False).mean()
        df[f"EMA{long}"] = df["Close"].ewm(span=long, adjust=False).mean()
        df[f"MACD{long}"] = df[f"EMA{short}"] - df[f"EMA{long}"]
        df[f"MACD{long}Signal"] = df[f"MACD{long}"].ewm(span=signal, adjust=False).mean()
        df = df.drop(columns=[f"EMA{short}", f"EMA{long}"])

    return df


def full_stochastic(df, stochastic_ranges=[(14, 3, 3)]):
    """
    Full Stochastic Indicator

    Parameters
    ----------
    df : pandas.DataFrame, must include columns ['High', 'Low', 'Close']
     Dataframe where the full_stochastic is extracted from

    stochastic_ranges : list, default [(14, 3, 3)]
     List of periods of (fast_k, fast_d, slow_d) of Full Stochastic Indicator to be extracted

    Return
    ------
    df : pandas.DataFrame
     Dataframe with Full Stochastic Indicators
    """

    df = df.copy()
    for (fast_k, fast_d, slow_d) in stochastic_ranges:
        df[f"L{fast_k}"] = df["Low"].rolling(window=fast_k).min()
        df[f"H{fast_k}"] = df["High"].rolling(window=fast_k).max()
        df[f"fast_%K{fast_k}"] = 100 * ((df["Close"] - df[f"L{fast_k}"])
                                        / (df[f"H{fast_k}"] - df[f"L{fast_k}"]))
        df[f"full_%K{fast_k}_fast_%D{fast_d}"] = df[f"fast_%K{fast_k}"].rolling(window=fast_d).mean()
        df[f"full_%K{fast_k}_slow_%D{slow_d}"] = df[f"full_%K{fast_k}_fast_%D{fast_d}"].rolling(window=slow_d).mean()
        df = df.drop(columns=[f"L{fast_k}", f"H{fast_k}", f"fast_%K{fast_k}"])

    return df


def rsi(df, rsi_ranges=[14]):
    """
    Relative Strength Index

    Parameters
    ----------
    df : pandas.DataFrame, must include columns ['Open', 'Close]
     Dataframe where the rsi is extracted from

    rsi_ranges: list, default [14]
     List of periods of rsi to be extracted

    Return
    ------
    df : pandas.DataFrame
     Dataframe with Relative Strength Index
    """
    
    df = df.copy()
    df["Up_Close"] = np.where((df["Close"] > df["Open"]), df["Close"], 0)
    df["Down_Close"] = np.where((df["Close"] < df["Open"]), df["Close"], 0)
    for period in rsi_ranges:
        df[f"RS{period}_RollUpAvg"] = df["Up_Close"].ewm(span=period, adjust=False).mean()
        df[f"RS{period}_RollDownAvg"] = df["Down_Close"].ewm(span=period, adjust=False).mean()
        df[f"RSI{period}"] = 100 - (100 / (1 + (df[f"RS{period}_RollUpAvg"] / df[f"RS{period}_RollDownAvg"])))
        df = df.drop(columns=[f"RS{period}_RollUpAvg", f"RS{period}_RollDownAvg"])
    df = df.drop(columns=["Up_Close", "Down_Close"])

    return df


def bollinger_bands(df, bollinger_period_sd_ranges=[(20, 2)]):
    """
    Bollinger Bands (including %B and Bandwidth)

    Parameters
    ----------
    df : pandas.DataFrame, must include columns ['High', 'Low', 'Close]
     Dataframe where the bollinger_bands is extracted from

    bollinger_period_sd_ranges : list, default [(20,2)]
     List of (period, standard_deviation) to be extracted

    Return
    ------
    df : pandas.DataFrame
     Dataframe with Bollinger Bands
    """
    
    df = df.copy()
    df["TypicalPrice"] = ((df["High"] + df["Low"] + df["Close"]) / 3)
    for (period, sd) in bollinger_period_sd_ranges:
        ma_typical_price = df["TypicalPrice"].rolling(window=period).mean()
        sd_typical_price = ma_typical_price.rolling(window=period).std(ddof=1)
        df[f"UpperBollinger{period}"] = ma_typical_price + sd * sd_typical_price
        df[f"LowerBollinger{period}"] = ma_typical_price - sd * sd_typical_price
        df[f"%B{period}"] = ((df["Close"] - df[f"LowerBollinger{period}"]) /
                             (df[f"UpperBollinger{period}"] - df[f"LowerBollinger{period}"]))
        df[f"Bandwidth{period}"] = (df[f"UpperBollinger{period}"] - df[f"LowerBollinger{period}"]) / ma_typical_price
    df = df.drop(columns=[f"TypicalPrice"])

    return df


def add_lag(df, lag=5):
    """
    Add lags to dataset to provide historical context

    Parameters
    -----------
    df : pandas.DataFrame
     Dataframe to add lags to

    lag: int, default 5
     Number of lags to be added

    Return
    ------
    df : pandas.DataFrame
     Dataframe with lags
    """

    df = df.copy()
    cols = list(df.columns)
    cols_len = len(cols)
    for i in range(1, lag + 1):
        df = pd.concat([df, df.iloc[:, :cols_len].shift(i)], axis=1)
        cols.extend([x + f"n-{i}" for x in df.columns[:cols_len]])
        df.columns = cols

    return df


def remove_outlier(df, one_side_remove_percentile=0.005, train_portion=0.8):
    """
    Remove the outliers of each column of the dataset

    Parameters
    ----------
    df : pandas.DataFrame
     Dataframe to remove outliers from

    one_side_remove_percentile : float, default 0.005
     The percentile of outliers to be removed on each end

    train_portion : float, default 0.8
     The percentage of dataset used for training

    Returns
    --------
    df : pandas.DataFrame
     Processed dataframe

    mins : numpy.array
     Min values of each columns

    maxs : numpy.array
     Max values of each columns
    """

    df = df.copy()
    mins, maxs = [], []
    df_cols = df.columns
    df_ind = df.index
    df = np.array(df)
    for i in range(df.shape[1]):
        temp = df[:, i]
        mins.append(np.sort(temp[:int(df.shape[0] * train_portion)])[int(df.shape[0] * one_side_remove_percentile)])
        maxs.append(np.sort(temp[:int(df.shape[0] * train_portion)])[-int(df.shape[0] * one_side_remove_percentile)])
    df = np.clip(df, mins, maxs)
    df = pd.DataFrame(df, columns=df_cols, index=df_ind)

    return df, np.array(mins), np.array(maxs)


def minmaxscaler(df, mins, maxs):
    """
    Rescale the data of the dataset using MinMaxScaler given the min and max values of each column

    Parameters
    ----------
    df : pandas.DataFrame
     Dataframe to be rescaled

    mins : numpy.array
     Min values of each columns

    maxs : numpy.array
     Max values of each columns

    Return
    -------
    df : pandas.DataFrame
     Rescaled dataframe
    """

    df = df.copy()
    df = (df - mins) / (maxs - mins)

    return df


def build_agent_obs_dataset(unprocessed_df,
                            rename_columns=['Open', 'High', 'Low', 'Close', 'AskOpen', 'AskHigh', 'AskLow', 'AskClose'],
                            ma_ranges=[10, 21, 50],
                            short_long_signals=[(12, 26, 9)],
                            stochastic_ranges=[(14, 3, 3)],
                            rsi_ranges=[14],
                            bollinger_period_sd_ranges=[(20, 2)],
                            lag=5,
                            one_side_remove_percentile=0.005,
                            train_portion=0.8):
    """
    Build an agent-observable dataset that represent states the agent encounter at each timestamp.
    The default parameters are in reference to the ones used in the explanation paper

    Parameters
    ----------
    unprocessed_df : pandas.DataFrame
     Dataframe where the price indicators are extracted from the dataframe must contains candlestick data.
     e.g. columns = ['BidOpen', 'BidHigh', 'BidLow', 'BidClose', 'AskOpen', 'AskHigh', 'AskLow', 'AskClose']

    rename_columns : list, default ['Open', 'High', 'Low', 'Close', 'AskOpen', 'AskHigh', 'AskLow', 'AskClose']
     (Default price indicators are extracted from bid prices)
     Rename columns to include ['Open', 'High', 'Low', 'Close'] for price indicators extraction. If they already exist,
     simply pass on the unprocessed dataframe columns.

    ma_ranges : list, default [10, 21, 50]
     List of periods of simple moving average to be extracted

    short_long_signals : list, default [(12, 26, 9)]
     List of periods (short_ema, long_ema, signal) of Moving Average Convergence Divergence to be extracted

    stochastic_ranges : list, default [(14, 3, 3)]
     List of periods (fast_k, fast_d, slow_d) of Full Stochastic Indicator to be extracted

    rsi_ranges : list, default [14]
     List of periods of Relative Strength Index to be extracted

    bollinger_period_sd_ranges : list, default [(20,2)]
     List of (period, standard_deviation) of Bollinger Bands (including %B and Bandwidth) to be extracted

    lag : int, default 5
     Number of lags added to the dataset to provide historical context

    one_side_remove_percentile : float, default 0.005
     The percentile of outliers to be removed on each end

    train_portion : float, default 0.8
     The percentage of dataset used for training

    Returns
    -------
    train_agent_obs_data : pandas.DataFrame
     Agent-observable dataset for training

    val_agent_obs_data : pandas.DataFrame
     Agent-observable dataset for validation
    """

    agent_obs_data = unprocessed_df.copy()
    agent_obs_data.columns = rename_columns
    agent_obs_data = ma(df=agent_obs_data, ma_ranges=ma_ranges)
    agent_obs_data = macd(df=agent_obs_data, short_long_signals=short_long_signals)
    agent_obs_data = full_stochastic(df=agent_obs_data, stochastic_ranges=stochastic_ranges)
    agent_obs_data = rsi(df=agent_obs_data, rsi_ranges=rsi_ranges)
    agent_obs_data = bollinger_bands(df=agent_obs_data, bollinger_period_sd_ranges=bollinger_period_sd_ranges)
    agent_obs_data = agent_obs_data.iloc[:, len(rename_columns):]
    agent_obs_data = add_lag(df=agent_obs_data, lag=lag)
    agent_obs_data = agent_obs_data.dropna()

    agent_obs_data, mins, maxs = remove_outlier(agent_obs_data, one_side_remove_percentile=one_side_remove_percentile,
                                                train_portion=train_portion)

    # We allow Upper and Lower BollingerBands to share the same mins and maxs
    mins[np.where(['UpperBollinger' in x for x in list(agent_obs_data.columns)])[0]] = mins[
        np.where(['LowerBollinger' in x for x in list(agent_obs_data.columns)])[0]]
    maxs[np.where(['LowerBollinger' in x for x in list(agent_obs_data.columns)])[0]] = maxs[
        np.where(['UpperBollinger' in x for x in list(agent_obs_data.columns)])[0]]

    # We do not specify train_portion here as the mins and maxs are extracted from the training set
    agent_obs_data = minmaxscaler(agent_obs_data, mins, maxs)

    train_agent_obs_data = agent_obs_data.iloc[:int(agent_obs_data.shape[0] * train_portion)]
    val_agent_obs_data = agent_obs_data.iloc[int(agent_obs_data.shape[0] * train_portion):]

    return train_agent_obs_data, val_agent_obs_data, mins, maxs
