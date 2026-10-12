def calculate_synchronized_macd(df, fast=8, slow=17, signal=9):
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    df["Signal_Hist_Turn"] = (
        (df["MACD_Hist"] < 0)
        & (df["MACD_Hist"] > df["MACD_Hist"].shift(1))
        & (df["MACD_Hist"].shift(1) < df["MACD_Hist"].shift(2))
    )
    return df