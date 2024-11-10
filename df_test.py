import time
import threading
import pyupbit
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import requests
import ta

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("discord_webhhok")
upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS"), os.getenv("UPBIT_SECRET"))

df_tickers = {}    # 전역변수:일봉 데이터프레임
count=50
t='KRW-CARV'
def load_ohlcv(ticker):
    global df_tickers
    if ticker not in df_tickers:   # 티커가 캐시에 없으면 데이터 가져오기     
        try:
            # df_tickers[ticker] = pyupbit.get_ohlcv(ticker, interval="minute15", count=200) 
            df_tickers[ticker]= pyupbit.get_ohlcv(t, interval="minute15", count=count) 
            # df_tickers[ticker]= pyupbit.get_ohlcv(t, interval="minute60", count=count) 
            

            if df_tickers[ticker] is None or df_tickers[ticker].empty:
                print(f"load_ohlcv / No data returned for ticker: {ticker}")
                time.sleep(0.1)  # API 호출 제한을 위한 대기

        except Exception as e:
            print(f"load_ohlcv / 디스코드 메시지 전송 실패 : {e}")
            time.sleep(1)
    return df_tickers.get(ticker)

def get_rsi(ticker, period):
    # df_rsi = pyupbit.get_ohlcv(ticker, interval="minute5", count=period)
    df_rsi = load_ohlcv(ticker)
    # df_rsi = pyupbit.get_ohlcv(ticker, interval="day", count=15)
    delta = df_rsi['close'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_ta_rsi(ticker, period):
    # 데이터 가져오기
    df_rsi = load_ohlcv(ticker)
    if df_rsi is None or df_rsi.empty:
        return None  # 데이터가 없으면 None 반환

    # TA 라이브러리를 사용하여 RSI 계산
    rsi = ta.momentum.RSIIndicator(df_rsi['close'], window=period).rsi()

    return rsi if not rsi.empty else None  # 마지막 RSI 값 반환

def get_rsi_and_stoch_rsi(ticker, rsi_period, stoch_period):

    df_rsi = load_ohlcv(ticker)
    # df_rsi = pyupbit.get_ohlcv(ticker, interval="minute60", count=20) 
    
    # RSI 계산
    delta = df_rsi['close'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 스토캐스틱 RSI 계산
    min_rsi = rsi.rolling(window=stoch_period).min()
    max_rsi = rsi.rolling(window=stoch_period).max()
    
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan)  # 0으로 나누는 경우를 방지하기 위해 np.nan으로 대체
    
    # return rsi.iloc[-1], stoch_rsi.iloc[-1] if not stoch_rsi.empty else 0
    return stoch_rsi if not stoch_rsi.empty else 0
    
def ta_stochastic_rsi(ticker):
    # 데이터 가져오기
    df = load_ohlcv(ticker)
    if df is None or df.empty:
        return None  # 데이터가 없으면 None 반환
    # RSI 계산
    rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # Stochastic RSI 계산
    min_rsi = rsi.rolling(window=14).min()
    max_rsi = rsi.rolling(window=14).max()

    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)
    stoch_rsi = stoch_rsi.replace([np.inf, -np.inf], np.nan)  # 무한대를 np.nan으로 대체

    # Stochastic RSI 값만 반환 
    return stoch_rsi if not stoch_rsi.empty else 0


df_day = pyupbit.get_ohlcv(t, interval="day", count=7)  

rsi = get_rsi(t, 14)
ta_rsi = get_ta_rsi(t, 14)

pre_rsi = rsi.iloc[-2]
pre_ta_rsi = ta_rsi.iloc[-2]
last_rsi = rsi.iloc[-1]
last_ta_rsi = ta_rsi.iloc[-1]
# print(pre_rsi)
# print(pre_ta_rsi)
# print(last_rsi)
# print(last_ta_rsi)
            
stoch_rsi = get_rsi_and_stoch_rsi(t, 14, 14)   #스토캐스틱 RSI 계산
ta_s_rsi = ta_stochastic_rsi(t)
last_srsi = stoch_rsi.iloc[-1]
last_ta_s_rsi = ta_s_rsi.iloc[-1]

print(f"[{t}] RSI:{last_rsi} / {last_ta_rsi}\n[{t}] sRSI:{last_srsi} / {last_ta_s_rsi}")
# print(last_ta_s_rsi)
# print(ta_s_rsi)

# last_stoch_rsi = stoch_rsi.iloc[-1]
# print(last_stoch_rsi)
# previous_stoch_rsi = stoch_rsi.iloc[-2]

# # print(f"[{t}] rsi:{pre_rsi:,.2f} < {last_rsi:,.2f} \n[{t}] s_rsi: {previous_stoch_rsi:,.2f} < {last_stoch_rsi:,.2f}")


# def get_ema(ticker, window):
#     # df = load_ohlcv(ticker)
#     df = pyupbit.get_ohlcv(ticker, interval="minute15", count=200)

#     if df is not None and not df.empty:
#         return df['close'].ewm(span=window, adjust=False).mean()  # EMA 계산 후 마지막 값 반환
    
#     else:
#         return 0  # 데이터가 없으면 0 반환

# # print(df)
# # print(df['close'].iloc[-1])

# # day_value_1 = df_day['value'].iloc[-1] 
# # day_value_2 = df_day['value'].iloc[-2] 
# # print(f"[{t}] value1: {day_value_1:,} \n[{t}] value2: {day_value_2:,}")

# # last_ema200 = get_ema(t, 200).iloc[-1]    #200봉 지수이동평균 계산
# # pre_ema200 = get_ema(t, 200).iloc[-2]