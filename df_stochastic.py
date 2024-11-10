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

def ta_stochastic(ticker, window=14):
    # 데이터 가져오기
    df = load_ohlcv(ticker)
    if df is None or df.empty:
        return None  # 데이터가 없으면 None 반환

    # Stochastic 계산
    high = df['high'].rolling(window=window).max()  # 지정된 기간의 최고가
    low = df['low'].rolling(window=window).min()    # 지정된 기간의 최저가
    current_close = df['close']                     # 현재 종가

    # Stochastic %K 계산
    stoch_k = (current_close - low) / (high - low)
    stoch_k = stoch_k.replace([np.inf, -np.inf], np.nan)  # 무한대를 np.nan으로 대체

    # Stochastic %D 계산 (스무딩)
    stoch_d = stoch_k.rolling(window=3).mean()  # %K의 3일 이동 평균

    return stoch_k, stoch_d  # Stochastic %K와 %D 반환

stoch_k, stoch_d = ta_stochastic(t)

latest_stoch_k = stoch_k.iloc[-1]
latest_stoch_d = stoch_d.iloc[-1]

    # 매수 신호
if latest_stoch_k < 0.2 and latest_stoch_k > latest_stoch_d:
    print(f"{t} 매수")

    