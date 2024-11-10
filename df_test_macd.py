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

def calculate_macd(ticker):
    # 데이터 가져오기
    df = load_ohlcv(ticker)
    if df is None or df.empty:
        return None  # 데이터가 없으면 None 반환

    # MACD 계산
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)

    # MACD 및 시그널선 계산
    df['MACD'] = macd.macd()
    df['Signal'] = macd.macd_signal()

    return df[['MACD', 'Signal']]  # MACD와 시그널선 반환

# MACD 계산
macd_df = calculate_macd(t)
# if macd_df is None:
#     return None  # 데이터가 없으면 None 반환

    # 마지막 두 값 비교
last_macd = macd_df['MACD'].iloc[-1]
last_signal = macd_df['Signal'].iloc[-1]
previous_macd = macd_df['MACD'].iloc[-2]
previous_signal = macd_df['Signal'].iloc[-2]

    # MACD가 시그널선을 상향 돌파했는지 확인
print(f"[{t}] macd1: {previous_macd:,.2f} < {previous_signal:,.2f} /macd2: {last_macd:,.2f} > {last_signal:,.2f}" )
if previous_macd < previous_signal and last_macd > last_signal :
    print("MACD가 시그널선을 상향 돌파했습니다.")