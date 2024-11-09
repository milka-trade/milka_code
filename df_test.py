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

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("discord_webhhok")
upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS"), os.getenv("UPBIT_SECRET"))

df_tickers = {}    # 전역변수:일봉 데이터프레임

def send_discord_message(msg):
    """discord 메시지 전송"""
    try:
        message ={"content":msg}
        requests.post(DISCORD_WEBHOOK_URL, data=message)
    except Exception as e:
        print(f"디스코드 메시지 전송 실패 : {e}")
        time.sleep(5) 

def load_ohlcv(ticker):
    global df_tickers
    if ticker not in df_tickers:   # 티커가 캐시에 없으면 데이터 가져오기     
        try:
            df_tickers[ticker] = pyupbit.get_ohlcv(ticker, interval="minute15", count=200) 
            if df_tickers[ticker] is None or df_tickers[ticker].empty:
                print(f"load_ohlcv / No data returned for ticker: {ticker}")
                send_discord_message(f"load_ohlcv / No data returned for ticker: {ticker}")
                time.sleep(0.1)  # API 호출 제한을 위한 대기

        except Exception as e:
            print(f"load_ohlcv / 디스코드 메시지 전송 실패 : {e}")
            send_discord_message(f"load_ohlcv / Error loading data for ticker {ticker}: {e}")
            time.sleep(1)
    return df_tickers.get(ticker)


# df= pyupbit.get_ohlcv('KRW-BTC', interval="minute60", count=3) 
# print(df)
# print(df['close'].iloc[-1])

def get_rsi(ticker, period):
    # df_rsi = pyupbit.get_ohlcv(ticker, interval="minute5", count=period)
    df_rsi = load_ohlcv(ticker)
    # df_rsi = pyupbit.get_ohlcv(ticker, interval="day", count=15)
    delta = df_rsi['close'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

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

try:
    tickers = pyupbit.get_tickers(fiat="KRW")  # 거래 가능한 모든 코인 조회
    
except Exception as e:
        send_discord_message(f"get_best_ticker/티커 조회 중 오류 발생: {e}")
        print(f"get_best_ticker/티커 조회 중 오류 발생: {e}")
        time.sleep(1)  # API 호출 제한을 위한 대기
        return None