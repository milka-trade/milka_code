import time
import threading
import pyupbit
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import requests
import pandas as pd
import ta

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

def get_balance(ticker):
    try:
        balances = upbit.get_balances()
        for b in balances:
            if b['currency'] == ticker:
                time.sleep(0.5)
                return float(b['balance']) if b['balance'] is not None else 0
            
    except Exception as e:
        print(f"get_balance/잔고 조회 오류: {e}")
        send_discord_message(f"get_balance/잔고 조회 오류: {e}")
        time.sleep(1)
        return 0
    return 0

def get_ema(ticker, window):
    df = load_ohlcv(ticker)
    # df = pyupbit.get_ohlcv(ticker, interval="minute15", count=200)

    if df is not None and not df.empty:
        return df['close'].ewm(span=window, adjust=False).mean()  # EMA 계산 후 마지막 값 반환
    
    else:
        return 0  # 데이터가 없으면 0 반환

def get_best_k(ticker="KRW-BTC"):
    bestK = 0.5  # 초기 K 값
    interest = 0  # 초기 수익률
    df = load_ohlcv(ticker)  # 데이터 로드
    # df = pyupbit.get_ohlcv(ticker, interval="minute60", count=30)

    if df is None or df.empty:
        return bestK  # 데이터가 없으면 초기 K 반환
    
    for k in np.arange(0.1, 0.3, 0.05):  # K 값을 0.05부터 0.15까지 반복
        df['range'] = (df['high'] - df['low']) * k      #변동성 계산
        df['target'] = df['open'] + df['range'].shift(1)  # 매수 목표가 설정
        fee = 0.0005  # 거래 수수료 (0.05%로 설정)
        df['ror'] = (df['close'] / df['target'] - fee).where(df['high'] > df['target'], 1)
        ror_series = df['ror'].cumprod()  # 누적 수익률
        if len(ror_series) < 2:  # 데이터가 부족한 경우
            continue
        ror = ror_series.iloc[-2]   # 마지막 이전 값
        if ror > interest:  # 이전 수익률보다 높으면 업데이트
            interest = ror
            bestK = k
            time.sleep(1)  # API 호출 제한을 위한 대기

    return bestK

def ta_stochastic(ticker, window=14):
    # 데이터 가져오기
    df = load_ohlcv(ticker)
    # df = pyupbit.get_ohlcv(ticker, interval="minute15", count=50) 
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

def get_ta_rsi(ticker, period):
    # 데이터 가져오기
    # df_rsi = load_ohlcv(ticker)
    df_rsi = pyupbit.get_ohlcv(ticker, interval="minute15", count=50) 
    if df_rsi is None or df_rsi.empty:
        return None  # 데이터가 없으면 None 반환

    # TA 라이브러리를 사용하여 RSI 계산
    rsi = ta.momentum.RSIIndicator(df_rsi['close'], window=period).rsi()

    return rsi if not rsi.empty else None  # 마지막 RSI 값 반환

def ta_stochastic_rsi(ticker):
    # 데이터 가져오기
    # df = load_ohlcv(ticker)
    df = pyupbit.get_ohlcv(ticker, interval="minute15", count=200) 
    if df is None or df.empty:
        return None  # 데이터가 없으면 None 반환
    
    # RSI 계산
    rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # Stochastic RSI 계산
    min_rsi = rsi.rolling(window=14).min()
    max_rsi = rsi.rolling(window=14).max()

    # NaN 제거
    rsi = rsi.bfill()  # 이후 값으로 NaN 대체
    min_rsi = min_rsi.bfill()
    max_rsi = max_rsi.bfill()
    
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)
    stoch_rsi = stoch_rsi.replace([np.inf, -np.inf], np.nan)  # 무한대를 np.nan으로 대체
    stoch_rsi = stoch_rsi.fillna(0)  # NaN을 0으로 대체 (필요 시)

    # Stochastic RSI 값만 반환 
    return stoch_rsi if not stoch_rsi.empty else 0

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

def calculate_ha_candles(ticker):
    df = load_ohlcv(ticker)  # 데이터 로드
    # df = pyupbit.get_ohlcv(ticker, interval="minute15", count=50) 
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    # 하이킨아시 캔들 초기화
    ha_df = pd.DataFrame(index=df.index)

    # 첫 번째 하이킨아시 캔들 시가, 종가는 일반 캔들과 동일
    ha_df.loc[0, 'HA_Close'] = (df['open'].iloc[0] + df['high'].iloc[0] + df['low'].iloc[0] + df['close'].iloc[0]) / 4
    ha_df.loc[0, 'HA_Open'] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    ha_df.loc[0, 'HA_High'] = df['high'].iloc[0]
    ha_df.loc[0, 'HA_Low'] = df['low'].iloc[0]

    # 나머지 하이킨아시 캔들 계산
    for i in range(1, len(df)):
        ha_df.loc[i, 'HA_Open'] = (ha_df.loc[i - 1, 'HA_Open'] + ha_df.loc[i - 1, 'HA_Close']) / 2
        ha_df.loc[i, 'HA_Close'] = (df['open'].iloc[i] + df['high'].iloc[i] + df['low'].iloc[i] + df['close'].iloc[i]) / 4
        ha_df.loc[i, 'HA_High'] = max(df['high'].iloc[i], ha_df.loc[i, 'HA_Open'], ha_df.loc[i, 'HA_Close'])
        ha_df.loc[i, 'HA_Low'] = min(df['low'].iloc[i], ha_df.loc[i, 'HA_Open'], ha_df.loc[i, 'HA_Close'])

    return ha_df

def get_atr(ticker, period):
    try:
        # df_atr_day = pyupbit.get_ohlcv(ticker, interval="minute5", count=period)
        df_atr_day = load_ohlcv(ticker)
        time.sleep(0.5)  # API 호출 제한을 위한 대기
    except Exception as e:
        print(f"API call failed: {e}")
        return None

    if df_atr_day is None or df_atr_day.empty:
        print(f"get_atr/ Error: No data for {ticker}")
        return None  # 또는 기본값을 반환할 수 있음
    
    high_low = df_atr_day['high'] - df_atr_day['low']
    high_close = abs(df_atr_day['high'] - df_atr_day['close'].shift().fillna(0))
    low_close = abs(df_atr_day['low'] - df_atr_day['close'].shift().fillna(0))
    tr = high_low.combine(high_close, max).combine(low_close, max)
    atr = tr.rolling(window=period).mean()

    if atr.empty or atr.iloc[-1] is None:
        print(f"Error: No ATR data for {ticker}")
        return None  # 또는 기본값을 반환할 수 있음
    
    return atr.iloc[-1]

def get_dynamic_threshold(tickers):
    atr_values = []
    for t in tickers:
        try:
            atr = get_atr(t, 21)
            # print(f"ATR for {t}: {atr}")
            if atr is not None and not np.isnan(atr):  # ATR이 None이나 NaN이 아닐 경우에만 추가
                atr_values.append(atr)

        except Exception as e:
            print(f"Error getting ATR for {t}: {e}")
            continue

    # NaN 값 필터링 후 중앙값 계산
    atr_values = [value for value in atr_values if not np.isnan(value)]

    return np.median(atr_values) if atr_values else 0.05  # Fallback to 0.05 if no ATR values

def filtered_tickers(tickers, held_coins):
    """특정 조건에 맞는 티커 필터링"""
    filtered_tickers = []
    threshold_value = get_dynamic_threshold(tickers)

    for t in tickers:
        currency = t.split("-")[1]      # 티커에서 통화 정보 추출
        if currency in ["BTC", "ETH"] or currency in held_coins:        # BTC, ETH을 확실히 제외, 그외 보유한 코인 제외
            continue

        try:
            df = load_ohlcv(t) 
            if df is None or df.empty:
                print(f"filtered_tickers/No data for ticker: {t}")
                send_discord_message(f"filtered_tickers/No data for ticker: {t}")
                continue

            df_day = pyupbit.get_ohlcv(t, interval="day", count=7)  
            if df_day is None or df_day.empty or 'high' not in df_day or 'low' not in df_day or 'open' not in df_day:
                continue  
            
            cur_price = pyupbit.get_current_price(t)
                      
            day_open_price_1 = df_day['open'].iloc[-1]  #9시 기준 당일 시가
            day_value_1 = df_day['value'].iloc[-1] 
            atr = get_atr(t, 14)

            # New Indicators and Patterns #2
            ha_df = calculate_ha_candles(t)   #하이킨 아시 캔들 계산
            if ha_df.empty or 'HA_Close' not in ha_df.columns:
                raise ValueError("ha_df:Heikin Ashi DataFrame is empty or HA_Close column is missing.")
            
            last_ema200 = get_ema(t, 200).iloc[-1]    #200봉 지수이동평균 계산
            pre_ema200 = get_ema(t, 200).iloc[-2]
            
            ta_rsi = get_ta_rsi(t, 14)
            last_ta_rsi = ta_rsi.iloc[-1]
            
            ta_stoch_rsi = ta_stochastic_rsi(t)   #스토캐스틱 RSI 계산
            if ta_stoch_rsi.empty or len(ta_stoch_rsi) < 2:
                raise ValueError("stoch_rsi : Stochastic RSI DataFrame is empty or has insufficient data.")

            # 스토캐스틱 RSI 마지막 값 확인
            if not ta_stoch_rsi.empty and len(ta_stoch_rsi) >= 2:
                last_ta_srsi = ta_stoch_rsi.iloc[-1]
                previous_ta_srsi = ta_stoch_rsi.iloc[-2]
            else:
                raise ValueError("stoch_rsi : Stochastic RSI data is insufficient.")

            last_ta_srsi = ta_stoch_rsi.iloc[-1]
            previous_ta_srsi = ta_stoch_rsi.iloc[-2]

            # 하이킨 아시 캔들의 종가
            pre_ha_close = ha_df['HA_Close'].iloc[-2]
            last_ha_close = ha_df['HA_Close'].iloc[-1]
            last_ha_open = ha_df['HA_Open'].iloc[-1]
            # print(f"Last HA Close: {last_ha_close}")  # 마지막 HA_Close 값 출력

            # MACD 계산
            macd_df = calculate_macd(t)
            # if macd_df is None:
            # return None  # 데이터가 없으면 None 반환

            # 마지막 두 값 비교
            last_macd = macd_df['MACD'].iloc[-1]
            last_signal = macd_df['Signal'].iloc[-1]
            previous_macd = macd_df['MACD'].iloc[-2]
            previous_signal = macd_df['Signal'].iloc[-2]

            stoch_k, stoch_d = ta_stochastic(t)

            # 스토캐스틱 매수전략 도입
            latest_stoch_k = stoch_k.iloc[-1]
            latest_stoch_d = stoch_d.iloc[-1]

            if day_value_1 > 10_000_000_000 :  
                # print(f"cond1: {t} / [value] : {day_value_1:,.0f} > 10십억")

                if threshold_value < atr :  # Volatility check
                    
                    if pre_ema200 < last_ema200 and last_ema200 < last_ha_open < last_ha_close:
                            
                    # 스토캐스틱 매수신호 검증
                    # print(f"[검증 1]: [{t}] stoch_d: {latest_stoch_d:,.2f} < stoch_k:{latest_stoch_k:,.2f} < 0.25")
                    # if latest_stoch_k < 0.25 and latest_stoch_k > latest_stoch_d:
                    #     print(f"[cond 1]: [{t}] stoch_d: {latest_stoch_d:,.2f} < stoch_k:{latest_stoch_k:,.2f} < 0.25")
                        
                        if 0 < last_ta_srsi <= 0.3:
                            print(f"[cond 2]: [{t}] 0 < s-RSI:{last_ta_srsi:,.2f} <= 0.3")

                            # print(f"[검증 2.RSI]: [{t}] 0 < {last_ta_rsi:,.2f} < 60")    
                            if last_ta_rsi < 70 :
                                    print(f"[cond 3]: [{t}] RSI:{last_ta_rsi:,.2f} < 70")    

                                # print(f"[cond 4]: [{t}] macd1:{previous_macd} < signal1:{previous_signal} / macd2:{last_macd} > signal2:{last_signal}")    
                                # if last_macd > last_signal:
                                #     print(f"[cond 4]: [{t}] macd1:{previous_macd:,.2f} < signal1:{previous_signal:,.2f} / macd2:{last_macd:,.2f} > signal2:{last_signal:,.2f}")    
                                    # print(f"[cond 4]: [{t}] macd:{last_macd:,.2f} > signal:{last_signal:,.2f}")    

                                    if cur_price < day_open_price_1 * 1.2:
                                        filtered_tickers.append(t)
            
        except Exception as e:
            send_discord_message(f"filtered_tickers/Error processing ticker {t}: {e}")
            time.sleep(5)  # API 호출 제한을 위한 대기

    return filtered_tickers

def get_best_ticker():  
    
    try:
        tickers = pyupbit.get_tickers(fiat="KRW")  # 거래 가능한 모든 코인 조회
        balances = upbit.get_balances()
        held_coins = {b['currency'] for b in balances if float(b['balance']) > 0}

    except Exception as e:
        send_discord_message(f"get_best_ticker/티커 조회 중 오류 발생: {e}")
        print(f"get_best_ticker/티커 조회 중 오류 발생: {e}")
        time.sleep(1)  # API 호출 제한을 위한 대기
        return None, None, None

    filtered_list = filtered_tickers(tickers, held_coins)
    send_discord_message(f"{filtered_list}") 
    print(f"{filtered_list}")       
    
    bestC = None  # 초기 최고 코인 초기화
    interest = 0  # 초기 수익률
    best_k = 0.5  # 초기 K 값

    for ticker in filtered_list:   # 조회할 코인 필터링
        k = get_best_k(ticker)
        # df = load_ohlcv(ticker)
        df = pyupbit.get_ohlcv(ticker, interval="minute60", count=30) 
        if df is None or df.empty:
            continue
    
        df['range'] = (df['high'] - df['low']) * k  # *고가 - 저가)*k로 range열 생성
        df['target'] = df['open'] + df['range'].shift(1)  # 시가 + range로 target열 생성
        df['ror'] = np.where(df['high'] > df['open'], df['close'] / df['open'], 1)  # 수익률 계산 : 시가보다 고가가 높으면 거래성사, 수익률(종가/시가) 계산
        df['hpr'] = df['ror'].cumprod()  # 누적 수익률 계산

        if interest < df['hpr'].iloc[-1]:  # 현재 수익률이 이전보다 높으면 업데이트
            bestC = ticker
            interest = df['hpr'].iloc[-1]
            best_k = k  # 최적 K 값도 업데이트

        time.sleep(1)  # API 호출 제한을 위한 대기

    return bestC, interest, best_k  # 최고의 코인, 수익률, K 반환
    
def get_target_price(ticker, k):  #변동성 돌파 전략 구현
    # df = load_ohlcv(ticker)
    df = pyupbit.get_ohlcv(ticker, interval="minute15", count=3) 
    if df is not None and not df.empty:
        return df['close'].iloc[-1] + (df['high'].iloc[-1] - df['low'].iloc[-1]) * k
    return 0

def trade_buy(ticker, k):
    # df = load_ohlcv(ticker) 
    krw = get_balance("KRW")
    buyed_amount = get_balance(ticker.split("-")[1]) 
    max_retries = 20  
    buy_size = min(krw * 0.9995, 200_000)  
    
    attempt = 0  # 시도 횟수 초기화
    target_price = None  # target_price 초기화
    
    ta_rsi = get_ta_rsi(ticker, 14)
    last_ta_rsi = ta_rsi.iloc[-1]

    stoch_rsi = ta_stochastic_rsi(ticker)
    last_stoch_rsi = stoch_rsi.iloc[-1]

    if buyed_amount == 0 and ticker.split("-")[1] not in ["BTC", "ETH"] and krw >= 50_000 :  # 매수 조건 확인
        target_price = get_target_price(ticker, k)
        
        while attempt < max_retries:
                current_price = pyupbit.get_current_price(ticker)
                send_discord_message(f"{ticker} /목표가: {target_price:,.2f} / 현재가{current_price:,.2f}")

                if current_price < target_price :
                    print(f"매수 시도: {ticker}")
                    buy_attempts = 3
                    for i in range(buy_attempts):
                        try:
                            buy_order = upbit.buy_market_order(ticker, buy_size)
                            print(f"매수 성공: {ticker}")
                            send_discord_message(f"매수 성공: [{ticker}] 현재가 {current_price:,.2f} < 목표가 {target_price:,.2f} /(시도 {attempt + 1}/{max_retries}) \n[{ticker}] RSI {last_ta_rsi:,.2f} / S_RSI {last_stoch_rsi:,.2f}")
                            return buy_order

                        except Exception as e:
                            print(f"매수 주문 실행 중 오류 발생: {e}, 재시도 중...({i+1}/{buy_attempts})")
                            # send_discord_message(f"매수 주문 실행 중 오류 발생: {e}, 재시도 중...({i+1}/{buy_attempts})")
                            time.sleep(5 * (i + 1))  # Exponential backoff

                        return "Buy order failed", None

                else:
                    # print(f"현재가가 목표 범위에 도달하지 않음. 다음 시도로 넘어갑니다.")
                    attempt += 1  # 시도 횟수 증가
                    time.sleep(60)

        # 10회 시도 후 가격 범위에 도달하지 못한 경우
        print(f"20회 시도완료: {ticker}, 목표가 범위에 도달하지 못함")
        # send_discord_message(f"20회 시도완료: {ticker}, 목표가 범위에 도달하지 못함")
        return "Price not in range after max attempts", None
            
def trade_sell(ticker):

    currency = ticker.split("-")[1]
    buyed_amount = get_balance(currency)
    avg_buy_price = upbit.get_avg_buy_price(currency)
    current_price = pyupbit.get_current_price(ticker)
    profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산
    
    max_attempts = 120  # 최대 조회 횟수
    attempts = 0  # 현재 조회 횟수

    ta_rsi = get_ta_rsi(ticker, 14)
    
    if ta_rsi is None or ta_rsi.empty:
        print("RSI 데이터가 유효하지 않습니다.")
        return None
    last_ta_rsi = ta_rsi.iloc[-1]

    stoch_rsi = ta_stochastic_rsi(ticker)   #스토캐스틱 RSI 계산
    
    if stoch_rsi is None or stoch_rsi.empty:
        print("스토캐스틱 RSI 데이터가 유효하지 않습니다.")
        return None
    last_stoch_rsi = stoch_rsi.iloc[-1]

    if profit_rate >= 0.5:  
        while attempts < max_attempts:
            current_price = pyupbit.get_current_price(ticker)  # 현재 가격 재조회
            profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0
                
            print(f"{ticker} / 시도 {attempts + 1} / {max_attempts} - / 현재가 {current_price} 수익률 {profit_rate:.2f}% ")
                
            if profit_rate >= 1.5:
                sell_order = upbit.sell_market_order(ticker, buyed_amount)
                send_discord_message(f"매도: [{ticker}] / 현재가: {current_price}/ 수익률: {profit_rate:.2f}/ 시도 {attempts + 1} / {max_attempts} \n[{ticker}] RSI: {last_ta_rsi:,.2f} s_RSI: {last_stoch_rsi:,.2f}%")
                return sell_order

            else:
                time.sleep(1)  # 짧은 대기                
            attempts += 1  # 조회 횟수 증가
            
        if profit_rate >= 0.5 :
            sell_order = upbit.sell_market_order(ticker, buyed_amount)
            send_discord_message(f"최종 매도: {ticker}/ 현재가: {current_price}/ 수익률: {profit_rate:.2f}% / s_RSI: {last_stoch_rsi:,.2f}")
            return sell_order   
        else:
            return None
    return None

def send_profit_report():
    while True:
        try:
            now = datetime.now()  # 현재 시간을 루프 시작 시마다 업데이트 (try 루프안에 있어야 실시간 업데이트 주의)
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)   # 다음 정시 시간을 계산 (현재 시간의 분, 초를 0으로 만들어 정시로 맞춤)
            time_until_next_hour = (next_hour - now).total_seconds()
            time.sleep(time_until_next_hour)    # 다음 정시까지 기다림

            balances = upbit.get_balances()     
            report_message = "현재 수익률 보고서:\n"
            
            for b in balances:
                if b['currency'] in ["KRW", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:  # 제외할 코인 리스트
                    continue
                
                ticker = f"KRW-{b['currency']}"
                buyed_amount = float(b['balance'])
                avg_buy_price = float(b['avg_buy_price'])
                current_price = pyupbit.get_current_price(ticker)
                last_stoch_rsi = ta_stochastic_rsi(ticker).iloc[-1]
                last_rsi = get_ta_rsi(ticker,14).iloc[-1]

                if buyed_amount > 0:
                    profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산
                    report_message += f"{b['currency']} 수익률 {profit_rate:.1f}% RSI {last_rsi:,.2f} S_RSI {last_stoch_rsi:.2f}\n"

            send_discord_message(report_message)  # 슬랙으로 보고서 전송

        except Exception as e:            
            print(f"send_profit_report/수익률 보고 중 오류 발생: {e}")
            send_discord_message(f"send_profit_report/수익률 보고 중 오류 발생: {e}")
            time.sleep(5)  # API 호출 제한을 위한 대기

trade_start = datetime.now().strftime('%m/%d %H:%M:%S')  # 시작시간 기록
print(f'{trade_start} trading start')

profit_report_thread = threading.Thread(target=send_profit_report)  # 수익률 보고 쓰레드 시작
profit_report_thread.daemon = True  # 메인 프로세스 종료 시 함께 종료되도록 설정
profit_report_thread.start()

def selling_logic():
    while True:
        try:
            balances = upbit.get_balances()
            # print(f"selling_logic/잔고체크 {balances}")
            for b in balances:
                if b['currency'] not in ["KRW", "BTC", "ETH", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:
                        ticker = f"KRW-{b['currency']}"
                        # print(f"selling_logic/매도함수 {ticker}")
                        trade_sell(ticker)
                time.sleep(1)

        except Exception as e:
            print(f"selling_logic / 에러 발생: {e}")
            send_discord_message(f"selling_logic / 에러 발생: {e}")
            time.sleep(5)

def buying_logic():
    # 매수 제한 시간 설정
    restricted_start_hour = 8
    restricted_start_minute = 30
    restricted_end_hour = 9
    restricted_end_minute = 10

    while True:
        try:
            stopbuy_time = datetime.now()
            restricted_start = stopbuy_time.replace(hour=restricted_start_hour, minute=restricted_start_minute, second=0, microsecond=0)
            restricted_end = stopbuy_time.replace(hour=restricted_end_hour, minute=restricted_end_minute, second=0, microsecond=0)

            if restricted_start <= stopbuy_time <= restricted_end:  # 매수 제한 시간 체크
                time.sleep(300) 
                continue

            else:  # 매수 금지 시간이 아닐 때
                krw_balance = get_balance("KRW")  # 현재 KRW 잔고 조회
                if krw_balance > 100_000: 
                    best_ticker, interest, best_k = get_best_ticker()

                    if best_ticker:
                        print(f"선정코인 : {best_ticker} / k값 : {best_k:,.2f} / 수익률 : {interest:,.2f}")
                        # send_discord_message(f"선정코인 : {best_ticker} / k값 : {best_k:,.2f} / 수익률 : {interest:,.2f}")
                        result = trade_buy(best_ticker, best_k)
                        if result:  # 매수 성공 여부 확인
                            time.sleep(60)
                        else:
                            return None
                    else:
                        time.sleep(60)

                else:
                    time.sleep(60)

        except Exception as e:
            print(f"buying_logic / 에러 발생: {e}")
            send_discord_message(f"buying_logic / 에러 발생: {e}")
            time.sleep(5)

# 매도 쓰레드 생성
selling_thread = threading.Thread(target=selling_logic)
selling_thread.start()

# 매수 쓰레드 생성
buying_thread = threading.Thread(target=buying_logic)
buying_thread.start()