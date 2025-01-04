import time
import threading
import pyupbit
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import requests
import ta
import pandas as pd

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("discord_webhhok")
upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS"), os.getenv("UPBIT_SECRET"))
# df_tickers = {}    # 전역변수:일봉 데이터프레임

def send_discord_message(msg):
    """discord 메시지 전송"""
    try:
        message ={"content":msg}
        requests.post(DISCORD_WEBHOOK_URL, data=message)
    except Exception as e:
        print(f"디스코드 메시지 전송 실패 : {e}")
        time.sleep(5) 

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

def get_atr(ticker, period):
    try:
        df_atr_day = pyupbit.get_ohlcv(ticker, interval="minute15", count=50)
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

def get_ema(ticker, window):
    df = pyupbit.get_ohlcv(ticker, interval="minute15", count=200)
    time.sleep(0.5)

    if df is not None and not df.empty:
        df['ema'] = ta.trend.EMAIndicator(close=df['close'], window=window).ema_indicator()
        return df['ema'].iloc[-1]  # EMA의 마지막 값 반환
    
    else:
        return 0  # 데이터가 없으면 0 반환

def stoch_rsi(ticker, interval="minute15"):
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=200)
    time.sleep(0.5)
     
    rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    min_rsi = rsi.rolling(window=14).min()
    max_rsi = rsi.rolling(window=14).max()
        
    rsi = rsi.bfill()  # 이후 값으로 NaN 대체
    min_rsi = min_rsi.bfill()
    max_rsi = max_rsi.bfill()
        
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)
    stoch_rsi = stoch_rsi.replace([np.inf, -np.inf], np.nan)  # 무한대를 np.nan으로 대체
    stoch_rsi = stoch_rsi.fillna(0)  # NaN을 0으로 대체 (필요 시)

    # %K와 %D 계산
    k_period = 3  # %K 기간
    d_period = 3  # %D 기간
        
    stoch_rsi_k = stoch_rsi.rolling(window=k_period).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(window=d_period).mean()

    # 결과를 DataFrame으로 묶어서 반환
    result_df = pd.DataFrame({
            'StochRSI': stoch_rsi,
            '%K': stoch_rsi_k,
            '%D': stoch_rsi_d
        })
        
    return result_df.tail(3)

def get_bollinger_bands(ticker, interval="minute15", window=20, std_dev=2):
    """특정 티커의 볼린저 밴드 상단 및 하단값을 가져오는 함수"""
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=200)
    time.sleep(0.5)
    if df is None or df.empty:
        return None  # 데이터가 없으면 None 반환

    # 볼린저 밴드 계산
    bollinger = ta.volatility.BollingerBands(df['close'], window=window, window_dev=std_dev)

    # 상단 및 하단 밴드 값
    upper_band = bollinger.bollinger_hband().fillna(0)  
    lower_band = bollinger.bollinger_lband().fillna(0)  
    
    # DataFrame으로 묶기
    bands_df = pd.DataFrame({
        'Upper_Band': upper_band,
        'Lower_Band': lower_band
    })

    return bands_df.tail(4)  # 최근 4개의 밴드 값

def filtered_tickers(tickers, held_coins):
    """특정 조건에 맞는 티커 필터링"""
    filtered_tickers = []
    threshold_value = get_dynamic_threshold(tickers)

    # 'KRW-SOL'의 거래량을 가져옵니다.
    # df_sol = pyupbit.get_ohlcv('KRW-SOL', interval="day", count=1)
    # if df_sol is None or df_sol.empty or 'value' not in df_sol:
    #     raise ValueError("KRW-SOL의 거래량을 가져오는 데 실패했습니다.")
    
    # krw_sol_day_value = df_sol['value'].tail(1).iloc[0]  # KRW-SOL의 당일 거래량
    
    for t in tickers:
        currency = t.split("-")[1]      # 티커에서 통화 정보 추출
        if currency in held_coins:        # 보유한 코인 제외
            continue
        
        try:
            df_day = pyupbit.get_ohlcv(t, interval="day", count=1)
            if df_day is None or df_day.empty or 'high' not in df_day or 'low' not in df_day or 'open' not in df_day:
                continue
            
            day_open_price_1 = df_day['open'].tail(1).iloc[0]  # 당일 시가
            day_value = df_day['value'].tail(1).iloc[0]  # 당일 거래량

            df_15 = pyupbit.get_ohlcv(t, interval="minute15", count=3)
            if df_15 is None or df_15.empty:
                continue

            df_15_close = df_15['close'].iloc[-3:].tolist()  # 15분 봉 종가 리스트
            # print(f'{t} df_15_close[0]: {df_15_close[0]:,.2f} / df_15_close[1]: {df_15_close[1]:,.2f} / df_15_close[2]: {df_15_close[2]:,.2f}')
            df_15_open = df_15['open'].iloc[-3:].tolist()

            bands_df = get_bollinger_bands(t, interval="minute5")
            Low_Bol_5min = bands_df['Lower_Band'].iloc[-3:].tolist()  # 볼린저 밴드 하단가 리스트
            up_Bol_5min = bands_df['Upper_Band'].iloc[-1]
            # print(f'{t} Low_Bol[0]: {Low_Bol[0]:,.2f} / Low_Bol[1]: {Low_Bol[1]:,.2f} / Low_Bol[2]: {Low_Bol[2]:,.2f}')

            atr = get_atr(t, 14)
            cur_price = pyupbit.get_current_price(t)

            if cur_price < day_open_price_1*1.05 :
                # print(f'[cond 1] {t} 현가 : {cur_price:,.2f} < 시가*5% : {day_open_price_1*1.05:,.2f}')
                
                # if threshold_value < atr:
                    # print(f'[cond 1-2] {t} 임계치 : {threshold_value:,.2f} < atr : {atr:,.2f}')
                    
                    if Low_Bol_5min[2] * 1.02 < up_Bol_5min :
                        # print(f'[cond 2] {t} low_bol*2% : {Low_Bol[2]*1.02:,.2f} < up_bol : {up_Bol1:,.2f}')

                        # if any(Low_Bol[i] >= df_15_close[i] for i in range(3)) and all(Low_Bol[i] > Low_Bol[i + 1] for i in range(2)):
                        if any(Low_Bol_5min[i] >= df_15_close[i] for i in range(3)) and all(df_15_open[i] > df_15_close[i] for i in range(3)):
                            # print(f'[cond 3] {t} 볼린저 하단 터치 볼린저-3: {Low_Bol[2]:,.2f} > 종가-3: {df_15_close[2]:,.2f} / 볼린저-2: {Low_Bol[1]:,.2f} > 종가-2: {df_15_close[1]:,.2f} / 볼린저-1: {Low_Bol[0]:,.2f} > 종가-1: {df_15_close[0]:,.2f}')
                        
                            if df_15_close[2] < cur_price < Low_Bol_5min[2] * 1.01:
                                print(f'[cond 4] {t}  종가: {df_15_close[2]:,.2f} < 현재가: {cur_price:,.2f} < Low_Bol_5min*1% : {Low_Bol_5min[2]*1.01:,.2f}')
                                send_discord_message(f'[cond 4] {t}  종가: {df_15_close[2]:,.2f} < 현재가: {cur_price:,.2f} < Low_Bol_5min*1% : {Low_Bol_5min[2]*1.01:,.2f}')
                                filtered_tickers.append(t)

        except Exception as e:
            send_discord_message(f"filtered_tickers/Error processing ticker {t}: {e}")
            time.sleep(5) 

    return filtered_tickers

def get_best_k(ticker):
    bestK = 0.3  # 초기 K 값
    interest = 0  # 초기 수익률
    df = pyupbit.get_ohlcv(ticker, interval="minute15", count=200)
    time.sleep(0.5)

    if df is None or df.empty:
        return bestK  # 데이터가 없으면 초기 K 반환
    
    for k in np.arange(0.3, 0.5, 0.1):  
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
            time.sleep(0.5)  # API 호출 제한을 위한 대기

    return bestK

# def get_best_ticker():
#     selected_tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-ADA", "KRW-XLM", "KRW-DOGE", "KRW-HBAR", "KRW-SAND"]  # 지정된 코인 리스트
#     excluded_tickers = ["QI", "ONX", "ETHF", "ETHW", "PURSE", "USDT"]  # 제외할 코인 리스트
    
#     try:
#         tickers = pyupbit.get_tickers(fiat="KRW")  # 거래 가능한 모든 코인 조회
#         tickers = [ticker for ticker in selected_tickers if ticker in pyupbit.get_tickers(fiat="KRW")]  # 지정된 코인만 조회
#         # tickers = [
#         #     ticker for ticker in selected_tickers 
#         #     if ticker in pyupbit.get_tickers(fiat="KRW") and ticker not in excluded_tickers
#         # ]
#         balances = upbit.get_balances()
#         held_coins = {b['currency'] for b in balances if float(b['balance']) > 0}

#     except Exception as e:
#         send_discord_message(f"get_best_ticker/티커 조회 중 오류 발생: {e}")
#         print(f"get_best_ticker/티커 조회 중 오류 발생: {e}")
#         time.sleep(1)  # API 호출 제한을 위한 대기
#         return None, None, None

#     filtered_time = datetime.now().strftime('%m/%d %H:%M:%S')  # 시작시간 기록
#     # filtered_list = filtered_tickers(tickers)
#     filtered_list = filtered_tickers(tickers, held_coins)
    
#     # # 제외할 티커를 필터링
#     # filtered_list = [ticker for ticker in filtered_list if ticker not in excluded_tickers]
    
#     send_discord_message(f"{filtered_time} [{filtered_list}]")
#     print(f"[{filtered_list}]")
    
#     bestC = None  # 초기 최고 코인 초기화
#     interest = 0  # 초기 수익률
#     best_k = 0.3  # 초기 K 값

#     for ticker in filtered_list:   # 조회할 코인 필터링
#         k = get_best_k(ticker)
#         df = pyupbit.get_ohlcv(ticker, interval="minute15", count=200)
#         if df is None or df.empty:
#             continue
    
#         df['range'] = (df['high'] - df['low']) * k  # *고가 - 저가)*k로 range열 생성
#         df['target'] = df['open'] + df['range'].shift(1)  # 시가 + range로 target열 생성
#         df['ror'] = np.where(df['high'] > df['open'], df['close'] / df['open'], 1)  # 수익률 계산 : 시가보다 고가가 높으면 거래성사, 수익률(종가/시가) 계산
#         df['hpr'] = df['ror'].cumprod()  # 누적 수익률 계산

#         if interest < df['hpr'].iloc[-1]:  # 현재 수익률이 이전보다 높으으면 업데이트
#             bestC = ticker
#             interest = df['hpr'].iloc[-1]
#             best_k = k  # 최적 K 값도 업데이트

#     return bestC, interest, best_k  # 최고의 코인, 수익률, K 반환

def get_best_ticker():
    excluded_tickers = ["QI", "ONX", "ETHF", "ETHW", "PURSE", "USDT"]  # 제외할 코인 리스트
    
    try:
        # SOL 코인의 거래량 가져오기
        df_sol = pyupbit.get_ohlcv('KRW-SOL', interval="day", count=1)
        if df_sol is None or df_sol.empty or 'value' not in df_sol:
            raise ValueError("KRW-SOL의 거래량을 가져오는 데 실패했습니다.")
        
        krw_sol_day_value = df_sol['value'].tail(1).iloc[0]  # KRW-SOL의 당일 거래량

        # 거래 가능한 모든 코인 조회
        all_tickers = pyupbit.get_tickers(fiat="KRW")
        selected_tickers = []

        # 모든 코인에서 거래량 확인 후 SOL 거래량보다 큰 코인만 추가
        for ticker in all_tickers:
            df = pyupbit.get_ohlcv(ticker, interval="day", count=1)
            if df is not None and not df.empty:
                day_value = df['value'].tail(1).iloc[0]  # 해당 코인의 당일 거래량
                if day_value > krw_sol_day_value:  # SOL의 거래량보다 높은 경우
                    selected_tickers.append(ticker)

        # 제외할 코인 리스트에서 필터링
        selected_tickers = [ticker for ticker in selected_tickers if ticker not in excluded_tickers]

        balances = upbit.get_balances()
        held_coins = {b['currency'] for b in balances if float(b['balance']) > 0}

    except Exception as e:
        send_discord_message(f"get_best_ticker/티커 조회 중 오류 발생: {e}")
        print(f"get_best_ticker/티커 조회 중 오류 발생: {e}")
        time.sleep(1)  # API 호출 제한을 위한 대기
        return None, None, None

    filtered_time = datetime.now().strftime('%m/%d %H:%M:%S')  # 시작시간 기록
    filtered_list = filtered_tickers(selected_tickers, held_coins)  # 필터링된 리스트

    send_discord_message(f"{filtered_time} [{filtered_list}]")
    print(f"[{filtered_list}]")
    
    bestC = None  # 초기 최고 코인 초기화
    interest = 0  # 초기 수익률
    best_k = 0.3  # 초기 K 값

    for ticker in filtered_list:   # 조회할 코인 필터링
        k = get_best_k(ticker)
        df = pyupbit.get_ohlcv(ticker, interval="minute15", count=200)
        if df is None or df.empty:
            continue
    
        df['range'] = (df['high'] - df['low']) * k  # *고가 - 저가)*k로 range열 생성
        df['target'] = df['open'] + df['range'].shift(1)  # 시가 + range로 target열 생성
        df['ror'] = np.where(df['high'] > df['open'], df['close'] / df['open'], 1)  # 수익률 계산 : 시가보다 고가가 높으면 거래성사, 수익률(종가/시가) 계산
        df['hpr'] = df['ror'].cumprod()  # 누적 수익률 계산

        if interest < df['hpr'].iloc[-1]:  # 현재 수익률이 이전보다 높으시면 업데이트
            bestC = ticker
            interest = df['hpr'].iloc[-1]
            best_k = k  # 최적 K 값도 업데이트

    return bestC, interest, best_k  # 최고의 코인, 수익률, K 반환

    
def get_target_price(ticker, k):  #변동성 돌파 전략 구현
    df = pyupbit.get_ohlcv(ticker, interval="minute1", count=1) 
    if df is not None and not df.empty:
        return df['close'].iloc[-1] + (df['high'].iloc[-1] - df['low'].iloc[-1]) * k
    return 0

def trade_buy(ticker, k):
    
    krw = get_balance("KRW")
    max_retries = 10  
    buy_size = min(1_000_000, krw*0.9995)
    
    attempt = 0  # 시도 횟수 초기화
    target_price = None  # target_price 초기화

    stoch_rsi_5min = stoch_rsi(ticker, interval="5min")   #스토캐스틱 RSI 계산
    if not stoch_rsi_5min.empty and len(stoch_rsi_5min) >= 2:
        srsi_k_1 = stoch_rsi_5min['%K'].iloc[-1]
        srsi_k_2 = stoch_rsi_5min['%K'].iloc[-2]
        srsi_d = stoch_rsi_5min['%D'].iloc[-1]
               
    else:
        raise ValueError("stoch_rsi : Stochastic RSI data is insufficient.")
    
    if krw >= 500_000 :  # 매수 조건 확인
        target_price = get_target_price(ticker, k)
        
        while attempt < max_retries:
                current_price = pyupbit.get_current_price(ticker)
                print(f"가격 확인 중: [{ticker}] 현재가:{current_price:,.2f} / < 목표가:{target_price:,.2f} / 0 < sRSI_K_1:{srsi_k_1:,.2f} < 0.4/ 시도:{attempt} - 최대:{max_retries}")
                # send_discord_message(f"가격 확인 중: [{ticker}] 현재가:{current_price:,.2f} / < 목표가:{target_price:,.2f} / 0.2 < sRSI_K_2:{srsi_k_2:,.2f} < sRSI_K_1:{srsi_k_1:,.2f} < 0.4/ 시도:{attempt} - 최대:{max_retries}")
                
                if current_price <= target_price and srsi_k_1 < 0.4:
                    buy_attempts = 3
                    for i in range(buy_attempts):
                        try:
                            buy_order = upbit.buy_market_order(ticker, buy_size)
                            print(f"매수 성공: [{ticker}]")
                            send_discord_message(f"매수 성공: [{ticker}] 현재가:{current_price:,.2f} < 목표가:{target_price:,.2f} / sRSI_K_5min:{srsi_k_1:,.2f} < 0.4 / / 시도횟수:{attempt}")
                            return buy_order

                        except Exception as e:
                            print(f"매수 주문 실행 중 오류 발생: {e}, 재시도 중...({i+1}/{buy_attempts})")
                            send_discord_message(f"매수 주문 실행 중 오류 발생: {e}, 재시도 중...({i+1}/{buy_attempts})")
                            time.sleep(5 * (i + 1))  # Exponential backoff

                        return "Buy order failed", None
                else:
                    attempt += 1  # 시도 횟수 증가
                    time.sleep(30)

        print(f"3회 시도완료: [{ticker}], 목표가 범위에 도달하지 못함")
        return "Price not in range after max attempts", None
            
def trade_sell(ticker):

    currency = ticker.split("-")[1]
    buyed_amount = get_balance(currency)
    avg_buy_price = upbit.get_avg_buy_price(currency)
    current_price = pyupbit.get_current_price(ticker)
    profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산
    last_ema20 = get_ema(ticker, 20)
    df_15 = pyupbit.get_ohlcv(ticker, interval="minute15", count=1)
    df_15_high = df_15['high'].tail(1).iloc[0]
    
    stoch_rsi_15min = stoch_rsi(ticker, interval="minute15")   #스토캐스틱 RSI 계산
    if not stoch_rsi_15min.empty and len(stoch_rsi_15min) >= 2:
        srsi_k_1 = stoch_rsi_15min['%K'].iloc[-1]
        srsi_k_2 = stoch_rsi_15min['%K'].iloc[-2]
        # srsi_d = stoch_rsi_15min['%D'].iloc[-1]
               
    else:
        raise ValueError("stoch_rsi : Stochastic RSI data is insufficient.")
    
    bands_df = get_bollinger_bands(ticker)
            
    if bands_df is not None:
        up_Bol1 = bands_df['Upper_Band'].iloc[-1]

    selltime = datetime.now()
    sell_start = selltime.replace(hour=8, minute=50 , second=00, microsecond=0)
    sell_end = selltime.replace(hour=8, minute=59, second=50, microsecond=0)

    max_attempts = 20  # 최대 조회 횟수
    attempts = 0  # 현재 조회 횟수
    
    if sell_start <= selltime <= sell_end:      # 매도 제한시간이면
        if profit_rate >= 0.5 and up_Bol1 * 0.99 <= current_price :
            sell_order = upbit.sell_market_order(ticker, buyed_amount)
            print(f"[장 시작전 매도]: [{ticker}] / 수익률: {profit_rate:.2f}% / {current_price:,.2f} < upperBand_99%: {up_Bol1 * 0.99:,.2f}")
            send_discord_message(f"[장 시작전 매도]: [{ticker}] / 수익률: {profit_rate:.2f}% / {current_price:,.2f} < upperBand_99%: {up_Bol1 * 0.99:,.2f}")
                           
    else:
        if profit_rate >= 0.6:  
            while attempts < max_attempts:
                current_price = pyupbit.get_current_price(ticker)  # 현재 가격 재조회
                profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0
                print(f"[{ticker}] / [매도시도 {attempts + 1} / {max_attempts}] / 수익률: {profit_rate:.2f}%") 
                    
                if profit_rate >= 2 and current_price > up_Bol1 * 1.01 :
                    sell_order = upbit.sell_market_order(ticker, buyed_amount)
                    # print(f"[!!목표가 달성!!]: [{ticker}] / 수익률: {profit_rate:.2f} / 현재가: {current_price:,.2f} / srsi:{srsi_k:,.2f}>=0.8 / 시도 {attempts + 1} / {max_attempts}")
                    # send_discord_message(f"[!!목표가 달성!!]: [{ticker}] / 수익률: {profit_rate:.2f} / 현재가: {current_price:,.2f} / srsi:{srsi_k:,.2f}>=0.8 / 시도 {attempts + 1} / {max_attempts}")
                    print(f"[!!목표가 달성!!]: [{ticker}] / 수익률: {profit_rate:.2f} / 현재가: {current_price:,.2f} / 시도 {attempts + 1} / {max_attempts}")
                    send_discord_message(f"[!!목표가 달성!!]: [{ticker}] / 수익률: {profit_rate:.2f} / 현재가: {current_price:,.2f} / 시도 {attempts + 1} / {max_attempts}")
                    return sell_order

                else:
                    time.sleep(1)  # 짧은 대기                                                                                                                                                    
                attempts += 1  # 조회 횟수 증가
                
            if profit_rate >= 0.6 and df_15_high > up_Bol1*0.995 and srsi_k_1 > 0.8:
            # if profit_rate > 0.6 and current_price >= last_ema20 :
                sell_order = upbit.sell_market_order(ticker, buyed_amount)
                # send_discord_message(f"[매도시도 초과]: [{ticker}] 수익률: {profit_rate:.2f}% 현재가: {current_price:,.2f} > ema20:{last_ema20:,.2f}")
                # print(f"[매도시도 초과]: [{ticker}] 수익률: {profit_rate:.2f}% 현재가: {current_price:,.2f} > ema20:{last_ema20:,.2f}")
                send_discord_message(f"[매도시도 초과]: [{ticker}] 수익률: {profit_rate:.2f}% 현재가: {current_price:,.2f} srsi : {srsi_k_1:,.2f} > 0.75")
                print(f"[매도시도 초과]: [{ticker}] 수익률: {profit_rate:.2f}% 현재가: {current_price:,.2f} srsi : {srsi_k_1:,.2f} > 0.75")
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
                
                s_rsi = stoch_rsi(ticker, interval="minute15")
                sRSI_K = s_rsi['%K'].iloc[-1]

                if buyed_amount > 0:
                    profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산
                    report_message += f"[{b['currency']}] 수익률:{profit_rate:.1f}% s_RSI_K15:{sRSI_K:.2f}\n"

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
            for b in balances:
                if b['currency'] not in ["KRW", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:
                        ticker = f"KRW-{b['currency']}"
                        trade_sell(ticker)
                time.sleep(0.5)

        except Exception as e:
            print(f"selling_logic / 에러 발생: {e}")
            send_discord_message(f"selling_logic / 에러 발생: {e}")
            time.sleep(5)

def buying_logic():

    restricted_start_hour = 8
    restricted_start_minute = 40
    restricted_end_hour = 9
    restricted_end_minute = 15

    while True:
        try:
            stopbuy_time = datetime.now()
            restricted_start = stopbuy_time.replace(hour=restricted_start_hour, minute=restricted_start_minute, second=0, microsecond=0)
            restricted_end = stopbuy_time.replace(hour=restricted_end_hour, minute=restricted_end_minute, second=0, microsecond=0)

            if restricted_start <= stopbuy_time <= restricted_end:  # 매수 제한 시간 체크
                time.sleep(60) 
                continue

            else:  # 매수 금지 시간이 아닐 때
                krw_balance = get_balance("KRW")  # 현재 KRW 잔고 조회
                if krw_balance > 500_000: 
                    best_ticker, interest, best_k = get_best_ticker()

                    if best_ticker:
                        buy_time = datetime.now().strftime('%m/%d %H:%M:%S')  # 시작시간 기록
                        print(f"[{buy_time}] 선정코인: [{best_ticker}] / k값: {best_k:,.2f} / 수익률: {interest:,.2f}")
                        send_discord_message(f"[{buy_time}] 선정코인: [{best_ticker}] / k값: {best_k:,.2f} / 수익률: {interest:,.2f}")
                        result = trade_buy(best_ticker, best_k)
                        if result:  # 매수 성공 여부 확인
                            time.sleep(120)
                        else:
                            return None
                    else:
                        time.sleep(120)

                else:
                    time.sleep(120)

        except Exception as e:
            print(f"buying_logic / 에러 발생: {e}")
            send_discord_message(f"buying_logic / 에러 발생: {e}")
            time.sleep(5)

def additional_buy_logic():
    while True:
        balances = upbit.get_balances()  # 잔고 조회
        for b in balances:
            if b['currency'] not in ["KRW", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:  # 특정 통화 제외
                ticker = f"KRW-{b['currency']}"  # 현재가 조회를 위한 티커 설정
                current_price = pyupbit.get_current_price(ticker)  # 현재가 조회
                
                df_15 = pyupbit.get_ohlcv(ticker, interval="minute15", count=4)
                if df_15 is None or df_15.empty:
                    continue
            
                df_15_close = df_15['close'].iloc[-3:].tolist()  # 15분 봉 종가 리스트
                # print(f'{t} df_15_close[0]: {df_15_close[0]:,.2f} / df_15_close[1]: {df_15_close[1]:,.2f} / df_15_close[2]: {df_15_close[2]:,.2f}')
                df_15_open = df_15['open'].iloc[-3:].tolist()  # 15분 봉 시가 리스트

                bands_df = get_bollinger_bands(ticker, interval="minute5")
                Low_Bol = bands_df['Lower_Band'].iloc[-4:].tolist()  # 볼린저 밴드 하단가 리스트

                avg_buy_price = upbit.get_avg_buy_price(b['currency'])  # 평균 매수 가격 조회
                profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산
                
                stoch_rsi_5min = stoch_rsi(ticker, interval="minute5")   #스토캐스틱 RSI 계산
                srsi_k_1 = stoch_rsi_5min['%K'].iloc[-1]
                srsi_k_2 = stoch_rsi_5min['%K'].iloc[-2]

                if profit_rate < -4 and any(Low_Bol[i] >= df_15_close[i] for i in range(3)) and all(df_15_open[i] > df_15_close[i] for i in range(3)):
                    if df_15_close[2] < current_price < Low_Bol[2] * 1.01:
                        if srsi_k_1 < 0.4:
                            krw = get_balance("KRW")
                            # print(f'매수 조건 만족: {ticker} / 현재가: {current_price:,.0f} / 볼린저 밴드 하단: {low_band}')
                            buy_size = min(1_000_000, krw*0.9995)  # 추가 매수할 금액 설정
                            result = upbit.buy_market_order(ticker, buy_size)  # 추가 매수 실행

                            # 매수 결과 메시지 전송
                            if result:
                                send_discord_message(f"추가 매수: {ticker} / 수익률: {profit_rate:,.1f} / 금액: {buy_size} / 볼린저: {Low_Bol[2]:,.2f} / sRSI_5min: {srsi_k_1:,.2f}")
                                print(f"추가 매수: {ticker} / 수익률: {profit_rate:,.1f} / 금액: {buy_size:,.1f} / 볼린저: {Low_Bol[2]:,.2f} / sRSI_5min: {srsi_k_1:,.2f}")

                else:
                    print(f'조건 미충족: {ticker} / 현재가: {current_price:,.0f} / 수익률 : {profit_rate:,.1f}')
        time.sleep(150)


# 매도 쓰레드 생성
selling_thread = threading.Thread(target=selling_logic)
selling_thread.start()

# 매수 쓰레드 생성
buying_thread = threading.Thread(target=buying_logic)
buying_thread.start()

# # 추가 매수 쓰레드 생성
additional_buy_thread = threading.Thread(target=additional_buy_logic)
additional_buy_thread.start()