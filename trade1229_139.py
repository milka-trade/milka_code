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
    if ticker not in df_tickers: 
        try:
            df_tickers[ticker] = pyupbit.get_ohlcv(ticker, interval="minute15", count=50)
            if df_tickers[ticker] is None or df_tickers[ticker].empty:
                print(f"load_ohlcv / No data returned for ticker: {ticker}")
                send_discord_message(f"load_ohlcv / No data returned for ticker: {ticker}")
                time.sleep(0.1)  

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
                time.sleep(0.1)
                return float(b['balance']) if b['balance'] is not None else 0
            
    except Exception as e:
        print(f"get_balance/잔고 조회 오류: {e}")
        send_discord_message(f"get_balance/잔고 조회 오류: {e}")
        time.sleep(1)
        return 0
    return 0

def get_ema(ticker, window):
    # df = load_ohlcv(ticker)
    df = pyupbit.get_ohlcv(ticker, interval="minute15", count=50)

    if df is not None and not df.empty:
        # return df['close'].ewm(span=window, adjust=False).mean()  # EMA 계산 후 마지막 값 반환
        df['ema'] = ta.trend.EMAIndicator(close=df['close'], window=window).ema_indicator()
        return df['ema'].iloc[-1]  # EMA의 마지막 값 반환
    
    else:
        return 0  # 데이터가 없으면 0 반환

def get_best_k(ticker):
    bestK = 0.1  # 초기 K 값
    interest = 0  # 초기 수익률
    # df = load_ohlcv(ticker)  # 데이터 로드
    df = pyupbit.get_ohlcv(ticker, interval="minute15", count=20)

    if df is None or df.empty:
        return bestK  # 데이터가 없으면 초기 K 반환
    
    for k in np.arange(0.3, 0.6, 0.1):  
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
            time.sleep(0.1)  # API 호출 제한을 위한 대기

    return bestK

def get_rsi(ticker, period):
    df_rsi = load_ohlcv(ticker)
    # df_rsi = pyupbit.get_ohlcv(ticker, interval="minute15", count=50)
    if df_rsi is None or df_rsi.empty:
        return None  # 데이터가 없으면 None 반환

    # TA를 사용하여 RSI 계산
    rsi_indicator = ta.momentum.RSIIndicator(df_rsi['close'], window=period)
    rsi = rsi_indicator.rsi()

    # NaN 처리
    rsi = rsi.dropna()  # NaN 제거

    return rsi if not rsi.empty else None  # 마지막 RSI 값 반환

def stoch_rsi(ticker):
    df = pyupbit.get_ohlcv(ticker, interval="minute15", count=50)
     
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
        
    return result_df.tail(4)

def get_atr(ticker, period):
    try:
        # df_atr_day = pyupbit.get_ohlcv(ticker, interval="minute15", count=20)
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
    
    return atr.iloc[-1]

def get_dynamic_threshold(tickers):
    atr_values = []
    for t in tickers:
        try:
            atr = get_atr(t, 14)
            # print(f"ATR for {t}: {atr}")
            if atr is not None and not np.isnan(atr):  # ATR이 None이나 NaN이 아닐 경우에만 추가
                atr_values.append(atr)

        except Exception as e:
            print(f"Error getting ATR for {t}: {e}")
            continue

    # NaN 값 필터링 후 중앙값 계산
    atr_values = [value for value in atr_values if not np.isnan(value)]

    return np.median(atr_values) if atr_values else 0.05  # Fallback to 0.05 if no ATR values

def get_bollinger_bands(ticker, window=20, std_dev=2):
    """특정 티커의 볼린저 밴드 상단 및 하단값을 가져오는 함수"""
    df = load_ohlcv(ticker)
    # df = pyupbit.get_ohlcv(ticker, interval="minute15", count=50)
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

def filtered_tickers(tickers):
    """특정 조건에 맞는 티커 필터링"""
    filtered_tickers = []
    threshold_value = get_dynamic_threshold(tickers)

    for t in tickers:
        try:
            df_day = pyupbit.get_ohlcv(t, interval="day", count=1)
            if df_day is None or df_day.empty or 'high' not in df_day or 'low' not in df_day or 'open' not in df_day:
                continue
            
            day_open_price_1 = df_day['open'].iloc[-1]  # 당일 시가
            day_value_1 = df_day['value'].iloc[-1]  # 당일 거래량

            df_15 = pyupbit.get_ohlcv(t, interval="minute15", count=3)
            if df_15 is None or df_15.empty:
                continue
            
            df_15_close = df_15['close'].iloc[-3:].tolist()  # 15분 봉 종가 리스트
            cur_price = pyupbit.get_current_price(t)

            atr = get_atr(t, 14)
            rsi = get_rsi(t, 14).iloc[-1]  # 최근 RSI 값
            bands_df = get_bollinger_bands(t)
            Low_Bol = bands_df['Lower_Band'].iloc[-3:].tolist()  # 볼린저 밴드 하단가 리스트
            up_Bol1 = bands_df['Upper_Band'].iloc[-1]

            # 조건 체크
            if day_value_1 > 20_000_000_000 and cur_price < day_open_price_1:
                print(f'[cond 1] {t} 거래량 20_000백만 이상:{day_value_1} / 현재가:{cur_price} < 시가:{day_open_price_1}')
                if threshold_value < atr and Low_Bol[0] * 1.03 < up_Bol1 and rsi < 60:
                    print(f'[cond 2] {t} 임계치:{threshold_value} < atr:{atr} / 볼밴상하단 3% 이상 / rsi:{rsi} < 60')
                    if Low_Bol[0] > Low_Bol[1] > Low_Bol[2] and all(Low_Bol[i] >= df_15_close[i] for i in range(3)) :  
                        print(f"[cond 3]: [{t}] 볼린저 밴드 하향 및 종가 하단 터치")
                        if Low_Bol[0] < cur_price < Low_Bol[0] * 1.01:
                            print(f'[cond 4] {t} 현재가 볼린저밴드 1% 이내 {Low_Bol[0]:,.2f} < 현재가 : {cur_price:,.2f}')
                            send_discord_message(f"[cond 4]: [{t}] 볼린저 밴드 하향 / 현재가 배드 하단 1% 이내 : {Low_Bol[0]:,.2f} < 현재가 : {cur_price:,.2f} < 밴드1% : {Low_Bol[0]*1.01:,.2f}")
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

    filtered_time = datetime.now().strftime('%m/%d %H:%M:%S')  # 시작시간 기록
    filtered_list = filtered_tickers(tickers)
    
    send_discord_message(f"{filtered_time} [{filtered_list}]")
    print(f"[{filtered_list}]")
    
    bestC = None  # 초기 최고 코인 초기화
    interest = 0  # 초기 수익률
    best_k = 0.1  # 초기 K 값

    for ticker in filtered_list:   # 조회할 코인 필터링
        k = get_best_k(ticker)
        df = load_ohlcv(ticker)
        # df = pyupbit.get_ohlcv(ticker, interval="minute15", count=5) 
        if df is None or df.empty:
            continue
    
        df['range'] = (df['high'] - df['low']) * k  # *고가 - 저가)*k로 range열 생성
        df['target'] = df['open'] + df['range'].shift(1)  # 시가 + range로 target열 생성
        df['ror'] = np.where(df['high'] > df['open'], df['close'] / df['open'], 1)  # 수익률 계산 : 시가보다 고가가 높으면 거래성사, 수익률(종가/시가) 계산
        df['hpr'] = df['ror'].cumprod()  # 누적 수익률 계산

        if interest < df['hpr'].iloc[-1]:  # 현재 수익률이 이전보다 높으으면 업데이트
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
    buyed_amount = get_balance(ticker.split("-")[1]) 
    max_retries = 3  
    buy_size = min(1_000_000, krw*0.9995)
    
    attempt = 0  # 시도 횟수 초기화
    target_price = None  # target_price 초기화

    stoch_rsi_value = stoch_rsi(ticker)   #스토캐스틱 RSI 계산
    if not stoch_rsi_value.empty and len(stoch_rsi_value) >= 2:
        srsi_k = stoch_rsi_value['%K'].iloc[-1]
        srsi_d = stoch_rsi_value['%D'].iloc[-1]
               
    else:
        raise ValueError("stoch_rsi : Stochastic RSI data is insufficient.")
    
    if krw >= 3_000_000 :  # 매수 조건 확인
        target_price = get_target_price(ticker, k)
        
        while attempt < max_retries:
                current_price = pyupbit.get_current_price(ticker)
                print(f"가격 확인 중: [{ticker}] 현재가:{current_price:,.2f} / 목표가:{target_price:,.2f} -(시도 {attempt + 1}/{max_retries})")
                send_discord_message(f"매수시도: [{ticker}] 현재가:{current_price:,.2f} < 목표가:{target_price:,.2f} / sRSI_D:{srsi_d:,.2f} < sRSI_K:{srsi_k:,.2f} < 0.3")
                
                if current_price <= target_price and 0 < srsi_d < srsi_k < 0.3:
                    buy_attempts = 3
                    for i in range(buy_attempts):
                        try:
                            buy_order = upbit.buy_market_order(ticker, buy_size)
                            print(f"매수 성공: [{ticker}]")
                            send_discord_message(f"매수 성공: [{ticker}] 현재가:{current_price:,.2f} < 목표가:{target_price:,.2f} / sRSI_D:{srsi_d:,.2f} < sRSI_K:{srsi_k:,.2f} < 0.3")
                            return buy_order

                        except Exception as e:
                            print(f"매수 주문 실행 중 오류 발생: {e}, 재시도 중...({i+1}/{buy_attempts})")
                            send_discord_message(f"매수 주문 실행 중 오류 발생: {e}, 재시도 중...({i+1}/{buy_attempts})")
                            time.sleep(5 * (i + 1))  # Exponential backoff

                        return "Buy order failed", None

                else:
                    attempt += 1  # 시도 횟수 증가
                    time.sleep(60)

        print(f"5회 시도완료: [{ticker}], 목표가 범위에 도달하지 못함")
        return "Price not in range after max attempts", None
            
def trade_sell(ticker):

    currency = ticker.split("-")[1]
    buyed_amount = get_balance(currency)
    avg_buy_price = upbit.get_avg_buy_price(currency)
    current_price = pyupbit.get_current_price(ticker)
    profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산
    last_ema20 = get_ema(ticker, 20)    #20봉 지수이동평균 계산
    
    # 볼린저 밴드값 도출
    bands_df = get_bollinger_bands(ticker)
            
    if bands_df is not None:
        up_Bol1 = bands_df['Upper_Band'].iloc[-1]

    selltime = datetime.now()
    sell_start = selltime.replace(hour=8, minute=50 , second=00, microsecond=0)
    sell_end = selltime.replace(hour=8, minute=59, second=50, microsecond=0)

    max_attempts = 20  # 최대 조회 횟수
    attempts = 0  # 현재 조회 횟수
    
    if sell_start <= selltime <= sell_end:      # 매도 제한시간이면
        if profit_rate >= 0.3 and up_Bol1 * 0.99 <= current_price :
            sell_order = upbit.sell_market_order(ticker, buyed_amount)
            print(f"[장 시작전 매도]: [{ticker}] / 수익률: {profit_rate:.2f}% / {current_price:,.2f} < upperBand_99%: {up_Bol1 * 0.99:,.2f}")
            send_discord_message(f"[장 시작전 매도]: [{ticker}] / 수익률: {profit_rate:.2f}% / {current_price:,.2f} < upperBand_99%: {up_Bol1 * 0.99:,.2f}")
                           
    else:
        if profit_rate >= 0.6:  
            while attempts < max_attempts:
                current_price = pyupbit.get_current_price(ticker)  # 현재 가격 재조회
                profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0
                print(f"[{ticker}] / [매도시도 {attempts + 1} / {max_attempts}] / 수익률: {profit_rate:.2f}%") 
                    
                if profit_rate >= 2 and current_price > last_ema20:
                    sell_order = upbit.sell_market_order(ticker, buyed_amount)
                    print(f"[!!목표가 달성!!]: [{ticker}] / 수익률: {profit_rate:.2f} / 현재가: {current_price:,.2f} / 시도 {attempts + 1} / {max_attempts}")
                    send_discord_message(f"[!!목표가 달성!!]: [{ticker}] / 수익률: {profit_rate:.2f} / 현재가: {current_price:,.2f} / 시도 {attempts + 1} / {max_attempts}")
                    return sell_order

                else:
                    time.sleep(0.5)  # 짧은 대기                                                                                                                                                    
                attempts += 1  # 조회 횟수 증가
                
            if profit_rate >= 0.6 and current_price > last_ema20*0.98:
                sell_order = upbit.sell_market_order(ticker, buyed_amount)
                send_discord_message(f"[매도시도 초과]: [{ticker}] 수익률: {profit_rate:.2f}% 현재가: {current_price:,.2f} / 현재가가 ema20 98% 이상")
                print(f"[매도시도 초과]: [{ticker}] 수익률: {profit_rate:.2f}% 현재가: {current_price:,.2f} / 현재가가 ema20 이상")
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
                
                s_rsi = stoch_rsi(ticker)
                sRSI_K = s_rsi['%K'].iloc[-1]
                sRSI_D = s_rsi['%D'].iloc[-1]

                if buyed_amount > 0:
                    profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산
                    report_message += f"[{b['currency']}] 수익률:{profit_rate:.1f}% s_RSI_D:{sRSI_D:,.2f} < s_RSI_K:{sRSI_K:.2f}\n"

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
                time.sleep(1)

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
                if krw_balance > 1_000_000: 
                    best_ticker, interest, best_k = get_best_ticker()

                    if best_ticker:
                        buy_time = datetime.now().strftime('%m/%d %H:%M:%S')  # 시작시간 기록
                        # print(f"선정코인 : {best_ticker} / k값 : {best_k:,.2f} / 수익률 : {interest:,.2f}")
                        send_discord_message( f"[{buy_time}] 선정코인: [{best_ticker}] / k값: {best_k:,.2f} / 수익률: {interest:,.2f}")
                        result = trade_buy(best_ticker, best_k)
                        if result:  # 매수 성공 여부 확인
                            time.sleep(30)
                        else:
                            return None
                    else:
                        time.sleep(30)

                else:
                    time.sleep(30)

        except Exception as e:
            print(f"buying_logic / 에러 발생: {e}")
            send_discord_message(f"buying_logic / 에러 발생: {e}")
            time.sleep(5)

# def additional_buy_logic():

#     while True:
#         krw_balance = get_balance("KRW")  # 현재 KRW 잔고 조회
#         if krw_balance > 1_000_000: 
#             balances = upbit.get_balances()  # 잔고 조회
            
#             for b in balances:
#                 if b['currency'] not in ["KRW", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:  # 특정 통화 제외
#                     ticker = f"KRW-{b['currency']}"  # 현재가 조회를 위한 티커 설정
#                     current_price = pyupbit.get_current_price(ticker)  # 현재가 조회
                    
#                     s_rsi = stoch_rsi(ticker)
#                     sRSI_K = s_rsi['%K'].iloc[-1]
#                     sRSI_D = s_rsi['%D'].iloc[-1]
                    
#                     # 볼린저 밴드값 도출
#                     bands_df = get_bollinger_bands(ticker)
#                     low_band = bands_df['Lower_Band'].iloc[-1]

#                     avg_buy_price = upbit.get_avg_buy_price(b['currency'])  # 평균 매수 가격 조회
#                     profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산

#                     if profit_rate < -4 and current_price < low_band * 1.005 and 0 < sRSI_D < sRSI_K < 0.2 :      # 조건 체크: 수익률이 -4% 이하 현재가가 볼린저 밴드 하단 미만, sRSI 0.2미만
                        
#                         krw = get_balance("KRW")
#                         # print(f'매수 조건 만족: {ticker} / 현재가: {current_price:,.0f} / 볼린저 밴드 하단: {low_band}')
#                         buy_size = min(1_000_000, krw*0.9995)  # 추가 매수할 금액 설정
#                         result = upbit.buy_market_order(ticker, buy_size)  # 추가 매수 실행

#                         # 매수 결과 메시지 전송
#                         if result:
#                             send_discord_message(f"추가 매수: {ticker} / 수익률 : {profit_rate:,.1f} / 볼린저*1.005 : {low_band*1.005:,.2f} / 현재가 : {current_price} / sRSI_D : {sRSI_D} < sRSI_K : {sRSI_K}")

#                     else:
#                         print(f'조건 미충족: {ticker} / 현재가: {current_price:,.0f} / 수익률 : {profit_rate:,.2f}')
#                         time.sleep(0.1)
#         time.sleep(300)

# 매도 쓰레드 생성
selling_thread = threading.Thread(target=selling_logic)
selling_thread.start()

# 매수 쓰레드 생성
buying_thread = threading.Thread(target=buying_logic)
buying_thread.start()

# # # 추가 매수 쓰레드 생성
# additional_buy_thread = threading.Thread(target=additional_buy_logic)
# additional_buy_thread.start()