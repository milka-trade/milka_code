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

count_200 = 200

minute = "minute15"
# minute5 = "minute5"

second=1.0

trade_Quant = 1_000_000
bol_touch_time = 2
bol_touch_time_add = 3
min_rate = 0.6
max_rate = 5.0
min_krw = 50_000
sell_time = 20
bol_upper_time = 1
up_bol_rate = 1.003
cut_rate = -5.0

add_buy_rate1  = -4.0
add_buy_quant1 = 1_000_000

# add_buy_rate2  = -2.0
add_buy_max    = 2_000_000


def send_discord_message(msg):
    """discord 메시지 전송"""
    try:
        message ={"content":msg}
        requests.post(DISCORD_WEBHOOK_URL, data=message)
    except Exception as e:
        print(f"디스코드 메시지 전송 실패 : {e}")
        time.sleep(5) 

def get_user_input():
    global trade_Quant, bol_touch_time, bol_upper_time, min_rate, max_rate, sell_time 

    trade_Quant = float(input("매수 금액 (예: 1_000_000): "))
    bol_touch_time = int(input("볼린저 밴드 하단 접촉 횟수 (예: 2): "))
    bol_upper_time = int(input("볼린저 밴드 상단 접촉 횟수 (예: 1): "))
    min_rate = float(input("최소 수익률 (예: 0.6): "))
    max_rate = float(input("최대 수익률 (예: 5.0): "))
    sell_time = int(input("매도감시횟수 (예: 10): "))

def get_balance(ticker):
    try:
        balances = upbit.get_balances()
        for b in balances:
            if b['currency'] == ticker:
                time.sleep(0.5)
                return float(b['balance']) if b['balance'] is not None else 0
            
    except (KeyError, ValueError) as e:
        print(f"get_balance/잔고 조회 오류: {e}")
        send_discord_message(f"get_balance/잔고 조회 오류: {e}")
        time.sleep(1)
        return 0
    return 0

def get_ema(ticker, interval = minute):
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count_200)
    # print(df)
    time.sleep(second)

    if df is not None and not df.empty:
        df['ema'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
        return df['ema'].tail(2)  # EMA의 마지막 값 반환
    
    else:
        return 0  # 데이터가 없으면 0 반환

def stoch_rsi(ticker, interval = minute):
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count_200)
    # print(df)
    time.sleep(second)
     
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

def get_bollinger_bands(ticker, interval = minute, window=20, std_dev=2):
    """특정 티커의 볼린저 밴드 상단 및 하단값을 가져오는 함수"""
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count_200)
    # print(df)
    time.sleep(second)
    if df is None or df.empty:
        return None  # 데이터가 없으면 None 반환

    bollinger = ta.volatility.BollingerBands(df['close'], window=window, window_dev=std_dev)

    upper_band = bollinger.bollinger_hband().fillna(0)  
    lower_band = bollinger.bollinger_lband().fillna(0)  
    
    # DataFrame으로 묶기
    bands_df = pd.DataFrame({
        'Upper_Band': upper_band,
        'Lower_Band': lower_band
    })

    return bands_df.tail(3)

def filtered_tickers(tickers):
    """특정 조건에 맞는 티커 필터링"""
    filtered_tickers = []
    
    for t in tickers:
        try:
            df = pyupbit.get_ohlcv(t, interval=minute, count=3)
            # print(df)
            if df is None:
                print(f"[filter_tickers] 데이터를 가져올 수 없습니다. {t}")
                send_discord_message(f"[filter_tickers] 데이터를 가져올 수 없습니다: {t}")
                continue  # 다음 티커로 넘어감
            time.sleep(second)
            
            df_open = df['open'].values            
            df_low = df['low'].values
            df_close = df['close'].values

            last_df_open = df_open[len(df_open) - 1]
            last_df_close = df_close[len(df_close) - 1]

            bands_df = get_bollinger_bands(t, interval = minute)
            upper_band = bands_df['Upper_Band'].values
            lower_band = bands_df['Lower_Band'].values
            band_diff = upper_band - lower_band

            stoch_Rsi = stoch_rsi(t, interval = minute)
            srsi_k = stoch_Rsi['%K'].values

            # filiter_time = datetime.now().strftime('%m/%d %H:%M:%S')  
            # print(f'[{filiter_time}] {t} df_close:{df_close} / lower_band:{lower_band} / srsi:{srsi_k}')
            
            # is_downing = all(lower_band[i] > lower_band[i + 1] for i in range(len(lower_band) - 1))
            is_increasing = all(band_diff[i] < band_diff[i + 1] for i in range(len(band_diff) - 1))
            count_below_lower_band = sum(1 for i in range(len(lower_band)) if df_low[i] < lower_band[i])
            lower_boliinger = count_below_lower_band >= bol_touch_time
            upper_candle = last_df_open < last_df_close
            srsi_buy = 0 <= srsi_k[1] < srsi_k[2] < 0.3
           
            # print(f'[test] {t} 볼린저 확대:{is_increasing} / 볼린저 터치:{lower_boliinger} / 양봉:{upper_candle} / srsi: {srsi_buy} {srsi_k[1]:,.2f} < {srsi_k[2]:,.2f}')
            if is_increasing :
                # print(f'[미선정] {t} 볼린저 하락: {is_downing} / 볼린저 터치: {lower_boliinger} / srsi: {srsi_buy} {srsi_k[1]:,.2f} < {srsi_k[2]:,.2f}')
                
                if lower_boliinger and upper_candle and srsi_buy :
                    print(f'{t} 볼린저 확대:{is_increasing} / 볼린저 터치:{lower_boliinger} / 양봉:{upper_candle} / srsi: {srsi_k[1]:,.2f} < {srsi_k[2]:,.2f}')
                    # send_discord_message(f'{t} 볼린저 하락:{is_downing} / 볼린저 터치:{lower_boliinger} / srsi: {srsi_buy} {srsi_k[1]:,.2f} < {srsi_k[2]:,.2f}')
                    filtered_tickers.append(t)
                    
        except (KeyError, ValueError) as e:
            send_discord_message(f"filtered_tickers/Error processing ticker {t}: {e}")
            time.sleep(5) 

    return filtered_tickers

def get_best_ticker():
    selected_tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-ADA", "KRW-HBAR", "KRW-XLM", "KRW-DOGE"] 
    # excluded_tickers = ["KRW-QI", "KRW-ONX", "KRW-ETHF", "KRW-ETHW", "KRW-PURSE", "KRW-USDT", "KRW-BERA", "KRW-VTHO", "KRW-SBD", "KRW-JTO", "KRW-SCR", "KRW-VIRTUAL", "KRW-SOLVE", "KRW-IOST"]  # 제외할 코인 리스트
    balances = upbit.get_balances()
    held_coins = []

    for b in balances:
        if float(b['balance']) > 0:  # 보유량이 0보다 큰 경우
            ticker = f"KRW-{b['currency']}"  # 현재가 조회를 위한 티커 설정
            held_coins.append(ticker)  # "KRW-코인명" 형태로 추가
    
    try:
        # df_sol = pyupbit.get_ohlcv('KRW-SOL', interval="day", count=1)
        # time.sleep(0.1)
        # krw_sol_day_value = df_sol['value'].iloc[0]  # KRW-SOL의 당일 거래량

        all_tickers = pyupbit.get_tickers(fiat="KRW")
        filtering_tickers = []

        for ticker in all_tickers:
            if ticker in selected_tickers and ticker not in held_coins:
            # if ticker not in excluded_tickers or ticker in selected_tickers :
                # if ticker not in held_coins : 
                
                    df_week = pyupbit.get_ohlcv(ticker, interval="week", count=1)
                    time.sleep(second)
                    week_price = df_week['open'].iloc[0]
                    
                    df_day = pyupbit.get_ohlcv(ticker, interval="day", count=1)
                    time.sleep(second)
                    day_price = df_day['open'].iloc[0]
                    # day_value = df_day['value'].iloc[0]
                    
                    cur_price = pyupbit.get_current_price(ticker)

                    # if day_value >= krw_sol_day_value and 
                    if cur_price < week_price * 1.3 and cur_price < day_price * 1.05:
                        filtering_tickers.append(ticker)

    except (KeyError, ValueError) as e:
        send_discord_message(f"get_best_ticker/티커 조회 중 오류 발생: {e}")
        # print(f"get_best_ticker/티커 조회 중 오류 발생: {e}")
        time.sleep(second)  # API 호출 제한을 위한 대기
        return None

    
    filtered_list = filtered_tickers(filtering_tickers)
    if len(filtered_list) > 0 :
        filtered_time = datetime.now().strftime('%m/%d %H:%M:%S')  # 시작시간 기록
        send_discord_message(f"{filtered_time} [{filtered_list}]")
    
    # 티커가 1개인 경우 바로 반환
    if len(filtered_list) == 1:
        return filtered_list[0]  # 티커가 1개인 경우 해당 티커 반환
    
    bestC = None  # 초기 최고 코인 초기화
    interest = 0  # 초기 수익률

    for ticker in filtered_list:   # 조회할 코인 필터링
        df = pyupbit.get_ohlcv(ticker, interval=minute, count=10)
        time.sleep(second)
        if df is None or df.empty:
            continue
    
        df['ror'] = np.where(df['high'] > df['open'], df['close'] / df['open'], 1)  # 수익률 계산 : 시가보다 고가가 높으면 거래성사, 수익률(종가/시가) 계산
        df['hpr'] = df['ror'].cumprod()  # 누적 수익률 계산

        if interest < df['hpr'].iloc[-1]:  # 현재 수익률이 이전보다 높으면 업데이트
            bestC = ticker
            interest = df['hpr'].iloc[-1]

    return bestC  # 최고의 코인, 수익률, K 반환

def trade_buy(ticker):
    
    krw = get_balance("KRW")
    max_retries = 10
    buy_size = min(trade_Quant, krw*0.9995)
    cur_price = pyupbit.get_current_price(ticker)
    
    attempt = 0 
       
    bands_df = get_bollinger_bands(ticker, interval = minute)
    lower_band = bands_df['Lower_Band'].values
    last_LBand = lower_band[len(lower_band) - 1]

    low_price = (last_LBand < cur_price < last_LBand * 1.01)
    
    if krw >= min_krw :
        
        while attempt < max_retries:
            
            print(f"[가격 확인 중]: {ticker} lowPrice: {low_price}  / 현재가: {cur_price:,.2f} / 시도: {attempt} - 최대: {max_retries}")
            # send_discord_message(f"가격 확인 중: [{ticker}] lowPrice: {low_price}  / 현재가: {cur_price:,.2f} / 시도: {attempt} - 최대: {max_retries}")
            
            if low_price :
                buy_attempts = 3
                for i in range(buy_attempts):
                    try:
                        buy_order = upbit.buy_market_order(ticker, buy_size)
                        # print(f"[매수 성공]: {ticker} / 현재가 :{cur_price:,.2f}")
                        send_discord_message(f"매수 성공: {ticker} / 현재가 :{cur_price:,.2f}")
                        return buy_order

                    except (KeyError, ValueError) as e:
                        print(f"매수 주문 실행 중 오류 발생: {e}, 재시도 중...({i+1}/{buy_attempts})")
                        send_discord_message(f"매수 주문 실행 중 오류 발생: {e}, 재시도 중...({i+1}/{buy_attempts})")
                        time.sleep(5 * (i + 1)) 

                return "Buy order failed", None
            else:
                attempt += 1  # 시도 횟수 증가
                time.sleep(2)

        # print(f"3회 시도완료: {ticker}, 목표가 범위에 도달하지 못함")
        send_discord_message(f"[매수 실패]: {ticker} / 현재가: {cur_price:,.2f} / 시도횟수: {attempt} ")
        return "Price not in range after max attempts", None
            
def trade_sell(ticker):
    currency = ticker.split("-")[1]
    buyed_amount = get_balance(currency)
    
    avg_buy_price = upbit.get_avg_buy_price(currency)
    cur_price = pyupbit.get_current_price(ticker)
    profit_rate = (cur_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산
    holding_value = buyed_amount * cur_price if cur_price is not None else 0

    last_ema = get_ema(ticker, interval = minute).iloc[1]
    pre_ema = get_ema(ticker, interval = minute).iloc[0]

    bands_df = get_bollinger_bands(ticker, interval = minute)
    upper_band = bands_df['Upper_Band'].values
    lower_band = bands_df['Lower_Band'].values
    band_diff = upper_band - lower_band
    is_increasing = all(band_diff[i] < band_diff[i + 1] for i in range(len(band_diff) - 1))
    
    stoch_Rsi = stoch_rsi(ticker, interval = minute)
    srsi = stoch_Rsi['%K'].values

    bands_df = get_bollinger_bands(ticker, interval = minute)
    up_Bol = bands_df['Upper_Band'].values
    

    df = pyupbit.get_ohlcv(ticker, interval = minute, count = 3)
    time.sleep(second)
    df_close = df['close'].values

    srsi_sell = 0.7 < srsi[1] > srsi[2] and 0.95 > srsi[2]
    # srsi_sell_m = 0.7 < srsi[1] > srsi[2] and 0.95 > srsi[2]

    count_upper_band = sum(1 for i in range(len(up_Bol)) if df_close[i] > up_Bol[i] * up_bol_rate)  #1.0025
    upper_boliinger = count_upper_band >= bol_upper_time

    upper_price = profit_rate >= min_rate and is_increasing and pre_ema < last_ema and upper_boliinger
    middle_price = profit_rate >= min_rate and cur_price > last_ema * 1.005 and srsi_sell
    cut_price = profit_rate < cut_rate or (is_increasing and pre_ema < last_ema and upper_boliinger) 


    max_attempts = sell_time
    attempts = 0

    if profit_rate >= min_rate:
        while attempts < max_attempts:
            # current_price = pyupbit.get_current_price(ticker)  # 현재 가격 재조회
            # profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0
            print(f"[{ticker}] / [매도시도 {attempts + 1} / {max_attempts}] / 수익률: {profit_rate:.2f}% / upper_price : {upper_price}")

            # if profit_rate >= max_rate or (upper_boliinger and upper_price) :
            if profit_rate >= max_rate or upper_price :
                sell_order = upbit.sell_market_order(ticker, buyed_amount)
                # print(f"[!!목표가 달성!!]: [{ticker}] / 수익률: {profit_rate:.2f}% / 현재가: {current_price:,.1f} / \n UP_price: {upper_price} / srsi: {srsi_sell} {srsi[1]:,.2f} > {srsi[2]:,.2f} / 시도 {attempts + 1} / {max_attempts}")
                send_discord_message(f"[!!목표가 달성!!]: [{ticker}] / 수익률: {profit_rate:.2f}%  / 현재가: {cur_price:,.1f} \n upper_price: {upper_price} / srsi5 {srsi[1]:,.2f} > {srsi[2]:,.2f} / 시도 {attempts + 1} / {max_attempts}")
                return sell_order

            else:
                time.sleep(second)
            attempts += 1  # 조회 횟수 증가
            
        if middle_price:
            sell_order = upbit.sell_market_order(ticker, buyed_amount)
            send_discord_message(f"[m_price 도달!]: [{ticker}] 수익률: {profit_rate:.2f}% / 현재가: {cur_price:,.1f} / srsi5: {srsi[1]:,.2f} > {srsi[2]:,.2f}")
            return sell_order   
        else:
            print(f"[m_price 미도달]: [{ticker}] 수익률: {profit_rate:.2f}% / 현재가: {cur_price:,.1f} / srsi5: {srsi[1]:,.2f} > {srsi[2]:,.2f}")
            return None
    else:
        if add_buy_max * 0.95 < holding_value:
            if cut_price:
                sell_order = upbit.sell_market_order(ticker, buyed_amount)
                send_discord_message(f"[손절조건 도달]: [{ticker}] 수익률: {profit_rate:.2f}% / 현재가: {cur_price:,.1f} / 보유금액: {holding_value:,.0f}  \n upper_price: {upper_price} / srsi5: {srsi[1]:,.2f} > {srsi[2]:,.2f}")
            else:
                # print(f"[손절조건 미도달]: [{ticker}] 수익률: {profit_rate:.2f}% / 보유금액: {holding_value:,.0f} /cut_cond: {cut_cond} / srsim: {srsi[1]:,.2f} > {srsi[2]:,.2f}")
                time.sleep(5)
                return None  
        else:
            return None  

def send_profit_report():
    while True:
        try:
            now = datetime.now()  # 현재 시간을 루프 시작 시마다 업데이트 (try 루프안에 있어야 실시간 업데이트 주의)
            next_hour = (now + timedelta(hours = 1)).replace(minute = 0, second = 0, microsecond = 0)   # 다음 정시 시간을 계산 (현재 시간의 분, 초를 0으로 만들어 정시로 맞춤)
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
                cur_price = pyupbit.get_current_price(ticker)
                
                if buyed_amount > 0:
                    profit_rate = (cur_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0
                    holding_value = buyed_amount * cur_price if cur_price is not None else 0
                    report_message += f"[{b['currency']}] 수익률: {profit_rate:.2f}% / 현재가: {cur_price:,.2f} / 보유금액: {holding_value:,.0f}원 \n"

            send_discord_message(report_message)

        except (KeyError, ValueError) as e:          
            print(f"send_profit_report/수익률 보고 중 오류 발생: {e}")
            send_discord_message(f"send_profit_report/수익률 보고 중 오류 발생: {e}")
            time.sleep(5)  # API 호출 제한을 위한 대기

trade_start = datetime.now().strftime('%m/%d %H:%M:%S')  # 시작시간 기록
get_user_input()
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
                time.sleep(second)

        except Exception as e:
            print(f"selling_logic / 에러 발생: {e}")
            send_discord_message(f"selling_logic / 에러 발생: {e}")
            time.sleep(5)

def buying_logic():
    while True:
        try:
            krw_balance = get_balance("KRW")  # 현재 KRW 잔고 조회
            if krw_balance > min_krw: 
                best_ticker = get_best_ticker()

                if best_ticker:
                    buy_time = datetime.now().strftime('%m/%d %H:%M:%S')
                    send_discord_message(f"[{buy_time}] 선정코인: [{best_ticker}]")
                    result = trade_buy(best_ticker)
                    
                    if result:
                        time.sleep(10)
                    else:
                        time.sleep(10)
                else:
                    time.sleep(10)

            else:
                time.sleep(600)

        except (KeyError, ValueError) as e:
            print(f"buying_logic / 에러 발생: {e}")
            send_discord_message(f"buying_logic / 에러 발생: {e}")
            time.sleep(5)

def additional_buy_logic():
    while True:
        try:
            balances = upbit.get_balances()
            krw = get_balance("KRW")
            add_Quant = min(trade_Quant, krw * 0.9995)

            for b in balances:
                if b['currency'] not in ["KRW", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:  # 특정 통화 제외
                    ticker = f"KRW-{b['currency']}"  # 현재가 조회를 위한 티커 설정
                    currency = ticker.split("-")[1]
                    buyed_amount = get_balance(currency)
                    
                    cur_price = pyupbit.get_current_price(ticker)
                    avg_buy_price = upbit.get_avg_buy_price(b['currency'])
                    profit_rate = (cur_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0
                    holding_value = buyed_amount * cur_price if cur_price is not None else 0
                
                    df = pyupbit.get_ohlcv(ticker, interval = minute, count = 3)
                    time.sleep(second)
                    df_close = df['close'].values
                    df_open = df['open'].values
                    # df_low = df['low'].values

                    bands_df = get_bollinger_bands(ticker, interval = minute)
                    upper_band = bands_df['Upper_Band'].values
                    lower_band = bands_df['Lower_Band'].values
                    band_diff = upper_band - lower_band

                    # is_downing = all(lower_band[i] > lower_band[i + 1] for i in range(len(lower_band) - 1))
                    is_increasing = all(band_diff[i] < band_diff[i + 1] for i in range(len(band_diff) - 1))

                    count_below_lower_band = sum(1 for i in range(len(lower_band)) if df_close[i] < lower_band[i])
                    lower_boliinger = count_below_lower_band >= bol_touch_time_add

                    last_LBand = lower_band[len(lower_band) - 1]
                    last_df_open = df_open[len(df_open) - 1]
                    last_df_close = df_close[len(df_close) - 1]
                    low_price = last_df_open < last_df_close and last_LBand < cur_price < last_LBand * 1.005
                                    
                    stoch_Rsi = stoch_rsi(ticker, interval = minute)
                    srsi_k = stoch_Rsi['%K'].values
                    srsi_buy = 0 < srsi_k[1] < srsi_k[2] < 0.3

                    # 새로운 수익률 조건 추가
                    if (holding_value < add_buy_quant1) and (profit_rate <= add_buy_rate1):    #평가금액이 50만원 미만이고 수익률이 -1프로 이하일때 추가매수
                        add_buy_cond = True
                    # elif (add_buy_quant1 < holding_value <= add_buy_max) and (profit_rate <= add_buy_rate2):  #평가금액이 50만원 초과이고 수익률이 -2프로 이하일때 추가매수
                    #     add_buy_cond = True
                    else:
                        add_buy_cond = False
                    
                    if add_buy_cond and krw > min_krw :
                        if is_increasing and lower_boliinger and srsi_buy :
                            if low_price :
                                result = upbit.buy_market_order(ticker, add_Quant)

                                if result:
                                    send_discord_message(f"[추가 매수]: {ticker} / 수익률: {profit_rate:,.2f} / 현재가: {cur_price:,.1f} / 금액: {add_Quant:,.0f}")
                                    time.sleep(600)

                            else:
                                time.sleep(60)

        except (KeyError, ValueError) as e:
            print(f"add_buy_logic / 에러 발생: {e}")
            send_discord_message(f"add_buy_logic / 에러 발생: {e}")
            time.sleep(5)  # 오류 발생 시 대기 후 재시작

# 매도 쓰레드 생성
selling_thread = threading.Thread(target = selling_logic)
selling_thread.start()

# 매수 쓰레드 생성
buying_thread = threading.Thread(target = buying_logic)
buying_thread.start()

# # 추가 매수 쓰레드 생성
additional_buy_thread = threading.Thread(target = additional_buy_logic)
additional_buy_thread.start()