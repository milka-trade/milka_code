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
count_50 = 50

minute = "minute15"
minute5 = "minute5"
window = 20

second=0.5

trade_Quant=1_000_000
add_trade_Quant=500_000
bol_touch_time = 2
min_rate = 0.6
max_rate = 3.0
profit_margin = -2
min_krw = 50_000
sell_time = 10

def send_discord_message(msg):
    """discord 메시지 전송"""
    try:
        message ={"content":msg}
        requests.post(DISCORD_WEBHOOK_URL, data=message)
    except Exception as e:
        print(f"디스코드 메시지 전송 실패 : {e}")
        time.sleep(5) 

def get_user_input():
    global trade_Quant, add_trade_Quant, bol_touch_time, max_rate, profit_margin, sell_time

    trade_Quant = float(input("최대 단위 금액 (예: 1_000_000): "))
    add_trade_Quant = float(input("추가매수 금액 (예: 500_000): "))
    bol_touch_time = int(input("볼린저 밴드 접촉 횟수 (예: 2): "))
    # min_rate = float(input("최소 수익률 (예: 0.6): "))
    max_rate = float(input("최대 수익률 (예: 3.0): "))
    profit_margin = float(input("추가매수 감시 수익률 (예: -2): "))
    # min_krw = float(input("최소 거래금액 (예: 50_000): "))
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

def get_ema(ticker, interval = minute, window=window):
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count_50)
    time.sleep(second)

    if df is not None and not df.empty:
        df['ema'] = ta.trend.EMAIndicator(close=df['close'], window=window).ema_indicator()
        return df['ema'].iloc[-1]  # EMA의 마지막 값 반환
    
    else:
        return 0  # 데이터가 없으면 0 반환

def stoch_rsi(ticker, interval = minute5):
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count_200)
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

def get_bollinger_bands(ticker, interval = minute5, window=20, std_dev=2):
    """특정 티커의 볼린저 밴드 상단 및 하단값을 가져오는 함수"""
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count_200)
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

    return bands_df.tail(4)

def filtered_tickers(tickers):
    """특정 조건에 맞는 티커 필터링"""
    filtered_tickers = []
    
    for t in tickers:
        try:
            df = pyupbit.get_ohlcv(t, interval=minute5, count=4)
            if df is None:
                print(f"[filter_tickers] 데이터를 가져올 수 없습니다. {t}")
                send_discord_message(f"[filter_tickers] 데이터를 가져올 수 없습니다: {t}")
                continue  # 다음 티커로 넘어감
            time.sleep(1)            
            df_low = df['low'].values

            bands_df = get_bollinger_bands(t, interval= minute5)
            lower_band = bands_df['Lower_Band'].values

            stoch_Rsi = stoch_rsi(t, interval = minute5)
            srsi_k = stoch_Rsi['%K'].values

            # (조건 1)볼린저 밴드 하단값이 4시계열 연속 하락하고 상하단 폭이 2%이상 유지하는지 확인
            is_downing = all(lower_band[i] > lower_band[i + 1] for i in range(len(lower_band) - 1))
            
            # (조건 2) 볼린저 밴드의 하단값과 종가를 비교하여, 종가가 하단값 이하인 경우가 n번 이상 발생하는지 확인
            count_below_lower_band = sum(1 for i in range(len(lower_band)) if df_low[i] < lower_band[i])
            lower_boliinger = count_below_lower_band >= bol_touch_time
            srsi_buy = 0 <= srsi_k[1] < srsi_k[2] < 0.5
           
            # print(f'[미선정] {t} 볼린저 하락: {is_downing} / 볼린저 터치: {lower_boliinger} / srsi: {srsi_buy} {srsi_k[1]:,.2f} < {srsi_k[2]:,.2f}')
            if is_downing :
                # print(f'[미선정] {t} 볼린저 하락: {is_downing} / 볼린저 터치: {lower_boliinger} / srsi: {srsi_buy} {srsi_k[1]:,.2f} < {srsi_k[2]:,.2f}')
                
                if lower_boliinger and srsi_buy :
                    print(f'{t} 볼린저 하락:{is_downing} / 볼린저 터치:{lower_boliinger} / srsi: {srsi_buy} {srsi_k[1]:,.2f} < {srsi_k[2]:,.2f}')
                    # send_discord_message(f'{t} 볼린저 하락:{is_downing} / 볼린저 터치:{lower_boliinger} / srsi: {srsi_buy} {srsi_k[1]:,.2f} < {srsi_k[2]:,.2f}')
                    filtered_tickers.append(t)
                    
        except (KeyError, ValueError) as e:
            send_discord_message(f"filtered_tickers/Error processing ticker {t}: {e}")
            time.sleep(5) 

    return filtered_tickers

def get_best_ticker():
    selected_tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-ADA", "KRW-XLM", "KRW-DOGE", "KRW-HBAR"]
    balances = upbit.get_balances()
    held_coins = []

    for b in balances:
        if float(b['balance']) > 0:  # 보유량이 0보다 큰 경우
            ticker = f"KRW-{b['currency']}"  # 현재가 조회를 위한 티커 설정
            held_coins.append(ticker)  # "KRW-코인명" 형태로 추가
    
    try:
        all_tickers = pyupbit.get_tickers(fiat="KRW")
        filtering_tickers = []

        for ticker in all_tickers:
            if ticker in selected_tickers and ticker not in held_coins:
                cur_price = pyupbit.get_current_price(ticker)
                df = pyupbit.get_ohlcv(ticker, interval="day", count=2)
                time.sleep(0.5)
                day_price = df['open'].values

                if cur_price < day_price[1] * 1.05 and cur_price < day_price[0] * 1.15:  
                    filtering_tickers.append(ticker)

    except (KeyError, ValueError) as e:
        send_discord_message(f"get_best_ticker/티커 조회 중 오류 발생: {e}")
        # print(f"get_best_ticker/티커 조회 중 오류 발생: {e}")
        time.sleep(second)  # API 호출 제한을 위한 대기
        return None, None, None

    filtered_list = filtered_tickers(filtering_tickers)
    
    # 티커가 1개인 경우 바로 반환
    if len(filtered_list) == 1:
        return filtered_list[0]  # 티커가 1개인 경우 해당 티커 반환
    
    bestC = None  # 초기 최고 코인 초기화
    interest = 0  # 초기 수익률

    for ticker in filtered_list:   # 조회할 코인 필터링
        k = 0.5
        df = pyupbit.get_ohlcv(ticker, interval=minute5, count=10)
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
    max_retries = 5  
    buy_size = min(trade_Quant, krw*0.9995)
    cur_price = pyupbit.get_current_price(ticker)
    
    attempt = 0 
    
    df = pyupbit.get_ohlcv(ticker, interval=minute5, count=4)
    time.sleep(second)            
    df_close = df['close'].values
    df_open = df['open'].values
    
    bands_df = get_bollinger_bands(ticker, interval = minute5)
    lower_band = bands_df['Lower_Band'].values
    last_df_open = df_open[len(df_open) - 1]
    last_df_close = df_close[len(df_close) - 1]

    low_price = (last_df_open < last_df_close) and (lower_band[len(lower_band) - 1] < cur_price < lower_band[len(lower_band) - 1] * 1.02)
    
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
    current_price = pyupbit.get_current_price(ticker)
    profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산

    last_ema20 = get_ema(ticker, interval = minute, window = window)
    
    stoch_Rsi = stoch_rsi(ticker, interval = minute5)
    srsi= stoch_Rsi['%K'].values

    bands_df_5 = get_bollinger_bands(ticker, interval= minute5)
    up_Bol_5 = bands_df_5['Upper_Band'].values
    srsi_sell = 0.8 < srsi[1] > srsi[2]
    upper_price = profit_rate >= min_rate and current_price > up_Bol_5[len(up_Bol_5)-1] * 1.01 and srsi_sell
    middle_price = profit_rate >= min_rate and current_price > last_ema20 * 1.005 and srsi_sell

    max_attempts = sell_time  # 최대 조회 횟수
    attempts = 0  # 현재 조회 횟수
    
    if profit_rate >= min_rate:
        while attempts < max_attempts:
            current_price = pyupbit.get_current_price(ticker)  # 현재 가격 재조회
            profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0
            print(f"[{ticker}] / [매도시도 {attempts + 1} / {max_attempts}] / 수익률: {profit_rate:.1f}% / up_price : {upper_price} srsi: {srsi_sell} {srsi[1]:,.2f} > {srsi[2]:,.2f}") 
                
            if profit_rate >= max_rate or upper_price :
                sell_order = upbit.sell_market_order(ticker, buyed_amount)
                # print(f"[!!목표가 달성!!]: [{ticker}] / 수익률: {profit_rate:.2f}% / 현재가: {current_price:,.1f} / \n UP_price: {upper_price} srsi:  {srsi_sell} {srsi[1]:,.2f} > {srsi[2]:,.2f} / 시도 {attempts + 1} / {max_attempts}")
                send_discord_message(f"[!!목표가 달성!!]: [{ticker}] / 수익률: {profit_rate:.2f}% / 현재가: {current_price:,.1f} / \n UP_price: {upper_price} srsi:  {srsi_sell} {srsi[1]:,.2f} > {srsi[2]:,.2f} / 시도 {attempts + 1} / {max_attempts}")
                return sell_order

            else:
                time.sleep(second)                                                                                                                            
            attempts += 1  # 조회 횟수 증가
            
        if middle_price :
            sell_order = upbit.sell_market_order(ticker, buyed_amount)
            # print(f"[매도시도 초과]: [{ticker}] 수익률: {profit_rate:.1f}% 현재가: {current_price:,.1f} ema: {last_ema20:,.2f} / srsi: {srsi[1]:,.2f} > {srsi[2]:,.2f}")
            send_discord_message(f"[매도시도 초과]: [{ticker}] 수익률: {profit_rate:.1f}% 현재가: {current_price:,.1f} ema: {last_ema20:,.2f} / srsi: {srsi[1]:,.2f} > {srsi[2]:,.2f}")
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
                
                if buyed_amount > 0:
                    profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산
                    report_message += f"[{b['currency']}] 현재가: {current_price:,.1f} 수익률: {profit_rate:.1f}% \n"

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
                    buy_time = datetime.now().strftime('%m/%d %H:%M:%S')  # 시작시간 기록
                    # print(f"[{buy_time}] 선정코인: [{best_ticker}]")
                    send_discord_message(f"[{buy_time}] 선정코인: [{best_ticker}]")
                    result = trade_buy(best_ticker)
                    if result:  # 매수 성공 여부 확인
                        time.sleep(30)
                    else:
                        return None
                else:
                    time.sleep(30)

            else:
                time.sleep(30)

        except (KeyError, ValueError) as e:
            print(f"buying_logic / 에러 발생: {e}")
            send_discord_message(f"buying_logic / 에러 발생: {e}")
            time.sleep(5)

def additional_buy_logic():
    while True:
        balances = upbit.get_balances()
        krw = get_balance("KRW")
        add_buy_size = min(add_trade_Quant, krw*0.9995)
        
        for b in balances:
            if b['currency'] not in ["KRW", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:  # 특정 통화 제외
                ticker = f"KRW-{b['currency']}"  # 현재가 조회를 위한 티커 설정
                
                currency = ticker.split("-")[1]
                buyed_amount = get_balance(currency)
                
                cur_price = pyupbit.get_current_price(ticker)  # 현재가 조회
                avg_buy_price = upbit.get_avg_buy_price(b['currency']) 
                profit_rate = (cur_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0 
                holding_value = buyed_amount * cur_price if cur_price is not None else 0
            
                df = pyupbit.get_ohlcv(ticker, interval=minute, count=4)
                time.sleep(second)
                df_close = df['close'].values
                df_open = df['open'].values
                df_low = df['low'].values

                bands_df = get_bollinger_bands(ticker, interval= minute)
                lower_band = bands_df['Lower_Band'].values
                
                is_downing = all(lower_band[i] > lower_band[i + 1] for i in range(len(lower_band) - 1))

                # 볼린저 밴드의 하단값과 저가를 비교하여, 종가가 하단값 이하인 경우가 n번 발생여부 확인
                count_below_lower_band = sum(1 for i in range(len(lower_band)) if df_low[i] < lower_band[i])
                lower_boliinger = count_below_lower_band >= bol_touch_time

                last_LBand = lower_band[len(lower_band) - 1]
                last_df_open = df_open[len(df_open) - 1]
                last_df_close = df_close[len(df_close) - 1]
                low_price = (last_df_open < last_df_close) and (last_LBand < cur_price < last_LBand * 1.02)
                                
                stoch_Rsi = stoch_rsi(ticker, interval = minute)
                srsi_k = stoch_Rsi['%K'].values
                srsi_buy = 0 <= srsi_k[1] < srsi_k[2] < 0.5
        
                if profit_rate < profit_margin and krw > min_krw and holding_value < add_buy_size * 6 :

                    if is_downing and lower_boliinger and srsi_buy :
                        if low_price :
                            result = upbit.buy_market_order(ticker, add_buy_size)  # 추가 매수 실행

                            if result:
                                # print(f"추가 매수: {ticker} / 수익률: {profit_rate:,.1f} / 금액: {add_buy_size:,.0f}")
                                send_discord_message(f"[추가 매수]: {ticker} / 현재가: {cur_price:,.1f} / 수익률: {profit_rate:,.1f} / 금액: {add_buy_size:,.0f}")

                        else:
                            print(f'[추가매수 미충족]: {ticker} / 수익률: {profit_rate:,.1f} / 현재가: {cur_price:,.1f}')
                            send_discord_message(f'[추가매수 미충족]: {ticker} / 수익률: {profit_rate:,.1f} / 현재가: {cur_price:,.1f} / 볼린저하락: {is_downing} / 볼린저터치: {lower_boliinger} / srsi: {srsi_buy} / low_price:{low_price}')
                            time.sleep(second)
            
        time.sleep(60)

# 매도 쓰레드 생성
selling_thread = threading.Thread(target=selling_logic)
selling_thread.start()

# 매수 쓰레드 생성
buying_thread = threading.Thread(target=buying_logic)
buying_thread.start()

# # 추가 매수 쓰레드 생성
additional_buy_thread = threading.Thread(target=additional_buy_logic)
additional_buy_thread.start()