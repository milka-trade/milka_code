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
minute5 = "minute5"

second=1.0
min_krw = 50_000
cut_rate = -3.0

def send_discord_message(msg):
    """discord 메시지 전송"""
    try:
        message ={"content":msg}
        requests.post(DISCORD_WEBHOOK_URL, data=message)
    except Exception as e:
        print(f"디스코드 메시지 전송 실패 : {e}")
        time.sleep(5) 

def get_user_input():
    while True:
        try:
            trade_Quant = float(input("매수 금액 (예: 1_000_000): "))
            min_rate = float(input("최소 수익률 (예: 0.5): "))
            max_rate = float(input("최대 수익률 (예: 2.5): "))
            sell_time = int(input("매도감시횟수 (예: 20): "))
            break  # 모든 입력이 성공적으로 완료되면 루프 종료
        except ValueError:
            print("잘못된 입력입니다. 다시 시도하세요.")

    return trade_Quant, min_rate, max_rate, sell_time

# 함수 호출 및 결과 저장
trade_Quant, min_rate, max_rate, sell_time = get_user_input()

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
    time.sleep(second)

    if df is not None and not df.empty:
        df['ema'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
        return df['ema'].tail(2)  # EMA의 마지막 값 반환
    
    else:
        return 0  # 데이터가 없으면 0 반환

def stoch_rsi(ticker, interval = minute):
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

    k_period = 3  # %K 기간
    d_period = 3  # %D 기간
        
    stoch_rsi_k = stoch_rsi.rolling(window=k_period).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(window=d_period).mean()

    result_df = pd.DataFrame({  # 결과를 DataFrame으로 묶어서 반환
            'StochRSI': stoch_rsi,
            '%K': stoch_rsi_k,
            '%D': stoch_rsi_d
        })
        
    return result_df.tail(3)

def get_bollinger_bands(ticker, interval = minute, window=20, std_dev=2):
    """특정 티커의 볼린저 밴드 상단 및 하단값을 가져오는 함수"""
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count_200)
    time.sleep(second)
    if df is None or df.empty:
        return None  # 데이터가 없으면 None 반환

    bollinger = ta.volatility.BollingerBands(df['close'], window=window, window_dev=std_dev)

    upper_band = bollinger.bollinger_hband().fillna(0)  
    lower_band = bollinger.bollinger_lband().fillna(0)  
    
    bands_df = pd.DataFrame({   # DataFrame으로 묶기
        'Upper_Band': upper_band,
        'Lower_Band': lower_band
    })

    return bands_df.tail(4)

def filtered_tickers(tickers):
    """특정 조건에 맞는 티커 필터링"""
    filtered_tickers = []
    
    for t in tickers:
        try:
            bands_df = get_bollinger_bands(t, interval = minute5)
            upper_band = bands_df['Upper_Band'].values
            lower_band = bands_df['Lower_Band'].values
            band_diff = (upper_band - lower_band) / lower_band
            # slopes = np.diff(lower_band)

            stoch_Rsi = stoch_rsi(t, interval = minute5)
            srsi_k = stoch_Rsi['%K'].values
            srsi_d = stoch_Rsi['%D'].values

            cur_price = pyupbit.get_current_price(t)
            last_ema = get_ema(t, interval = minute5).iloc[1]
            df_ema = get_ema(t, interval = minute).values
            ema_rising = df_ema[0] < df_ema[1]
            # print(f'{t} {df_ema[0]:,.2f} < {df_ema[1]:,.2f}')
            
            is_increasing = band_diff[len(band_diff) - 1] > 0.02 #for i in range(len(band_diff) - 1))
            # decreasing = all(slopes[i] > slopes[i + 1] for i in range(len(slopes) - 1))
            srsi_d_rising = 0.15 < srsi_k[2] < 0.35 and srsi_d[2] < srsi_k[2]
            under_ema = cur_price < last_ema

            if ema_rising :
                # print(f'{t} [con1] ema상향: {df_ema[0]:,.2f} < {df_ema[1]:,.2f}')
                if is_increasing :
                    # print(f'{t} [con2] 볼린저 최소폭 유지')
                    if under_ema :
                        print(f'{t} [con3] 현재가 ema 하단')
                        if srsi_d_rising :
                            print(f'{t} [con4] srsi K-D 교차 D: {srsi_d[2]:,.2f} < K: {srsi_k[2]:,.2f} < 0.35')
                            send_discord_message(f'{t} [con3] srsi K-D 교차 D: {srsi_d[2]:,.2f} < K: {srsi_k[2]:,.2f} < 0.35')
                            filtered_tickers.append(t)
                
        except (KeyError, ValueError) as e:
            send_discord_message(f"filtered_tickers/Error processing ticker {t}: {e}")
            time.sleep(5) 

    return filtered_tickers

def get_best_ticker():
    selected_tickers = ["KRW-ETH", "KRW-BTC", "KRW-XRP", "KRW-SOL", "KRW-ADA", "KRW-HBAR", "KRW-XLM", "KRW-DOGE"]  #"KRW-BTC", 
    # excluded_tickers = ["KRW-QI", "KRW-ONX", "KRW-ETHF", "KRW-ETHW", "KRW-PURSE", "KRW-USDT", "KRW-BERA", "KRW-VTHO", "KRW-SBD", "KRW-JTO", "KRW-SCR", "KRW-VIRTUAL", "KRW-SOLVE", "KRW-IOST"]  # 제외할 코인 리스트
    balances = upbit.get_balances()
    held_coins = []

    for b in balances:
        if float(b['balance']) > 0:  # 보유량이 0보다 큰 경우
            ticker = f"KRW-{b['currency']}"  # 현재가 조회를 위한 티커 설정
            held_coins.append(ticker)  # "KRW-코인명" 형태로 추가
    
    try:
# 
        all_tickers = pyupbit.get_tickers(fiat="KRW")
        filtering_tickers = []

        for ticker in all_tickers:
            if ticker in selected_tickers and ticker not in held_coins:

                df_day = pyupbit.get_ohlcv(ticker, interval="day", count=1)
                time.sleep(second)
                day_price = df_day['open'].iloc[0]
                
                cur_price = pyupbit.get_current_price(ticker)

                if cur_price < day_price * 1.03:
                    filtering_tickers.append(ticker)
                            
    except (KeyError, ValueError) as e:
        send_discord_message(f"get_best_ticker/티커 조회 중 오류 발생: {e}")
        time.sleep(second) 
        return None

    filtered_list = filtered_tickers(filtering_tickers)
    if len(filtered_list) > 0 :
        filtered_time = datetime.now().strftime('%m/%d %H:%M:%S')
        send_discord_message(f"{filtered_time} [{filtered_list}]")
    
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
    max_retries = 5
    buy_size = min(trade_Quant, krw*0.9995)
    cur_price = pyupbit.get_current_price(ticker)
    # last_ema = get_ema(ticker, interval = minute5).iloc[1]
    
    attempt = 0 
       
    stoch_Rsi = stoch_rsi(ticker, interval = minute5)
    srsi_k = stoch_Rsi['%K'].values
    srsi_d = stoch_Rsi['%D'].values
    srsi_buy = 0.15 < srsi_k[2] < 0.35 and srsi_d[2] < srsi_k[2]
    # under_ema = cur_price < last_ema

    if krw >= min_krw :
        while attempt < max_retries:
            print(f"[가격 확인 중]: {ticker} srsi_buy: {srsi_buy} / 현재가: {cur_price:,.2f} / 시도: {attempt} - 최대: {max_retries}")
            
            if srsi_buy :
                buy_attempts = 3
                for i in range(buy_attempts):
                    try:
                        buy_order = upbit.buy_market_order(ticker, buy_size)
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

        print(f"[매수 실패]: {ticker} / 현재가: {cur_price:,.2f} / srsi_d: {srsi_d[2]:,.2f} < srsi_k: {srsi_k[2]:,.2f} < 0.35")
        send_discord_message(f"[매수 실패]: {ticker} / 현재가: {cur_price:,.2f} / srsi_d: {srsi_d[2]:,.2f} < srsi_k: {srsi_k[2]:,.2f} < 0.35")
        return "Price not in range after max attempts", None
            
def trade_sell(ticker):
    currency = ticker.split("-")[1]
    buyed_amount = get_balance(currency)
    
    avg_buy_price = upbit.get_avg_buy_price(currency)
    cur_price = pyupbit.get_current_price(ticker)
    profit_rate = (cur_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산

    bands_df = get_bollinger_bands(ticker, interval = minute5)
    up_Bol = bands_df['Upper_Band'].values

    stoch_Rsi = stoch_rsi(ticker, interval = minute5)
    srsi_k = stoch_Rsi['%K'].values
    srsi_d = stoch_Rsi['%D'].values

    srsi_sell = srsi_d[2] > 0.7 and srsi_k[2] < srsi_d[2]
    upper_boliinger = cur_price > up_Bol[3] and srsi_d[2] > 0.8
    upper_price = profit_rate >= min_rate and upper_boliinger
    middle_price = srsi_sell
    cut_time_price = srsi_k[2] < srsi_d[2]

    max_attempts = sell_time
    attempts = 0

    cut_time = datetime.now()
    cut_start = cut_time.replace(hour=8, minute=55, second=00, microsecond=0)
    cut_end = cut_time.replace(hour=9, minute=1, second=55, microsecond=0)

    if cut_start <= cut_time <= cut_end:      # 매도 제한시간이면
        if cut_time_price :
            sell_order = upbit.sell_market_order(ticker, buyed_amount)
            send_discord_message(f"[장시작전매도]: [{ticker}] 수익률: {profit_rate:.2f}% / 현재가: {cur_price:,.1f} srsi_d: {srsi_d[2]:,.2f} > srsi_k: {srsi_k[2]:,.2f}")
        else:
            time.sleep(1)
            return None  

    else:
        if profit_rate >= min_rate:
            while attempts < max_attempts:       
                print(f"[{ticker}] / [매도시도 {attempts + 1} / {max_attempts}] / 수익률: {profit_rate:.2f}% / upper_price : {upper_price}")

                if profit_rate >= max_rate or upper_price :
                    sell_order = upbit.sell_market_order(ticker, buyed_amount)
                    print(f"[!!목표가 달성!!]: [{ticker}] / 수익률: {profit_rate:.2f}%  / 현재가: {cur_price:,.1f} \n upper_price: {upper_price} / srsi_d {srsi_d[2]:,.2f} > 0.7 / 시도 {attempts + 1} / {max_attempts}")
                    send_discord_message(f"[!!목표가 달성!!]: [{ticker}] / 수익률: {profit_rate:.2f}%  / 현재가: {cur_price:,.1f} \n upper_price: {upper_price} / srsi_d {srsi_d[2]:,.2f} > 0.7 / 시도 {attempts + 1} / {max_attempts}")
                    return sell_order

                else:
                    time.sleep(second)
                attempts += 1  # 조회 횟수 증가
                
            if middle_price:
                sell_order = upbit.sell_market_order(ticker, buyed_amount)
                print(f"[m_price 도달]: [{ticker}] 수익률: {profit_rate:.2f}% / 현재가: {cur_price:,.1f} / srsi_d: {srsi_d[2]:,.2f} > srsi_k: {srsi_k[2]:,.2f}")
                send_discord_message(f"[m_price 도달]: [{ticker}] 수익률: {profit_rate:.2f}% / 현재가: {cur_price:,.1f} / srsi_d: {srsi_d[2]:,.2f} > srsi_k: {srsi_k[2]:,.2f}")
                return sell_order   
            else:
                middle_price_time = datetime.now().strftime('%m/%d %H:%M:%S')
                print(f"[m_price 미도달]: [{middle_price_time}][{ticker}] 수익률: {profit_rate:.2f}% / 현재가: {cur_price:,.1f} / srsi_d: {srsi_d[2]:,.2f} > srsi_k: {srsi_k[2]:,.2f}")
                return None
        else:
            if profit_rate < cut_rate:
                sell_order = upbit.sell_market_order(ticker, buyed_amount)
                send_discord_message(f"[손절_CutRate]: [{ticker}] 수익률: {profit_rate:.2f}% / 현재가: {cur_price:,.1f} / srsi_d: {srsi_d[2]:,.2f} > srsi_k: {srsi_k[2]:,.2f}")
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
                    report_message += f"[{b['currency']}] 수익률: {profit_rate:.2f}% / 현재가: {cur_price:,.2f} \n"

            send_discord_message(report_message)

        except (KeyError, ValueError) as e:          
            print(f"send_profit_report/수익률 보고 중 오류 발생: {e}")
            send_discord_message(f"send_profit_report/수익률 보고 중 오류 발생: {e}")
            time.sleep(5)

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

# 매도 쓰레드 생성
selling_thread = threading.Thread(target = selling_logic)
selling_thread.start()

# 매수 쓰레드 생성
buying_thread = threading.Thread(target = buying_logic)
buying_thread.start()