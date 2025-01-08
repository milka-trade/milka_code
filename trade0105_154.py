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

count_200 = 200
count_50 = 50
count_20 = 20

minute1 = "minute1"
minute5 = "minute5"
# minute15 = "minute15"

second60=60
second1=1
second05=0.5
# second01=0.1

trade=1_500_000
bol_con = 2
min_rate = 0.6

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
                time.sleep(second05)
                return float(b['balance']) if b['balance'] is not None else 0
            
    except (KeyError, ValueError) as e:
        print(f"get_balance/잔고 조회 오류: {e}")
        send_discord_message(f"get_balance/잔고 조회 오류: {e}")
        time.sleep(1)
        return 0
    return 0

# def get_ema(ticker, window):
#     df = pyupbit.get_ohlcv(ticker, interval=minute5, count=count_200)
#     time.sleep(second05)

#     if df is not None and not df.empty:
#         df['ema'] = ta.trend.EMAIndicator(close=df['close'], window=window).ema_indicator()
#         return df['ema'].iloc[-1]  # EMA의 마지막 값 반환
    
#     else:
#         return 0  # 데이터가 없으면 0 반환

def stoch_rsi(ticker, interval=minute5):
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count_50)
    time.sleep(second05)
     
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

def get_bollinger_bands(ticker, interval=minute5, window=20, std_dev=2):
    """특정 티커의 볼린저 밴드 상단 및 하단값을 가져오는 함수"""
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count_200)
    time.sleep(second05)
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
    
    for t in tickers:
        try:
            cur_price = pyupbit.get_current_price(t)
            
            df_5 = pyupbit.get_ohlcv(t, interval=minute5, count=4)
            time.sleep(second05)            
            df_5_close = df_5['close'].values

            bands_df = get_bollinger_bands(t, interval=minute5)
            lower_band = bands_df['Lower_Band'].values
            upper_band = bands_df['Upper_Band'].values
            band_diff = upper_band - lower_band

            stoch_Rsi = stoch_rsi(t, interval=minute5)
            srsi_k = stoch_Rsi['%K'].values

            is_increasing = all(band_diff[i] < band_diff[i + 1] for i in range(len(band_diff) - 1))
            
            # 볼린저 밴드의 하단값과 종가를 비교하여, 종가가 하단값 이하인 경우의 수를 센다
            count_below_lower_band = sum(1 for i in range(len(lower_band)) if df_5_close[i] < lower_band[i])
            
            # 종가가 볼린저 밴드의 하단값 이하인 경우가 2번 이상 발생하는지 확인
            lower_boliinger = count_below_lower_band >= bol_con
            
            # low_price = (df_5_close[len(band_diff) - 1] < cur_price < lower_band[len(band_diff) - 1] * 1.005) or (cur_price < lower_band[len(band_diff) - 1]*0.99)

            if is_increasing:
                # print(f'[cond 1] {t} 볼린저 폭 확대: {lower_band[0]:,.1f} > {lower_band[1]:,.1f} > {lower_band[2]:,.1f}')
                    
                if lower_boliinger and 0 < srsi_k[1] < srsi_k[2] < 0.3 : 
                    # print(f'[cond 2] {t} lower_boliinger: {lower_boliinger} >= 2 / 0 < srsi1: {srsi_k[1]:,.2f} < srsk2: {srsi_k[2]:.2f} < 0.3')
                    # send_discord_message(f'[cond 2] {t} lower_boliinger: {lower_boliinger} >= 2 / 0 < srsi1: {srsi_k[1]:,.2f} < srsk2: {srsi_k[2]:,.2f} < 0.3')
                    filtered_tickers.append(t)

        except (KeyError, ValueError) as e:
            send_discord_message(f"filtered_tickers/Error processing ticker {t}: {e}")
            time.sleep(5) 

    return filtered_tickers

def get_best_k(ticker):
    bestK = 0.3  # 초기 K 값
    interest = 0  # 초기 수익률
    df = pyupbit.get_ohlcv(ticker, interval=minute5, count=count_20)
    time.sleep(second05)

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
            time.sleep(second05)  # API 호출 제한을 위한 대기

    return bestK

def get_best_ticker():
    selected_tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-ADA", "KRW-XLM", "KRW-DOGE", "KRW-HBAR"]  # 지정된 코인 리스트
    excluded_tickers = ["KRW-QI", "KRW-ONX", "KRW-ETHF", "KRW-ETHW", "KRW-PURSE", "KRW-USDT", 
                        "KRW-BTG", "KRW-SBD", "KRW-STG", "KRW-HIVE", "KRW-TOKAMAK", "KRW-AKT", "KRW-HPO" ]  # 제외할 코인 리스트
    balances = upbit.get_balances()
    held_coins = []

    for b in balances:
        # if b['currency'] not in ["KRW", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:  # 특정 통화 제외
        if float(b['balance']) > 0:  # 보유량이 0보다 큰 경우
            ticker = f"KRW-{b['currency']}"  # 현재가 조회를 위한 티커 설정
            held_coins.append(ticker)  # "KRW-코인명" 형태로 추가
    
    try:
        df_sol = pyupbit.get_ohlcv('KRW-SOL', interval="day", count=1)
        time.sleep(second05)
        sol_value = df_sol['value'].values  # KRW-SOL의 당일 거래량
        
        all_tickers = pyupbit.get_tickers(fiat="KRW")
        filtering_tickers = []
        
        for ticker in all_tickers:
            if ticker not in excluded_tickers and ticker not in held_coins:
                df = pyupbit.get_ohlcv(ticker, interval="day", count=1)
                time.sleep(second05)               
                day_value = df['value'].values
                day_price = df['open'].values

                cur_price = pyupbit.get_current_price(ticker)

                if ticker in selected_tickers or day_value[0] > sol_value[0] * 2 :
                    if cur_price < day_price[0] * 1.05:  
                        # print(f'{ticker} 일 거래량: {day_value[0]:,.0f} >= SOL 거래량: {sol_value[0]:,.0f}')
                        filtering_tickers.append(ticker)

    except (KeyError, ValueError) as e:
        send_discord_message(f"get_best_ticker/티커 조회 중 오류 발생: {e}")
        print(f"get_best_ticker/티커 조회 중 오류 발생: {e}")
        time.sleep(second1)  # API 호출 제한을 위한 대기
        return None, None, None

    filtered_time = datetime.now().strftime('%m/%d %H:%M:%S')  # 시작시간 기록
    filtered_list = filtered_tickers(filtering_tickers)
    
    send_discord_message(f"{filtered_time} [{filtered_list}]")
    print(f"[{filtered_list}]")
    
    bestC = None  # 초기 최고 코인 초기화
    interest = 0  # 초기 수익률
    best_k = 0.3  # 초기 K 값

    for ticker in filtered_list:   # 조회할 코인 필터링
        k = get_best_k(ticker)
        df = pyupbit.get_ohlcv(ticker, interval=minute5, count=count_20)
        time.sleep(second05)
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
    
def get_target_price(ticker, k):  
    df = pyupbit.get_ohlcv(ticker, interval=minute1, count=1)
    time.sleep(second05)
    if df is not None and not df.empty:
        return df['close'].iloc[-1] + (df['high'].iloc[-1] - df['low'].iloc[-1]) * k
    return 0

def trade_buy(ticker, k):
    
    krw = get_balance("KRW")
    max_retries = 3  
    buy_size = min(trade, krw*0.9995)
    cur_price = pyupbit.get_current_price(ticker)
    
    attempt = 0  # 시도 횟수 초기화
    target_price = None  # target_price 초기화

    # df = pyupbit.get_ohlcv(ticker, interval=minute5, count=4)
    # time.sleep(second05)            
    # df_close = df['close'].values

    bands_df = get_bollinger_bands(ticker, interval=minute5)
    lower_band = bands_df['Lower_Band'].values

    low_price = (lower_band[len(lower_band) - 1] < cur_price < lower_band[len(lower_band) - 1] * 1.005) or (cur_price < lower_band[len(lower_band) - 1]*0.99)
    
    if krw >= 500_000 :  # 매수 조건 확인
        target_price = get_target_price(ticker, k)
        
        while attempt < max_retries:
            
            print(f"가격 확인 중: [{ticker}] 현재가:{cur_price:,.2f} / < 목표가:{target_price:,.2f}  시도:{attempt} - 최대:{max_retries}")
            # send_discord_message(f"가격 확인 중: [{ticker}] 현재가:{cur_price:,.2f} / < 목표가:{target_price:,.2f}  시도:{attempt} - 최대:{max_retries}")
            
            if cur_price <= target_price and low_price:
                buy_attempts = 3
                for i in range(buy_attempts):
                    try:
                        buy_order = upbit.buy_market_order(ticker, buy_size)
                        print(f"매수 성공: {ticker} < 타겟: {target_price:,.2f} / low_price 조건 충족: {low_price} / 시도횟수:{attempt}")
                        send_discord_message(f"매수 성공: {ticker} < 타겟: {target_price:,.2f} / low_price 조건 충족: {low_price} / 시도횟수:{attempt}")
                        return buy_order

                    except (KeyError, ValueError) as e:
                        print(f"매수 주문 실행 중 오류 발생: {e}, 재시도 중...({i+1}/{buy_attempts})")
                        send_discord_message(f"매수 주문 실행 중 오류 발생: {e}, 재시도 중...({i+1}/{buy_attempts})")
                        time.sleep(5 * (i + 1)) 

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
    # last_ema20 = get_ema(ticker, 20)
    
    df = pyupbit.get_ohlcv(ticker, interval=minute5, count=1)
    time.sleep(second05)
    df_high = df['high'].values
    
    stoch_Rsi = stoch_rsi(ticker, interval=minute5)   #스토캐스틱 RSI 계산
    srsi_k_1 = stoch_Rsi['%K'].values
    
    bands_df = get_bollinger_bands(ticker, interval=minute5)
    up_Bol = bands_df['Upper_Band'].values

    selltime = datetime.now()
    sell_start = selltime.replace(hour=8, minute=50 , second=00, microsecond=0)
    sell_end = selltime.replace(hour=8, minute=59, second=50, microsecond=0)

    max_attempts = 10  # 최대 조회 횟수
    attempts = 0  # 현재 조회 횟수
    
    if sell_start <= selltime <= sell_end:      # 매도 제한시간이면
        if profit_rate >= 0.5 and up_Bol[len(up_Bol)-1] * 0.99 <= current_price :
            sell_order = upbit.sell_market_order(ticker, buyed_amount)
            print(f"[장 시작전 매도]: [{ticker}] / 수익률: {profit_rate:.1f}% / {current_price:,.2f} < upperBand_99%: {up_Bol[len(up_Bol)-1] * 0.99:,.2f}")
            send_discord_message(f"[장 시작전 매도]: [{ticker}] / 수익률: {profit_rate:.1f}% / {current_price:,.2f} < upperBand_99%: {up_Bol[len(up_Bol)-1] * 0.99:,.2f}")
                           
    else:
        if profit_rate >= min_rate:
            while attempts < max_attempts:
                current_price = pyupbit.get_current_price(ticker)  # 현재 가격 재조회
                profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0
                print(f"[{ticker}] / [매도시도 {attempts + 1} / {max_attempts}] / 수익률: {profit_rate:.1f}%") 
                    
                if profit_rate >= 3 and current_price > up_Bol[len(up_Bol)-1] * 1.01 :
                    sell_order = upbit.sell_market_order(ticker, buyed_amount)
                    print(f"[!!목표가 달성!!]: [{ticker}] / 수익률: {profit_rate:.2f} / 현재가: {current_price:,.2f} / 시도 {attempts + 1} / {max_attempts}")
                    send_discord_message(f"[!!목표가 달성!!]: [{ticker}] / 수익률: {profit_rate:.2f} / 현재가: {current_price:,.2f} / 시도 {attempts + 1} / {max_attempts}")
                    return sell_order

                else:
                    time.sleep(second05)                                                                                                                            
                attempts += 1  # 조회 횟수 증가
                
            if profit_rate >= min_rate and df_high[0] > up_Bol[len(up_Bol)-1] * 0.99 and srsi_k_1[2] > 0.9:
                sell_order = upbit.sell_market_order(ticker, buyed_amount)
                send_discord_message(f"[매도시도 초과]: [{ticker}] 수익률: {profit_rate:.1f}% 현재가: {current_price:,.2f} \n 고가: {df_high[0]:,.2f} > 볼밴상단: {up_Bol[len(up_Bol)-1] * 0.99:,.2f} srsi: {srsi_k_1[2]:,.2f} > 0.9")
                print(f"[매도시도 초과]: [{ticker}] 수익률: {profit_rate:.1f}% 현재가: {current_price:,.2f} \n 고가: {df_high[0]:,.2f} > 볼밴상단: {up_Bol[len(up_Bol)-1] * 0.99:,.2f} srsi: {srsi_k_1[2]:,.2f} > 0.9")
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
                
                s_rsi = stoch_rsi(ticker, interval=minute5)
                sRSI_K = s_rsi['%K'].values

                if buyed_amount > 0:
                    profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산
                    report_message += f"[{b['currency']}] 수익률:{profit_rate:.1f}% s_RSI_K:{sRSI_K[2]:.2f}\n"

            send_discord_message(report_message)

        except (KeyError, ValueError) as e:          
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
            # print(balances)
            for b in balances:
                if b['currency'] not in ["KRW", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:
                        ticker = f"KRW-{b['currency']}"
                        trade_sell(ticker)
                time.sleep(second05)

        except Exception as e:
            print(f"selling_logic / 에러 발생: {e}")
            send_discord_message(f"selling_logic / 에러 발생: {e}")
            time.sleep(5)

def buying_logic():

    restricted_start_hour = 8
    restricted_start_minute = 30
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
                            time.sleep(second60)
                        else:
                            return None
                    else:
                        time.sleep(second60)

                else:
                    time.sleep(120)

        except (KeyError, ValueError) as e:
            print(f"buying_logic / 에러 발생: {e}")
            send_discord_message(f"buying_logic / 에러 발생: {e}")
            time.sleep(5)

def additional_buy_logic():
    while True:
        balances = upbit.get_balances()
        krw = get_balance("KRW")
        buy_size = min(trade, krw*0.9995)
        
        for b in balances:
            if b['currency'] not in ["KRW", "QI", "ONX", "ETHF", "ETHW", "PURSE"]:  # 특정 통화 제외
                ticker = f"KRW-{b['currency']}"  # 현재가 조회를 위한 티커 설정
                
                current_price = pyupbit.get_current_price(ticker)  # 현재가 조회
                avg_buy_price = upbit.get_avg_buy_price(b['currency']) 
                profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0 
            
                df = pyupbit.get_ohlcv(ticker, interval=minute5, count=4)
                time.sleep(second05)
                df_close = df['close'].values

                bands_df = get_bollinger_bands(ticker, interval=minute5)
                lower_band = bands_df['Lower_Band'].values
                upper_band = bands_df['Upper_Band'].values
                band_diff = upper_band - lower_band
                
                # is_increasing = all(lower_band[i] < lower_band[i + 1] for i in range(len(lower_band) - 1))
                is_increasing = all(band_diff[i] < band_diff[i + 1] for i in range(len(band_diff) - 1))

                # 볼린저 밴드의 하단값과 종가를 비교하여, 종가가 하단값 이하인 경우의 수를 센다
                count_below_lower_band = sum(1 for i in range(len(lower_band)) if df_close[i] < lower_band[i])
                
                # 종가가 볼린저 밴드의 하단값 이하인 경우가 2번 이상 발생하는지 확인
                lower_boliinger = count_below_lower_band >= bol_con
                # low_bollenger = any(lower_band[i + 1] >= df_close[i + 1] for i in range(len(band_diff) - 1))
                low_price = (df_close[len(band_diff) - 1] < current_price < lower_band[len(band_diff) - 1] * 1.005) or (current_price < lower_band[len(band_diff) - 1]*0.99)
                                
                stoch_Rsi = stoch_rsi(ticker, interval=minute5)
                srsi_k = stoch_Rsi['%K'].values
        
                if profit_rate < -2 and krw > 200_000 and is_increasing and lower_boliinger :   #-2% 미만, 조건1, 조건1
                    if low_price and srsi_k[2] < 0.3 :                      #조건3, sRSI 0.4미만
                        result = upbit.buy_market_order(ticker, buy_size)  # 추가 매수 실행

                        if result:
                            send_discord_message(f"추가 매수: {ticker} / 수익률: {profit_rate:,.1f} / 금액: {buy_size:,.0f} / lower_boliinger: {lower_boliinger} >= 2 / 0 < srsi1: {srsi_k[1]} < srsk2: {srsi_k[2]} < 0.3")
                            print(f"추가 매수: {ticker} / 수익률: {profit_rate:,.1f} / 금액: {buy_size:,.0f} / lower_boliinger: {lower_boliinger} >= 2 / 0 < srsi1: {srsi_k[1]} < srsk2: {srsi_k[2]} < 0.3")

                    else:
                        print(f'조건 미충족: {ticker} / 수익률 : {profit_rate:,.1f}')
            
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