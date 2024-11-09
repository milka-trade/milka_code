# def get_sma(ticker, window):
#     df = load_ohlcv(ticker)
#     return df['close'].rolling(window=window).mean().iloc[-1] if df is not None and not df.empty else 0

# def get_ema(ticker, window):
#     df = load_ohlcv(ticker)

#     if df is not None and not df.empty:
#         return df['close'].ewm(span=window, adjust=False).mean().iloc[-1]  # EMA 계산 후 마지막 값 반환
    
#     else:
#         return 0  # 데이터가 없으면 0 반환


# def get_rsi(ticker, period):
#     # df_rsi = pyupbit.get_ohlcv(ticker, interval="minute5", count=period)
#     df_rsi = load_ohlcv(ticker)
#     # df_rsi = pyupbit.get_ohlcv(ticker, interval="day", count=15)
#     delta = df_rsi['close'].diff(1)
#     gain = delta.where(delta > 0, 0).rolling(window=period).mean()
#     loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
#     rs = gain / loss
#     return 100 - (100 / (1 + rs)).iloc[-1]
# time.sleep(1)  # API 호출 제한을 위한 대기

# def get_stoch_rsi(ticker, rsi_period, stoch_period):
#     # RSI 계산
#     rsi = get_rsi(ticker, rsi_period)
    
#     # RSI의 최근 n일 데이터 가져오기
#     df_rsi = load_ohlcv(ticker)
#     rsi_values = df_rsi['close'].rolling(window=rsi_period).apply(lambda x: get_rsi(x, rsi_period), raw=False)
    
#     # 스토캐스틱 RSI 계산
#     min_rsi = rsi_values.rolling(window=stoch_period).min()
#     max_rsi = rsi_values.rolling(window=stoch_period).max()
    
#     # stoch_rsi = (rsi_values - min_rsi) / (max_rsi - min_rsi)
#     stoch_rsi = (rsi_values - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan)  # 0으로 나누는 경우를 방지하기 위해 np.nan으로 대체
    
#     return stoch_rsi if not stoch_rsi.empty else 0


# def get_rsi(ticker, period):
#     # df_rsi = pyupbit.get_ohlcv(ticker, interval="minute5", count=period)
#     df_rsi = load_ohlcv(ticker)
#     # df_rsi = pyupbit.get_ohlcv(ticker, interval="day", count=15)
#     delta = df_rsi['close'].diff(1)
#     gain = delta.where(delta > 0, 0).rolling(window=period).mean()
#     loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
#     rs = gain / loss
#     return 100 - (100 / (1 + rs)).iloc[-1]
# time.sleep(1)  # API 호출 제한을 위한 대기

# def get_sma(ticker, window):
#     df = load_ohlcv(ticker)
#     return df['close'].rolling(window=window).mean().iloc[-1] if df is not None and not df.empty else 0

# def get_ema(ticker, window):
#     df = load_ohlcv(ticker)

#     if df is not None and not df.empty:
#         return df['close'].ewm(span=window, adjust=False).mean().iloc[-1]  # EMA 계산 후 마지막 값 반환
    
#     else:
#         return 0  # 데이터가 없으면 0 반환
    
    
# def trade_sell(ticker):
#     """주어진 티커에 대해 매도 실행 및 수익률 출력, 매도 시간 체크"""
#     selltime = datetime.now()

#     """전량매도 : EC2 변환"""
#     # sell_start = selltime.replace(hour=23, minute=59, second=0, microsecond=0)    #EC2
#     # sell_end = selltime.replace(hour=0, minute=1, second=0, microsecond=0)      #EC2
#     sell_start = selltime.replace(hour=8, minute=58 , second=00, microsecond=0)       #VC
#     sell_end = selltime.replace(hour=8, minute=59, second=50, microsecond=0)         #VC

#     currency = ticker.split("-")[1]
#     buyed_amount = get_balance(currency)
#     avg_buy_price = upbit.get_avg_buy_price(currency)
#     current_price = pyupbit.get_current_price(ticker)
#     profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0  # 수익률 계산
    
#     max_attempts = 1000  # 최대 조회 횟수
#     attempts = 0  # 현재 조회 횟수
    
#     stoch_rsi = get_rsi_and_stoch_rsi(ticker, 14, 14)   #스토캐스틱 RSI 계산
#     last_stoch_rsi = stoch_rsi.iloc[-1]

    # if sell_start <= selltime <= sell_end:      # 매도 제한시간이면
    #     sell_order = upbit.sell_market_order(ticker, buyed_amount)  # 시장가로 전량 매도
    #     send_discord_message(f"전량 매도: {ticker}, 현재가 {current_price} 수익률 {profit_rate:.2f}%")
    #     return sell_order           
    
    # else:
    # if profit_rate >= 0.5:  
    #     while attempts < max_attempts:
    #         current_price = pyupbit.get_current_price(ticker)  # 현재 가격 재조회
    #         profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0
                
    #         print(f"{ticker} / 시도 {attempts + 1} / {max_attempts} - / 현재가 {current_price} 수익률 {profit_rate:.2f}%")
                
    #         if profit_rate >= 1.0:
    #             sell_order = upbit.sell_market_order(ticker, buyed_amount)
    #             send_discord_message(f"매도: {ticker}/ 현재가: {current_price}/ 수익률: {profit_rate:.2f}%")
    #             return sell_order
    #         else:
    #             time.sleep(2.5)  # 짧은 대기                
    #         attempts += 1  # 조회 횟수 증가
            
    #     last_stoch_rsi = stoch_rsi.iloc[-1]
    #     if last_stoch_rsi < 0.3 :
    #         return None   
    #     else:
    #         sell_order = upbit.sell_market_order(ticker, buyed_amount)
    #         send_discord_message(f"최종 매도: {ticker}/ 현재가: {current_price}/ 수익률: {profit_rate:.2f}% / s_RSI: {last_stoch_rsi:,.2f}")
    #         return sell_order

    # return None


                                        # ai_decision = get_ai_decision(t)  
                                        # send_discord_message(f"{t} / AI: {ai_decision}")
                                        # if ai_decision == "BUY" :



# def get_current_price(ticker):
#     """현재가를 조회합니다."""
#     if not ticker.startswith("KRW-"):
#         print(f"current_price/잘못된 티커 형식: {ticker}")
#         send_discord_message(f"current_price/잘못된 티커 형식: {ticker}")
#         return None
    
#     try:
#         orderbook = pyupbit.get_orderbook(ticker=ticker)
#         if orderbook is None or "orderbook_units" not in orderbook or not orderbook["orderbook_units"]:
#             raise ValueError(f"'{ticker}'에 대한 유효한 orderbook이 없습니다.")
#         current_price = orderbook["orderbook_units"][0]["ask_price"]
#         time.sleep(0.5)
#         return current_price
    
#     except Exception as e:
#         print(f"current_price/현재가 조회 오류 ({ticker}): {e}")
#         send_discord_message(f"current_price/현재가 조회 오류 ({ticker}): {e}")
#         time.sleep(1)
#         return None