import time
import pyupbit
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import requests
import threading

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("discord_webhhok")
upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS"), os.getenv("UPBIT_SECRET"))

def send_discord_message(msg):
    """discord 메시지 전송"""
    try:
        message ={"content":msg}
        requests.post(DISCORD_WEBHOOK_URL, data=message)
    except Exception as e:
        print(f"디스코드 메시지 전송 실패 : {e}")
        time.sleep(5) 

count_200 = 200

rsi_buy_s = 25
rsi_buy_e = 45
rsi_sell_s = 65
rsi_sell_e = 80

def get_user_input():
    while True:
        try:
            min_rate = float(input("최소 수익률 (예: 1.1): "))
            max_rate = float(input("최대 수익률 (예: 5.0): "))
            sell_time = int(input("매도감시횟수 (예: 20): "))
            break
        except ValueError:
            print("잘못된 입력입니다. 다시 시도하세요.")

    return min_rate, sell_time, max_rate  
# 함수 호출 및 결과 저장
min_rate, sell_time, max_rate = get_user_input() 

second = 1.0
min_krw = 10_000
cut_rate = -3.0

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

def calculate_rsi(closes, period=14):
    """RSI (Relative Strength Index) 계산"""
    if len(closes) < period + 1:
        return 50.0
    
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    for i in range(period, len(closes)-1):
        avg_gain = (avg_gain * (period-1) + gains[i]) / period
        avg_loss = (avg_loss * (period-1) + losses[i]) / period
    
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))

def calculate_ema(closes, period=12):
    """EMA (Exponential Moving Average) 계산"""
    if len(closes) < period:
        return closes[-1]
    
    ema = [closes[0]]
    alpha = 2 / (period + 1)
    
    for close in closes[1:]:
        ema.append(alpha * close + (1 - alpha) * ema[-1])
    
    return ema[-1]

def calculate_bb(closes, window=20, std_dev=2.0):
    """볼린저 밴드 계산"""
    if len(closes) < window:
        window = len(closes)
    
    sma = np.mean(closes[-window:])
    std = np.std(closes[-window:])
    
    lower = sma - (std * std_dev)
    upper = sma + (std * std_dev)
    
    position = (closes[-1] - lower) / (upper - lower + 1e-8)
    width = (upper - lower) / sma * 100
    
    return lower, sma, upper, max(0, min(1, position)), width

def trade_buy(ticker=None):
    """
    🚀 초단기 복리 매수 시스템 v7.0 - BB 하단→상단 전략 완화
    
    핵심 개선:
    1. 필터링 대폭 완화 (반등 60%, 점수 45점)
    2. BB 하단 기준 완화 (30% 미만)
    3. MACD, Stochastic 보조지표 추가
    4. 15분봉 조건 70% 미만으로 완화
    5. API 호출 최적화 (0.05초)
    
    목표: 10만원 → 10억 (2년, 2% 복리)
    """
    import numpy as np
    import time
    import pyupbit # [추가] pyupbit을 최상단에서 import
    
    # ==================== 내부 함수 ====================
    
    def get_krw_balance():
        """KRW 잔고"""
        try:
            balances = upbit.get_balances()
            for b in balances:
                if b['currency'] == "KRW":
                    return float(b['balance'])
        except:
            pass
        return 0.0
    
    def get_total_crypto_value():
        """암호화폐 평가액"""
        try:
            balances = upbit.get_balances()
            total = 0.0
            for balance in balances:
                if balance['currency'] == 'KRW':
                    continue
                amount = float(balance['balance'])
                if amount > 0:
                    ticker_name = f"KRW-{balance['currency']}"
                    try:
                        price = pyupbit.get_current_price(ticker_name)
                        if price:
                            total += amount * price
                    except:
                        continue
            return total
        except:
            return 0.0
    
    def get_held_coins():
        """보유 코인 목록"""
        try:
            balances = upbit.get_balances()
            return {f"KRW-{b['currency']}" for b in balances
                    if float(b.get('balance', 0)) > 0 and b['currency'] != 'KRW'}
        except:
            return set()
    
    def calculate_rsi(prices, period=14):
        """RSI"""
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # [수정] 배열 슬라이싱에 필요한 최소 길이를 충족하는지 확인
        if len(gains) < period:
            return 50.0 
            
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bb(prices, window=20):
        """볼린저밴드"""
        if len(prices) < window:
            return None, None, None, 0.5, 0.0
        sma = np.mean(prices[-window:])
        std = np.std(prices[-window:])
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        
        # [수정] prices[-1] 접근 가능성 체크
        if len(prices) < 1:
            return None, None, None, 0.5, 0.0
            
        current = prices[-1]
        if upper == lower:
            position = 0.5
        else:
            position = (current - lower) / (upper - lower)
        width = (std * 4) / sma * 100 if sma > 0 else 0
        return lower, sma, upper, position, width
    
    def calculate_ema(prices, period):
        """EMA"""
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0 # [수정] 빈 배열 예외처리
        multiplier = 2 / (period + 1)
        
        # [수정] 초기 EMA 계산을 위한 충분한 데이터 확인
        if len(prices[:period]) < period:
            return prices[-1]
            
        ema = np.mean(prices[:period])
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        return ema
    
    def calculate_macd(closes):
        """MACD 계산"""
        if len(closes) < 35: # MACD Signal line 계산에 필요한 최소 35개 봉
            return None, None, None, False
        
        # calculate_ema 내부에서 closes를 계속 슬라이싱하므로 closes의 최소 길이를 보장해야 함.
        ema_12 = calculate_ema(closes, 12)
        ema_26 = calculate_ema(closes, 26)
        macd_line = ema_12 - ema_26
        
        # Signal line (MACD의 9일 EMA)
        macd_values = []
        for i in range(26, len(closes)):
            #closes[:i+1]이 최소 27개부터 시작
            e12 = calculate_ema(closes[:i+1], 12)
            e26 = calculate_ema(closes[:i+1], 26)
            macd_values.append(e12 - e26)
        
        # macd_values는 최소 9개 (35-26)이므로 계산 가능
        signal_line = calculate_ema(np.array(macd_values), 9)
        
        histogram = macd_line - signal_line
        
        # 골든크로스: MACD > Signal and 이전에는 MACD < Signal
        is_golden_cross = macd_line > signal_line and histogram > 0
        
        return macd_line, signal_line, histogram, is_golden_cross
    
    def calculate_stochastic(df, period=14):
        """Stochastic Oscillator"""
        if len(df) < period + 2: # %D 계산을 위해 최소 period + 2개 필요
            return 50.0, 50.0, False
        
        recent = df.iloc[-period:]
        lowest_low = recent['low'].min()
        highest_high = recent['high'].max()
        current_close = df['close'].iloc[-1]
        
        if highest_high == lowest_low:
            k = 50.0
        else:
            k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # %D는 %K의 3일 이동평균
        k_values = []
        for i in range(-3, 0):
            # i-period+1이 유효한 인덱스 범위를 벗어나지 않도록 주의 (len(df) 검사로 해결)
            segment = df.iloc[i-period+1:i+1]
            
            # [수정] segment의 길이가 period를 보장하지 않으면 건너뛰기
            if len(segment) < period:
                continue

            ll = segment['low'].min()
            hh = segment['high'].max()
            cc = segment['close'].iloc[-1] # segment가 비어있지 않음이 보장됨
            if hh != ll:
                k_values.append(((cc - ll) / (hh - ll)) * 100)
        
        d = np.mean(k_values) if k_values else k
        
        # 과매도: K < 20
        is_oversold = k < 20
        
        return k, d, is_oversold
    
    def analyze_price_momentum(closes):
        """가격 모멘텀 분석 (간소화)"""
        if len(closes) < 10:
            return None
        
        recent = np.mean(closes[-3:])
        middle = np.mean(closes[-6:-3])
        older = np.mean(closes[-9:-6])
        
        v1 = (middle - older) / older
        v2 = (recent - middle) / middle
        accel = v2 - v1
        
        # 하락 둔화 or 상승 전환
        is_good = (
            (v2 < 0 and v1 < 0 and accel > 0) or  # 하락 둔화
            (v2 > 0 and v1 < 0)  # 상승 전환
        )
        
        return {
            'is_favorable': is_good,
            'velocity': v2,
            'acceleration': accel
        }
    
    def calculate_position_size(total_asset, krw_balance, rebound_prob):
        """포지션 사이징"""
        krw_ratio = krw_balance / total_asset if total_asset > 0 else 0
        
        # 원화 10% 미만이면 전량
        if krw_ratio < 0.10:
            return krw_balance * 0.995
        
        # 켈리 공식
        win_rate = rebound_prob
        kelly = (2 * win_rate - (1 - win_rate)) / 2
        
        if kelly <= 0:
            return 0.0
        
        # 단계별 공격성
        if total_asset < 1_000_000:
            aggression, max_ratio = 2.5, 0.60
        elif total_asset < 5_000_000:
            aggression, max_ratio = 2.0, 0.50
        elif total_asset < 10_000_000:
            aggression, max_ratio = 1.5, 0.40
        else:
            aggression, max_ratio = 1.2, 0.30
        
        adjusted_kelly = kelly * aggression
        base = total_asset * adjusted_kelly
        max_pos = total_asset * max_ratio
        avail = krw_balance * 0.995
        
        return min(base, max_pos, avail)
    
    def analyze_ticker_fast(ticker_symbol):
        """빠른 종목 분석 (최적화)"""
        try:
            
            # 필수 데이터만 수집
            # MACD signal line 계산에 필요한 최소 35개 봉을 기준으로 함
            MIN_ROWS_5M = 35 
            MIN_ROWS_15M = 20
            
            df_5m = pyupbit.get_ohlcv(ticker_symbol, interval="minute5", count=50)
            time.sleep(0.05)
            df_15m = pyupbit.get_ohlcv(ticker_symbol, interval="minute15", count=30)
            time.sleep(0.05)
            df_1d = pyupbit.get_ohlcv(ticker_symbol, interval="day", count=3)
            time.sleep(0.05)
            current_price = pyupbit.get_current_price(ticker_symbol)
            
            # 🚨 [수정된 부분] None 검사 및 최소 데이터 개수 검사
            if (df_5m is None or len(df_5m) < MIN_ROWS_5M or
                df_15m is None or len(df_15m) < MIN_ROWS_15M or
                df_1d is None or len(df_1d) < 3 or 
                current_price is None):
                return {'valid': False}
            
            c_5m = df_5m['close'].values
            v_5m = df_5m['volume'].values
            c_15m = df_15m['close'].values
            
            # 기본 지표
            rsi_5m = calculate_rsi(c_5m, 14)
            
            # BB (가장 중요!)
            _, _, _, bb_5m_pos, bb_5m_width = calculate_bb(c_5m, 20)
            _, _, _, bb_15m_pos, _ = calculate_bb(c_15m, 20)
            
            # [추가] BB 계산 결과 유효성 재확인
            if bb_5m_pos is None or bb_15m_pos is None:
                return {'valid': False}
                
            # 보조 지표
            macd_line, signal_line, histogram, is_golden = calculate_macd(c_5m)
            stoch_k, stoch_d, is_oversold = calculate_stochastic(df_5m, 14)
            momentum = analyze_price_momentum(c_5m)
            
            # [추가] MACD 계산 결과 유효성 재확인
            if macd_line is None:
                return {'valid': False}

            # 거래량
            vol_recent = np.mean(v_5m[-3:])
            vol_normal = np.mean(v_5m[-15:-3])
            vol_ratio = vol_recent / (vol_normal + 1e-8)
            vol_krw = vol_recent * current_price
            
            # 일봉
            daily_open = df_1d['open'].iloc[-1]
            daily_prev = df_1d['close'].iloc[-2]
            intraday_change = (current_price - daily_open) / daily_open * 100
            
            # ========== 신호 점수 (간소화) ==========
            score = 0
            signals = []
            
            # [1] BB 하단 (40점) ⭐최우선⭐
            if bb_5m_pos < 0.15:
                score += 40
                signals.append(f"BB극하단({bb_5m_pos*100:.0f}%)")
            elif bb_5m_pos < 0.25:
                score += 30
                signals.append(f"BB하단({bb_5m_pos*100:.0f}%)")
            elif bb_5m_pos < 0.35:
                score += 20
                signals.append("BB하단근처")
            
            # [2] RSI 과매도 (20점)
            if rsi_5m < 25:
                score += 20
                signals.append(f"RSI극과매도({rsi_5m:.0f})")
            elif rsi_5m < 30:
                score += 15
                signals.append(f"RSI과매도({rsi_5m:.0f})")
            elif rsi_5m < 35:
                score += 10
            
            # [3] MACD 골든크로스 (15점)
            if is_golden and histogram > 0:
                score += 15
                signals.append("MACD골든크로스")
            elif macd_line > signal_line:
                score += 8
            
            # [4] Stochastic 과매도 (10점)
            if is_oversold and stoch_k < 15:
                score += 10
                signals.append(f"Stoch과매도({stoch_k:.0f})")
            elif is_oversold:
                score += 6
            
            # [5] 모멘텀 (10점)
            if momentum and momentum['is_favorable']:
                score += 10
                signals.append("모멘텀양호")
            
            # [6] 거래량 (5점)
            if vol_krw >= 100_000_000 and vol_ratio >= 1.3:
                score += 5
                signals.append(f"거래량({vol_ratio:.1f}x)")
            
            # ========== 반등 확률 (간소화) ==========
            prob_score = 0
            
            if bb_5m_pos < 0.20:
                prob_score += 40
            elif bb_5m_pos < 0.30:
                prob_score += 30
            elif bb_5m_pos < 0.40:
                prob_score += 20
            
            if rsi_5m < 25:
                prob_score += 20
            elif rsi_5m < 30:
                prob_score += 15
            elif rsi_5m < 35:
                prob_score += 10
            
            if bb_15m_pos < 0.30:
                prob_score += 20
            elif bb_15m_pos < 0.50:
                prob_score += 15
            elif bb_15m_pos < 0.70:
                prob_score += 10
            
            if is_golden:
                prob_score += 10
            if is_oversold:
                prob_score += 10
            
            rebound_prob = min(prob_score / 100, 0.95)
            
            return {
                'valid': True,
                'current_price': current_price,
                'rebound_prob': rebound_prob,
                'signal_score': score,
                'indicators': {
                    'rsi_5m': rsi_5m,
                    'bb_5m_pos': bb_5m_pos,
                    'bb_5m_width': bb_5m_width,
                    'bb_15m_pos': bb_15m_pos,
                    'vol_ratio': vol_ratio,
                    'vol_krw': vol_krw,
                    'intraday_change': intraday_change,
                    'macd_golden': is_golden,
                    'stoch_oversold': is_oversold,
                    'momentum': momentum
                },
                'signals': signals
            }
            
        except Exception as e:
            # print(f"분석 오류 ({ticker_symbol}): {e}") # 디버깅을 위해 주석 해제 가능
            return {'valid': False}
    
    # ==================== 메인 로직 ====================
    
    print("\n" + "="*60)
    print("🚀 v7.0 - BB 하단→상단 전략 (완화)")
    print("="*60)
    
    # 자산 현황
    krw_balance = get_krw_balance()
    crypto_value = get_total_crypto_value()
    total_asset = krw_balance + crypto_value
    
    print(f"\n💰 자산: {total_asset:,.0f}원")
    print(f"   KRW: {krw_balance:,.0f}원 ({krw_balance/total_asset*100:.1f}%)")
    print(f"   코인: {crypto_value:,.0f}원 ({crypto_value/total_asset*100:.1f}%)")
    
    MIN_ORDER = 5000
    if krw_balance < MIN_ORDER:
        print("❌ 잔고 부족")
        return "Insufficient balance", None
    
    # 종목 선정
    if ticker is None:
        print("\n🔍 종목 스캔...")
        
        COINS = [
            "KRW-BTC","KRW-ETH","KRW-XRP","KRW-SOL","KRW-DOGE",
            "KRW-TRX","KRW-ADA","KRW-LINK","KRW-AVAX","KRW-XLM",
            "KRW-SUI","KRW-BCH","KRW-HBAR","KRW-SHIB","KRW-DOT",
            "KRW-UNI","KRW-AAVE","KRW-PEPE","KRW-NEAR","KRW-APT"
        ]
        
        held = get_held_coins()
        candidates = [t for t in COINS if t not in held]
        
        if not candidates:
            return "No candidates", None
        
        print(f"   대상: {len(candidates)}개")
        
        viable = []
        
        for t in candidates:
            analysis = analyze_ticker_fast(t)
            
            if not analysis['valid']:
                continue
            
            ind = analysis['indicators']
            score = analysis['signal_score']
            prob = analysis['rebound_prob']
            
            # ========== 강화된 필터 (빈도 조절) ==========
            # 1. 15분봉 60% 미만 (70%에서 강화)
            if ind['bb_15m_pos'] >= 0.60:
                continue
            
            # 2. 반등 확률 65% 이상 (60%에서 강화)
            if prob < 0.65:
                continue
            
            # 3. 일봉 ±2% 이내 (3%에서 강화)
            if abs(ind['intraday_change']) > 2.0:
                continue
            
            # 4. 가격 범위
            if not (50 <= analysis['current_price'] <= 200000):
                continue
            
            # 5. 거래량 8천만 (5천만에서 강화)
            if ind['vol_krw'] < 80_000_000:
                continue
            
            # 6. 신호 점수 50점 이상 (45점에서 강화)
            # 7. BB 5분봉 30% 미만 필수 (하단 확실히)
            if score >= 50 and ind['bb_5m_pos'] < 0.30:
                viable.append({
                    'ticker': t,
                    'score': score,
                    'prob': prob,
                    'signals': analysis['signals'],
                    'analysis': analysis
                })
                print(f"   ✓ {t}: {score}점 | {prob*100:.0f}% | BB{ind['bb_5m_pos']*100:.0f}% | {analysis['signals'][:2]}")
            
            time.sleep(0.03)
        
        print(f"\n📊 후보: {len(viable)}개")
        
        if not viable:
            return "No viable candidates", None
        
        # 최고 종목 선택
        viable.sort(key=lambda x: (x['prob'], x['score']), reverse=True)
        best = viable[0]
        
        selected_ticker = best['ticker']
        selected_analysis = best['analysis']
        selected_score = best['score']
        selected_prob = best['prob']
        selected_signals = best['signals']
        
        print(f"\n🎯 선택: {selected_ticker}")
        print(f"   점수: {selected_score}점 | 확률: {selected_prob*100:.0f}%")
        print(f"   시그널: {', '.join(selected_signals[:3])}")
        
    else:
        # 특정 종목
        selected_analysis = analyze_ticker_fast(ticker)
        
        if not selected_analysis['valid']:
            return "Data failed", None
        
        selected_ticker = ticker
        selected_score = selected_analysis['signal_score']
        selected_prob = selected_analysis['rebound_prob']
        selected_signals = selected_analysis['signals']
    
    # ========== 최종 검증 ==========
    
    ind = selected_analysis['indicators']
    current_price = selected_analysis['current_price']
    
    print(f"\n📈 지표")
    print(f"   RSI: {ind['rsi_5m']:.0f}")
    print(f"   BB: 5m={ind['bb_5m_pos']*100:.0f}% | 15m={ind['bb_15m_pos']*100:.0f}%")
    print(f"   폭: {ind['bb_5m_width']:.1f}%")
    print(f"   거래량: {ind['vol_ratio']:.1f}x ({ind['vol_krw']/1e8:.1f}억)")
    
    if ind.get('macd_golden'):
        print(f"   MACD: 골든크로스 ✓")
    if ind.get('stoch_oversold'):
        print(f"   Stoch: 과매도 ✓")
    
    # 안전 검증 (완화)
    safety = {
        'BB하단': ind['bb_5m_pos'] < 0.40,  # 40% 미만
        'RSI과매도': ind['rsi_5m'] < 45,  # 45 미만
        '15분봉': ind['bb_15m_pos'] < 0.70,  # 70% 미만
        '반등확률': selected_prob >= 0.60,  # 60% 이상
        '일간변동': abs(ind['intraday_change']) <= 3.0
    }
    
    passed = sum(safety.values())
    
    print(f"\n🛡️ 안전성: {passed}/5")
    for k, v in safety.items():
        print(f"   {'✓' if v else '✗'} {k}")
    
    # 최종 매수 조건 (대폭 완화!)
    can_buy = (
        selected_score >= 45 and  # 45점 이상
        selected_prob >= 0.60 and  # 60% 이상
        ind['bb_15m_pos'] < 0.70 and  # 15분봉 70% 미만
        passed >= 4  # 5개 중 4개
    )
    
    print(f"\n{'🟢 매수 GO!' if can_buy else '🔴 조건 미달'}")
    print(f"   점수: {selected_score}/45 | 확률: {selected_prob*100:.0f}%/60% | 안전: {passed}/4")
    
    if not can_buy:
        return "Conditions not met", None
    
    # 포지션
    buy_size = calculate_position_size(total_asset, krw_balance, selected_prob)
    
    if buy_size < MIN_ORDER:
        return "Size too small", None
    
    print(f"\n💵 매수액: {buy_size:,.0f}원 ({buy_size/total_asset*100:.1f}%)")
    
    # 🚀 매수 실행
    for attempt in range(1, 4):
        try:
            
            verify_price = pyupbit.get_current_price(selected_ticker)
            time.sleep(0.05)
            
            price_change = (verify_price - current_price) / current_price
            
            if price_change > 0.03:
                print(f"⚠️ 급등 감지 (+{price_change*100:.1f}%)")
                time.sleep(2)
                continue
            
            buy_order = upbit.buy_market_order(selected_ticker, buy_size)
            
            grade = "🏆 PERFECT" if selected_score >= 70 else "⭐ EXCELLENT" if selected_score >= 60 else "✨ STRONG"
            
            msg = f"{grade} 매수!\n"
            msg += f"{selected_ticker} | {verify_price:,.0f}원 | {buy_size:,.0f}원\n"
            msg += f"점수{selected_score} | 확률{selected_prob*100:.0f}%\n"
            msg += f"BB{ind['bb_5m_pos']*100:.0f}% | RSI{ind['rsi_5m']:.0f}\n"
            msg += f"자산: {total_asset:,.0f}원"
            
            print(f"\n✅ {msg}")
            
            try:
                # send_discord_message 함수가 정의되어 있어야 함
                send_discord_message(msg)
            except:
                pass
            
            return buy_order
            
        except Exception as e:
            print(f"❌ 오류 ({attempt}/3): {e}")
            if attempt < 3:
                time.sleep(2)
            else:
                try:
                    # send_discord_message 함수가 정의되어 있어야 함
                    send_discord_message(f"매수 실패: {selected_ticker}\n{e}")
                except:
                    pass
                return "Order failed", None
    
    return "Max attempts", None

def trade_sell(ticker):
    """지능형 적응형 매도 시스템 v2.0 - BB 기반 + 폭락 예측"""
    import numpy as np
    import time
    
    # ==================== 내부 함수 정의 ====================
    
    def calculate_recovery_probability(df, current_price, avg_buy_price):
        """반등 확률 계산"""
        if df is None or len(df) < 20:
            return 0.3
        
        closes = df['close'].values
        recovery_count = 0
        similar_situations = 0
        current_drop = (current_price - avg_buy_price) / avg_buy_price
        
        for i in range(10, len(closes) - 5):
            period_drop = (closes[i] - closes[i-5]) / closes[i-5]
            if abs(period_drop - current_drop) < 0.01:
                similar_situations += 1
                if closes[i+5] > closes[i]:
                    recovery_count += 1
        
        if similar_situations < 3:
            return 0.4
        
        return recovery_count / similar_situations
    
    def analyze_crash_probability(df_5m, df_15m, current_price, avg_buy_price, profit_rate):
        """폭락 확률 예측 시스템"""
        if df_5m is None or len(df_5m) < 30:
            return None
        
        closes_5m = df_5m['close'].values
        volumes_5m = df_5m['volume'].values
        lows_5m = df_5m['low'].values
        
        score = 0
        max_score = 100
        factors = []
        
        # [1] 급락 가속도 (25점)
        recent_3 = np.mean(closes_5m[-3:])
        middle_3 = np.mean(closes_5m[-6:-3])
        older_3 = np.mean(closes_5m[-9:-6])
        
        velocity_1 = (middle_3 - older_3) / older_3
        velocity_2 = (recent_3 - middle_3) / middle_3
        acceleration = velocity_2 - velocity_1
        
        if acceleration < -0.02:
            score += 25
            factors.append(f"급락가속({acceleration*100:.1f}%)")
        elif acceleration < -0.01:
            score += 15
            factors.append(f"하락가속({acceleration*100:.1f}%)")
        elif velocity_2 < -0.03:
            score += 10
            factors.append(f"급락중({velocity_2*100:.1f}%)")
        
        # [2] 거래량 폭증 + 하락 (20점)
        vol_recent = np.mean(volumes_5m[-3:])
        vol_normal = np.mean(volumes_5m[-15:-3])
        vol_ratio = vol_recent / (vol_normal + 1e-8)
        price_change_recent = (closes_5m[-1] - closes_5m[-3]) / closes_5m[-3]
        
        if vol_ratio > 2.0 and price_change_recent < -0.02:
            score += 20
            factors.append(f"공포매도(거래량{vol_ratio:.1f}x)")
        elif vol_ratio > 1.5 and price_change_recent < -0.01:
            score += 12
            factors.append("매도압력증가")
        
        # [3] BB 분석 (20점)
        bb_lower, bb_mid, bb_upper, bb_pos, bb_width = calculate_bb(closes_5m, 20)
        
        if bb_pos < -0.1:
            score += 15
            factors.append(f"BB하단이탈({bb_pos*100:.0f}%)")
        elif bb_pos < 0:
            score += 8
            factors.append("BB하단근접")
        
        if bb_width > 8.0:
            score += 5
            factors.append(f"고변동성(BB{bb_width:.1f}%)")
        
        # [4] RSI 급락 (15점)
        rsi = calculate_rsi(closes_5m, 14)
        
        if rsi < 20:
            score += 15
            factors.append(f"RSI극과매도({rsi:.0f})")
        elif rsi < 25:
            score += 10
            factors.append(f"RSI과매도({rsi:.0f})")
        elif rsi < 30:
            score += 5
        
        # [5] 15분봉 하락 (15점)
        if df_15m is not None and len(df_15m) >= 10:
            closes_15m = df_15m['close'].values
            trend_15m = (closes_15m[-1] - closes_15m[-5]) / closes_15m[-5]
            
            if trend_15m < -0.05:
                score += 15
                factors.append(f"15분봉급락({trend_15m*100:.1f}%)")
            elif trend_15m < -0.03:
                score += 8
                factors.append("15분봉하락세")
        
        # [6] 지지선 붕괴 (5점)
        support_level = np.min(lows_5m[-20:-3])
        current_low = lows_5m[-1]
        
        if current_low < support_level * 0.98:
            score += 5
            factors.append("지지선붕괴")
        
        # 확률 계산
        probability = min(score / max_score, 0.95)
        
        # 위험도 등급
        if probability >= 0.70:
            risk_level = 'CRITICAL'
        elif probability >= 0.55:
            risk_level = 'HIGH'
        elif probability >= 0.40:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # 손절 권장
        should_cut = (probability >= 0.70 and profit_rate < 0)
        
        return {
            'crash_probability': probability,
            'risk_level': risk_level,
            'factors': factors,
            'should_cut': should_cut,
            'score': score,
            'acceleration': acceleration,
            'vol_ratio': vol_ratio,
            'rsi': rsi
        }
    
    def analyze_bb_sell_signal(current_price, closes, volumes):
        """BB 기반 매도 신호 분석"""
        if len(closes) < 20:
            return None
        
        # BB 계산
        bb_lower, bb_mid, bb_upper, bb_position, bb_width = calculate_bb(closes, 20)
        
        # current_price 명시적 사용
        price_to_mid_ratio = (current_price - bb_mid) / bb_mid if bb_mid > 0 else 0
        
        # 추가 지표
        rsi = calculate_rsi(closes, 14)
        
        # 거래량 추세
        vol_recent = np.mean(volumes[-3:])
        vol_normal = np.mean(volumes[-10:-3])
        vol_surge = vol_recent / (vol_normal + 1e-8) > 1.5
        
        # 가격 모멘텀
        price_momentum = (closes[-1] - closes[-5]) / closes[-5]
        
        # BB 위치별 판단
        if bb_position >= 0.70:
            urgency = 'HIGH'
            should_hold = False
            reason = f"BB상단{bb_position*100:.0f}%(과열)"
            
            if rsi > 70:
                urgency = 'CRITICAL'
                reason += f"+RSI{rsi:.0f}"
        
        elif bb_position >= 0.50:
            urgency = 'MEDIUM'
            should_hold = False
            
            if price_momentum > 0.01 and rsi < 65:
                should_hold = True
                reason = f"BB중상단{bb_position*100:.0f}%+상승추세"
            else:
                reason = f"BB중상단{bb_position*100:.0f}%"
        
        elif bb_position >= 0.30:
            urgency = 'LOW'
            
            if price_momentum < -0.01:
                should_hold = False
                reason = f"BB중단{bb_position*100:.0f}%+하락"
            else:
                should_hold = True
                reason = f"BB중단{bb_position*100:.0f}%+상승여력"
        
        else:
            urgency = 'NONE'
            should_hold = True
            reason = f"BB하단{bb_position*100:.0f}%(상승여력)"
            
            if price_momentum < -0.03 and vol_surge:
                urgency = 'MEDIUM'
                should_hold = False
                reason = f"BB하단+급락중({price_momentum*100:.1f}%)"
        
        return {
            'bb_position': bb_position,
            'bb_width': bb_width,
            'rsi': rsi,
            'sell_urgency': urgency,
            'should_hold': should_hold,
            'reason': reason,
            'momentum': price_momentum
        }
    
    # ==================== 메인 로직 시작 ====================
    
    currency = ticker.split("-")[1]
    
    try:
        buyed_amount = get_balance(currency)
        if buyed_amount <= 0:
            return None
        
        avg_buy_price = upbit.get_avg_buy_price(currency)
        cur_price = pyupbit.get_current_price(ticker)
        if cur_price is None:
            return None
        
        profit_rate = (cur_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0
        
    except Exception as e:
        print(f"[{ticker}] 초기 정보 조회 오류: {e}")
        return None
    
    # ==================== 최소수익률 미달 처리 ====================
    
    if profit_rate < min_rate:
        # print(f"[{ticker}] 최소수익률({min_rate}%) 미달 | 현재: {profit_rate:.2f}%")
        
        # 손실 구간 폭락 확률 체크
        if profit_rate < 0:
            df_15m_loss = pyupbit.get_ohlcv(ticker, interval="minute15", count=30)
            time.sleep(0.1)
            df_5m_loss = pyupbit.get_ohlcv(ticker, interval="minute5", count=50)
            time.sleep(0.1)
            
            if df_5m_loss is not None and len(df_5m_loss) >= 30:
                crash_analysis = analyze_crash_probability(
                    df_5m_loss, df_15m_loss, cur_price, avg_buy_price, profit_rate
                )
                
                if crash_analysis:
                    # print(f"🚨 폭락위험: {crash_analysis['crash_probability']*100:.0f}% ({crash_analysis['risk_level']})")
                    # print(f"   요인: {', '.join(crash_analysis['factors'][:3])}")
                    
                    # -3% 이상 손실 + 폭락 70% 이상 → 손절
                    if profit_rate <= -3.0 and crash_analysis['should_cut']:
                        sell_order = upbit.sell_market_order(ticker, buyed_amount)
                        msg = f"🛑 **[지능형손절]**: [{ticker}] {profit_rate:.2f}% / {cur_price:,.1f}원\n"
                        msg += f"폭락확률: {crash_analysis['crash_probability']*100:.0f}%\n"
                        msg += f"요인: {', '.join(crash_analysis['factors'])}"
                        print(msg)
                        send_discord_message(msg)
                        return sell_order
                    
                    # -5% 이상 손실 + 폭락 55% 이상 → 손절
                    elif profit_rate <= -5.0 and crash_analysis['crash_probability'] >= 0.55:
                        sell_order = upbit.sell_market_order(ticker, buyed_amount)
                        msg = f"🚨 **[긴급손절]**: [{ticker}] {profit_rate:.2f}% / {cur_price:,.1f}원\n"
                        msg += f"폭락확률: {crash_analysis['crash_probability']*100:.0f}%"
                        print(msg)
                        send_discord_message(msg)
                        return sell_order
        
        # 기존 긴급 탈출선 (백업)
        emergency_cut = cut_rate - 1.0
        if profit_rate < emergency_cut:
            df_30m = pyupbit.get_ohlcv(ticker, interval="minute30", count=10)
            time.sleep(0.1)
            if df_30m is not None and len(df_30m) >= 5:
                recent_trend = (df_30m['close'].iloc[-1] - df_30m['close'].iloc[-5]) / df_30m['close'].iloc[-5]
                if recent_trend < -0.05:
                    sell_order = upbit.sell_market_order(ticker, buyed_amount)
                    msg = f"🚨 **[긴급탈출]**: [{ticker}] {profit_rate:.2f}% / {cur_price:,.1f}원"
                    print(msg)
                    send_discord_message(msg)
                    return sell_order
        
        return None
    
    # ==================== 데이터 수집 및 분석 ====================
    
    df_5m = pyupbit.get_ohlcv(ticker, interval="minute5", count=50)
    time.sleep(0.1)
    df_15m = pyupbit.get_ohlcv(ticker, interval="minute15", count=30)
    time.sleep(0.1)
    
    if df_5m is None or len(df_5m) < 30:
        print(f"[{ticker}] 데이터 부족")
        return None
    
    closes = df_5m['close'].values
    volumes = df_5m['volume'].values
    current_rsi = calculate_rsi(closes)
    
    # 폭락 위험 분석
    crash_analysis = analyze_crash_probability(df_5m, df_15m, cur_price, avg_buy_price, profit_rate)
    
    if crash_analysis:
        print(f"📊 폭락위험: {crash_analysis['crash_probability']*100:.0f}% ({crash_analysis['risk_level']})")
        if crash_analysis['factors']:
            print(f"   {', '.join(crash_analysis['factors'][:3])}")
    
    # 수익 구간 폭락 위험 조기 매도
    if crash_analysis and crash_analysis['risk_level'] == 'CRITICAL':
        if min_rate <= profit_rate < min_rate * 1.3:
            sell_order = upbit.sell_market_order(ticker, buyed_amount)
            msg = f"⚠️ **[폭락위험조기매도]**: [{ticker}] {profit_rate:.2f}% / {cur_price:,.1f}원\n"
            msg += f"폭락확률: {crash_analysis['crash_probability']*100:.0f}%"
            print(msg)
            send_discord_message(msg)
            return sell_order
    
    # BB 매도 신호
    bb_analysis = analyze_bb_sell_signal(cur_price, closes, volumes)
    
    if bb_analysis:
        print(f"BB분석: {bb_analysis['reason']} | 긴급도: {bb_analysis['sell_urgency']}")
    
    # 반등 확률
    recovery_prob = calculate_recovery_probability(df_5m, cur_price, avg_buy_price)
    
    # ==================== 매도 신호 계산 ====================
    
    signals = []
    sell_strength = 0
    
    sma20 = np.mean(closes[-20:])
    std20 = np.std(closes[-20:])
    bb_upper = sma20 + (2.0 * std20)
    bb_lower = sma20 - (2.0 * std20)
    bb_position_simple = (cur_price - sma20) / std20
    
    # BB 상단 과열
    if bb_analysis and bb_analysis['sell_urgency'] in ['HIGH', 'CRITICAL']:
        signals.append("BB상단과열")
        sell_strength += 5
    
    if current_rsi > 70 and bb_position_simple > 1.5:
        if cur_price < closes[-2]:
            signals.append("과열후하락")
            sell_strength += 4
    
    # 추세 이탈
    sma10 = np.mean(closes[-10:])
    if cur_price < sma10 and sma10 < sma20:
        trend_break_volume = np.mean(volumes[-3:]) / np.mean(volumes[-10:-3])
        if trend_break_volume > 1.3:
            signals.append("추세이탈")
            sell_strength += 3
    
    # RSI 다이버전스
    if len(closes) >= 10:
        price_trend = closes[-1] - closes[-5]
        prev_rsi = calculate_rsi(closes[:-5])
        if price_trend > 0 and current_rsi < prev_rsi - 5:
            signals.append("RSI다이버전스")
            sell_strength += 3
    
    # 매도 기준 설정
    if profit_rate >= max_rate:
        required_score = 1
        hold_bonus = 0
    elif profit_rate >= min_rate * 2:
        required_score = 2
        hold_bonus = 1 if recovery_prob > 0.6 else 0
    elif profit_rate >= min_rate * 1.5:
        required_score = 3
        hold_bonus = 2 if recovery_prob > 0.7 else 0
    else:
        required_score = 4
        hold_bonus = 3 if recovery_prob > 0.8 else 1
    
    # BB 홀딩 보너스
    if bb_analysis and bb_analysis['should_hold']:
        hold_bonus += 2
    
    adjusted_required_score = required_score + hold_bonus
    should_sell_technical = sell_strength >= adjusted_required_score
    signal_text = " + ".join(signals) + f" ({sell_strength}/{adjusted_required_score})"
    
    # ==================== 매도 실행 루프 ====================
    
    max_attempts = min(sell_time, 25)
    attempts = 0
    consecutive_no_change = 0
    last_price = cur_price
    
    while attempts < max_attempts:
        cur_price = pyupbit.get_current_price(ticker)
        profit_rate = (cur_price - avg_buy_price) / avg_buy_price * 100 if avg_buy_price > 0 else 0
        
        price_change = abs(cur_price - last_price) / last_price
        if price_change < 0.001:
            consecutive_no_change += 1
        else:
            consecutive_no_change = 0
        last_price = cur_price
        
        # 실시간 BB 업데이트
        df_5m_live = pyupbit.get_ohlcv(ticker, interval="minute5", count=50)
        time.sleep(0.1)
        if df_5m_live is not None and len(df_5m_live) >= 30:
            closes_live = df_5m_live['close'].values
            volumes_live = df_5m_live['volume'].values
            bb_analysis_live = analyze_bb_sell_signal(cur_price, closes_live, volumes_live)
        else:
            bb_analysis_live = bb_analysis
        
        print(f"[{ticker}] {attempts + 1}/{max_attempts} | {profit_rate:.2f}% | "
              f"{sell_strength}/{adjusted_required_score} | "
              f"{bb_analysis_live['reason'] if bb_analysis_live else 'N/A'}")
        
        # [1] 목표 달성
        if profit_rate >= max_rate:
            sell_order = upbit.sell_market_order(ticker, buyed_amount)
            msg = f"🎯 **[목표달성]**: [{ticker}] {profit_rate:.2f}% / {cur_price:,.1f}원"
            print(msg)
            send_discord_message(msg)
            return sell_order
        
        # [2] 기술적 매도 + BB 검증
        if should_sell_technical and profit_rate >= min_rate * 1.2:
            if bb_analysis_live and bb_analysis_live['should_hold']:
                print(f"   ⏸️ BB하단으로 홀딩: {bb_analysis_live['reason']}")
            else:
                sell_order = upbit.sell_market_order(ticker, buyed_amount)
                msg = f"📊 **[기술적매도]**: [{ticker}] {profit_rate:.2f}% / {cur_price:,.1f}원\n{signal_text}"
                print(msg)
                send_discord_message(msg)
                return sell_order
        
        # [3] 정체 매도 + BB 검증
        if consecutive_no_change >= 8 and profit_rate >= min_rate * 1.5:
            if bb_analysis_live and bb_analysis_live['should_hold']:
                print(f"   ⏸️ 정체지만 BB하단: {bb_analysis_live['reason']}")
            else:
                sell_order = upbit.sell_market_order(ticker, buyed_amount)
                msg = f"⏸️ **[정체매도]**: [{ticker}] {profit_rate:.2f}% / {cur_price:,.1f}원"
                print(msg)
                send_discord_message(msg)
                return sell_order
        
        # [4] BB 긴급 매도
        if bb_analysis_live and bb_analysis_live['sell_urgency'] == 'CRITICAL':
            if profit_rate >= min_rate * 1.1:
                sell_order = upbit.sell_market_order(ticker, buyed_amount)
                msg = f"🚨 **[BB긴급매도]**: [{ticker}] {profit_rate:.2f}% / {cur_price:,.1f}원\n{bb_analysis_live['reason']}"
                print(msg)
                send_discord_message(msg)
                return sell_order
        
        time.sleep(second)
        attempts += 1
    
    # ==================== 시간 종료 처리 ====================
    
    print(f"\n[{ticker}] 시간종료 - BB 최종판단")
    
    df_5m_final = pyupbit.get_ohlcv(ticker, interval="minute5", count=50)
    time.sleep(0.1)
    if df_5m_final is not None and len(df_5m_final) >= 30:
        closes_final = df_5m_final['close'].values
        volumes_final = df_5m_final['volume'].values
        bb_analysis_final = analyze_bb_sell_signal(cur_price, closes_final, volumes_final)
    else:
        bb_analysis_final = bb_analysis
    
    if profit_rate >= min_rate:
        # BB 하단~중단이면 홀딩
        if bb_analysis_final and bb_analysis_final['should_hold']:
            msg = f"🤝 **[시간종료-홀딩]**: [{ticker}] {profit_rate:.2f}% / {cur_price:,.1f}원\n"
            msg += f"{bb_analysis_final['reason']} (반등:{recovery_prob:.1%})"
            print(msg)
            send_discord_message(msg)
            return None
        
        # BB 상단이면 매도
        else:
            sell_order = upbit.sell_market_order(ticker, buyed_amount)
            msg = f"⏰ **[시간종료-BB상단매도]**: [{ticker}] {profit_rate:.2f}% / {cur_price:,.1f}원\n"
            msg += f"{bb_analysis_final['reason']}"
            print(msg)
            send_discord_message(msg)
            return sell_order
    
    else:
        msg = f"🤝 **[홀딩지속]**: [{ticker}] {profit_rate:.2f}% / {cur_price:,.1f}원\n"
        msg += f"최소수익률 미달 (반등:{recovery_prob:.1%})"
        print(msg)
        send_discord_message(msg)
    
    return None


# 누적 자산 기록용 변수 
last_total_krw = 0.0 
profit_report_running = False

def send_profit_report():
    """
    효율화된 수익률 보고서 - 매시간 정시 실행
    """
    global profit_report_running
    
    if profit_report_running:
        return
    
    profit_report_running = True
    
    try:
        while True:
            # 1. 정시를 기다리는 초기 대기 로직 (첫 실행 시 정시가 아닐 경우)
            now = datetime.now()
            
            # 현재 시각이 정시가 아니면 (예: 12:15) → 다음 정시(13:00)까지 대기
            if now.minute != 0 or now.second > 5: # 정시에 실행될 때 약간의 오차 허용
                # 다음 정시 계산
                next_run = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                wait_seconds = (next_run - now).total_seconds()
                
                # 30초 여유는 불필요함. 정확히 대기 후 바로 실행
                if wait_seconds > 0:
                    # print(f"현재 {now.strftime('%H:%M:%S')}. 다음 실행까지 {wait_seconds:.0f}초 대기...")
                    time.sleep(wait_seconds)
                
                # 대기 후 루프를 다시 시작하여 정시임을 확인하고 실행
                now = datetime.now() # 새로운 now 업데이트
                
            
            # --- 보고서 생성/전송 로직 (이제 now는 정시에 매우 가까움) ---
            try:
                # 잔고 조회
                balances = upbit.get_balances()
                if not balances:
                    raise Exception("잔고 조회 실패")
                
                # 자산 계산
                total_value = 0.0
                crypto_value = 0.0
                krw_balance = 0.0
                holdings = []
                
                EXCLUDED = {'QI', 'ONK', 'ETHF', 'ETHW', 'PURSE'}
                
                for b in balances:
                    currency = b.get('currency')
                    if not currency:
                        continue
                    
                    balance = float(b.get('balance', 0)) + float(b.get('locked', 0))
                    
                    if currency == 'KRW':
                        krw_balance = balance
                        total_value += balance
                        continue
                    
                    if balance <= 0 or currency in EXCLUDED:
                        continue
                    
                    # 현재가 조회 (1회만)
                    ticker = f"KRW-{currency}"
                    try:
                        current_price = pyupbit.get_current_price(ticker)
                        if not current_price:
                            continue
                    except:
                        continue
                    
                    avg_buy = float(b.get('avg_buy_price', 0))
                    eval_value = balance * current_price
                    profit_rate = ((current_price - avg_buy) / avg_buy * 100) if avg_buy > 0 else 0
                    net_profit = eval_value - (balance * avg_buy)
                    
                    crypto_value += eval_value
                    total_value += eval_value
                    
                    holdings.append({
                        'name': currency,
                        'rate': profit_rate,
                        'value': eval_value,
                        'profit': net_profit
                    })
                    
                    time.sleep(0.1)
                
                # 평가액 순 정렬
                holdings.sort(key=lambda x: x['value'], reverse=True)
                
                # 보고서 생성
                msg = f"[{now.strftime('%m/%d %H시')} 정시 보고서]\n" # now는 이미 정시!
                msg += "━━━━━━━━━━━━━━━━━━━━\n"
                msg += f"총자산: {total_value:,.0f}원\n"
                msg += f"KRW: {krw_balance:,.0f}원 | 암호화폐: {crypto_value:,.0f}원\n\n"
                
                if holdings:
                    msg += f"보유자산 ({len(holdings)}개)\n"
                    msg += "━━━━━━━━━━━━━━━━━━━━\n"
                    
                    for i, h in enumerate(holdings, 1):
                        emoji = "🔥" if h['rate'] > 5 else "📈" if h['rate'] > 0 else "➡️" if h['rate'] > -5 else "📉"
                        msg += (
                            f"{i}. {h['name']:<4} {emoji} "
                            f"{h['rate']:+6.2f}% | "
                            f"평가 {h['value']:>10,.0f}원 | "
                            f"순익 {h['profit']:>+10,.0f}원\n"
                        )
                else:
                    msg += "보유 코인 없음\n"
                
                send_discord_message(msg)
                print(f"[{now.strftime('%H시')}] 보고서 전송 완료 (총자산: {total_value:,.0f}원)")
                
                # 2. 다음 정시까지 대기하는 로직 (핵심 수정)
                # 보고서 전송 후 다음 정시까지 남은 시간 계산 (약 1시간에서 전송 시간을 뺀 값)
                next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                wait_duration = (next_hour - datetime.now()).total_seconds()
                
                if wait_duration > 0:
                    time.sleep(wait_duration)
                
            except Exception as e:
                error_msg = f"수익률 보고서 오류\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{str(e)}"
                print(error_msg)
                # send_discord_message(error_msg) # 에러 메시지는 반복 전송하지 않도록 주석 처리 권장
                time.sleep(300) # 5분 후 재시도
    
    finally:
        profit_report_running = False
        
def selling_logic():
    """매도 로직 - 보유 코인 매도 처리"""
    try:
        balances = upbit.get_balances()
    except Exception as e:
        print(f"selling_logic / 잔고 조회 오류: {e}")
        return False
    
    has_holdings = False
    excluded_currencies = {"KRW", "QI", "ONX", "ETHF", "ETHW", "PURSE"}
    
    if isinstance(balances, list):
        for b in balances:
            currency = b.get('currency')
            if currency in excluded_currencies:
                continue
                
            balance = float(b.get('balance', 0))
            if balance <= 0:
                continue
            
            ticker = f"KRW-{currency}"
            
            try:
                result = trade_sell(ticker)
                has_holdings = True
                if result:
                    print(f"✅ {ticker} 매도 처리 완료")
            except Exception as e:
                print(f"selling_logic / {ticker} 매도 처리 오류: {e}")
                has_holdings = True
    
    return has_holdings

def buying_logic():
    """개선된 메인 매매 로직 - 통합 시스템 연동"""
    
    # 수익률 보고 스레드 시작
    profit_thread = threading.Thread(target=send_profit_report, daemon=True)
    profit_thread.start()
    print("수익률 보고 스레드 시작됨")
    
    while True:
        try:
            # ========== 1. 매도 로직 우선 실행 ==========
            has_holdings = selling_logic()
            
            # ========== 2. 매수 제한 시간 확인 ==========
            now = datetime.now()
            restricted_start = now.replace(hour=8, minute=50, second=0, microsecond=0)
            restricted_end = now.replace(hour=9, minute=10, second=0, microsecond=0)
            
            if restricted_start <= now <= restricted_end:
                print("매수 제한 시간 (08:50~09:10). 60초 대기...")
                time.sleep(60)
                continue
            
            # ========== 3. 원화 잔고 확인 ==========
            try:
                krw_balance = get_balance("KRW")
            except Exception as e:
                print(f"KRW 잔고 조회 오류: {e}")
                time.sleep(10)
                continue
            
            # ========== 4. 통합 매수 로직 실행 (종목 선정 + 매수) ==========
            if krw_balance > min_krw:
                print(f"매수 가능 잔고: {krw_balance:,.0f}원")
                
                try:
                    # trade_buy()가 종목 선정부터 매수까지 모두 처리
                    buy_time = datetime.now().strftime('%m/%d %H:%M:%S')
                    print(f"[{buy_time}] 최적 종목 자동 선정 + 매수 시작...")
                    
                    result = trade_buy(ticker=None)  # None이면 자동 선정 모드
                    
                    # 결과 판단
                    if result and isinstance(result, dict):
                        # 매수 성공
                        success_msg = "매수 성공! 다음 기회까지 "
                        wait_time = 15 if has_holdings else 30
                        print(f"{success_msg}{wait_time}초 대기")
                        time.sleep(wait_time)
                        
                    elif result and isinstance(result, tuple):
                        # 매수 실패 (이유 포함)
                        reason, _ = result
                        
                        if reason == "No candidates found":
                            wait_time = 10 if has_holdings else 30
                            print(f"매수할 코인 없음. {wait_time}초 후 재탐색...")
                            time.sleep(wait_time)
                            
                        elif reason == "Conditions not met":
                            print("매수 조건 미충족. 20초 후 재시도...")
                            time.sleep(20)
                            
                        elif reason == "Position limit reached":
                            wait_time = 60 if has_holdings else 120
                            print(f"포지션 상한 도달. {wait_time}초 대기...")
                            time.sleep(wait_time)
                            
                        elif reason == "Insufficient balance":
                            wait_time = 60 if has_holdings else 120
                            print(f"잔고 부족. {wait_time}초 대기...")
                            time.sleep(wait_time)
                            
                        else:
                            # 기타 실패 사유
                            print(f"매수 실패: {reason}. 30초 후 재시도...")
                            time.sleep(30)
                    else:
                        # 예상치 못한 결과
                        print("알 수 없는 결과. 30초 후 재시도...")
                        time.sleep(30)
                        
                except Exception as e:
                    print(f"매수 로직 실행 오류: {e}")
                    send_discord_message(f"매수 로직 오류: {e}")
                    time.sleep(30)
                    
            else:
                wait_time = 60 if has_holdings else 120
                print(f"매수 자금 부족: {krw_balance:,.0f}원. {wait_time}초 대기...")
                time.sleep(wait_time)
                
        except KeyboardInterrupt:
            print("\n프로그램 종료 요청...")
            break
            
        except Exception as e:
            print(f"메인 루프 오류: {e}")
            send_discord_message(f"메인 루프 오류: {e}")
            time.sleep(30)

# ========== 프로그램 시작 ==========
if __name__ == "__main__":
    # trade_start = datetime.now().strftime('%m/%d %H시%M분%S초')
    # trade_msg = f'🚀 {trade_start} 통합 복리 매수 시스템 v3.0\n'
    trade_msg = f'📊 설정: 수익률 {min_rate}%~{max_rate}% | 매도시도 {sell_time}회 | 손절 {cut_rate}%\n'
    trade_msg += f'📈 RSI 매수: {rsi_buy_s}~{rsi_buy_e} | RSI 매도: {rsi_sell_s}~{rsi_sell_e}\n'
    trade_msg += f'💡 개선사항: 조건완화, 병렬처리, 자동보고'
    
    print(trade_msg)
    send_discord_message(trade_msg)
    
    # 메인 매매 로직 실행
    buying_logic()
    # try:
    #     buying_logic()
    # except KeyboardInterrupt:
    #     print("\n\n프로그램이 종료되었습니다.")
    # except Exception as e:
    #     print(f"\n\n치명적 오류: {e}")
    #     send_discord_message(f"시스템 종료: {e}")