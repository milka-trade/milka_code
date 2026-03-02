"""
╔══════════════════════════════════════════════════════════════════════╗
║    한국투자증권(KIS) 산업별 주도주 자동매매 시스템                    ║
║    Milka Trading System — KIS Edition                               ║
║                                                                      ║
║    kis_config.py         ← 설정 파일                                 ║
║    kis_trading_system.py ← 이 파일 (전체 시스템)                     ║
║                                                                      ║
║    실행: python kis_trading_system.py                                ║
║    의존: pip install requests websocket-client                       ║
╚══════════════════════════════════════════════════════════════════════╝

[키움 v7 대비 변경 사항]
  Part 1. KiwoomAPI → KisAPI
    - COM/OCX 제거, requests(REST) + websocket-client(WS) 채택
    - OAuth 2.0 토큰 기반 인증 (24시간 유효, 자동 갱신)
    - PyQt5 QEventLoop 제거 → threading + asyncio 혼용 구조
    - TR 조회/실시간 수신이 독립 스레드로 분리 (블로킹 없음)
  Part 2. 스크리닝 엔진 — pykrx 하이브리드 아키텍처 (v8 신규)
    - 데이터소스 이원화: 스크리닝=pykrx(KRX공식), 주문/잔고=KIS API
    - pykrx 도입으로 스크리닝 시 KIS API 호출 165회 → 0회
    - 모의투자 스크리닝 소요시간 ~33초 → ~3초
    - pykrx 장애 시 전일 DB 캐시 → KIS API 순으로 자동 폴백
    - KRX 업종명 기반 섹터 분류 (config.PYKRX_SECTOR_MAP 참조)
  Part 3~6. 포트폴리오·트레일링·DB·리포트
    - 로직 100% 동일 유지

구조:
  Part 1. KIS API 래퍼              (KisAPI)
  Part 2. 산업별 주도주 스크리닝 엔진 (SectorLeaderScreener)
  Part 3. 포트폴리오 매니저           (PortfolioManager)
  Part 4. 트레일링 스탑 매도 엔진     (TrailingStopEngine)
  Part 5. 리포트 & DB               (TradingDB, DailyReporter)
  Part 6. 메인 엔진 & 실행           (TradingEngine, main)
"""

import os
import sys
import json
import time
import signal
import sqlite3
import logging
import threading
import traceback
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum

#pip install requests websocket-client pandas pykrx
import requests
import websocket  # pip install websocket-client

# pykrx: 스크리닝 전용 데이터소스 (KRX 공식, Rate Limit 없음)
# pip install pykrx
try:
    from pykrx import stock as pykrx_stock
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False
    logging.getLogger("Screener").warning(
        "pykrx 미설치 — KIS API 폴백 모드로 스크리닝. "
        "'pip install pykrx' 실행 권장"
    )

import kis_config as config


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Part 1. KIS API 래퍼                                            ║
# ╚════════════════════════════════════════════════════════════════════╝

logger_api = logging.getLogger("KisAPI")


class KisAPI:
    """
    한국투자증권 OpenAPI (KIS Developers) 래퍼

    [구조 설명]
    ┌──────────────────────────────────────────────────────────┐
    │  REST API  → requests 동기 호출 (스크리닝·주문·잔고)     │
    │  WebSocket → 별도 스레드에서 실시간 체결 수신            │
    │  두 채널이 독립적으로 동작 → 블로킹 없음                 │
    └──────────────────────────────────────────────────────────┘

    [인증 흐름]
    APP_KEY + APP_SECRET → access_token (24시간) → API 호출 헤더
    """

    def __init__(self):
        self.is_virtual     = config.IS_VIRTUAL
        self.base_url       = (config.BASE_URL_VIRTUAL
                               if self.is_virtual
                               else config.BASE_URL_REAL)
        self.ws_url         = (config.WS_URL_VIRTUAL
                               if self.is_virtual
                               else config.WS_URL_REAL)
        self.api_delay      = (config.API_DELAY_VIRTUAL
                               if self.is_virtual
                               else config.API_DELAY_REAL)

        # 인증 토큰
        self.access_token    = ""
        self.token_expired   = datetime.now()
        self.approval_key    = ""   # WebSocket 접속키

        # 계좌 정보
        self.account_no      = config.ACCOUNT_NO
        self.account_product = config.ACCOUNT_PRODUCT

        # WebSocket 상태
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_connected  = threading.Event()
        self._ws_subscribed: Dict[str, str] = {}  # {종목코드: tr_id}

        # 실시간 콜백 (TrailingStopEngine이 등록)
        self.real_data_callback: Optional[Callable] = None

        # Rate Limit: 마지막 API 호출 시각 기록용 락
        self._rate_lock = threading.Lock()
        self._last_call  = 0.0

        logger_api.info(
            f"KisAPI 초기화 | {'모의투자' if self.is_virtual else '🔴 실서버'}"
        )

    # ── 인증 ──────────────────────────────────────────────────────
    def _get_base_url(self) -> str:
        return self.base_url

    def _is_token_valid(self) -> bool:
        return bool(self.access_token) and datetime.now() < self.token_expired

    def _load_token_cache(self) -> bool:
        """저장된 토큰 캐시 로드 (24시간 유효)"""
        try:
            if not os.path.exists(config.TOKEN_CACHE_PATH):
                return False
            with open(config.TOKEN_CACHE_PATH, "r") as f:
                cache = json.load(f)
            key  = "virtual" if self.is_virtual else "real"
            data = cache.get(key, {})
            exp  = datetime.fromisoformat(data.get("expires_at", "2000-01-01"))
            if datetime.now() < exp - timedelta(minutes=10):
                self.access_token  = data["access_token"]
                self.token_expired = exp
                logger_api.info("캐시에서 토큰 복원 성공")
                return True
        except Exception:
            pass
        return False

    def _save_token_cache(self):
        """토큰을 파일에 캐시 저장"""
        try:
            os.makedirs(os.path.dirname(config.TOKEN_CACHE_PATH), exist_ok=True)
            cache = {}
            if os.path.exists(config.TOKEN_CACHE_PATH):
                with open(config.TOKEN_CACHE_PATH, "r") as f:
                    cache = json.load(f)
            key = "virtual" if self.is_virtual else "real"
            cache[key] = {
                "access_token": self.access_token,
                "expires_at":   self.token_expired.isoformat(),
            }
            with open(config.TOKEN_CACHE_PATH, "w") as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            logger_api.warning(f"토큰 캐시 저장 실패: {e}")

    def issue_token(self) -> bool:
        """액세스 토큰 발급 (POST /oauth2/tokenP)"""
        if self._load_token_cache():
            return True

        url  = f"{self.base_url}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey":     config.APP_KEY,
            "appsecret":  config.APP_SECRET,
        }
        try:
            resp = requests.post(url, json=body, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            self.access_token  = data["access_token"]
            # 만료 시각: 응답의 access_token_token_expired 또는 현재+23.5시간
            exp_str = data.get("access_token_token_expired", "")
            if exp_str:
                self.token_expired = datetime.strptime(
                    exp_str, "%Y-%m-%d %H:%M:%S"
                )
            else:
                self.token_expired = datetime.now() + timedelta(hours=23, minutes=30)
            self._save_token_cache()
            logger_api.info("액세스 토큰 발급 성공")
            return True
        except Exception as e:
            logger_api.error(f"토큰 발급 실패: {e}")
            return False

    def revoke_token(self):
        """토큰 폐기 (종료 시 호출)"""
        try:
            url  = f"{self.base_url}/oauth2/revokeP"
            body = {
                "appkey":       config.APP_KEY,
                "appsecret":    config.APP_SECRET,
                "token":        self.access_token,
            }
            requests.post(url, json=body, timeout=5)
            logger_api.info("토큰 폐기 완료")
        except Exception:
            pass

    def issue_ws_approval_key(self) -> bool:
        """WebSocket 접속키 발급"""
        url  = f"{self.base_url}/oauth2/Approval"
        body = {
            "grant_type": "client_credentials",
            "appkey":     config.APP_KEY,
            "secretkey":  config.APP_SECRET,
        }
        try:
            resp = requests.post(url, json=body, timeout=10)
            resp.raise_for_status()
            self.approval_key = resp.json().get("approval_key", "")
            logger_api.info("WebSocket 접속키 발급 성공")
            return bool(self.approval_key)
        except Exception as e:
            logger_api.error(f"WebSocket 접속키 발급 실패: {e}")
            return False

    # ── REST 공통 호출 ────────────────────────────────────────────
    def _headers(self, tr_id: str, extra: Optional[dict] = None) -> dict:
        """공통 요청 헤더 생성"""
        h = {
            "Content-Type":  "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "appkey":        config.APP_KEY,
            "appsecret":     config.APP_SECRET,
            "tr_id":         tr_id,
            "custtype":      "P",   # P=개인
        }
        if extra:
            h.update(extra)
        return h

    def _rate_limited_get(self, url: str, headers: dict,
                          params: dict) -> Optional[dict]:
        """Rate Limit 준수 GET 요청"""
        with self._rate_lock:
            elapsed = time.time() - self._last_call
            if elapsed < self.api_delay:
                time.sleep(self.api_delay - elapsed)
            self._last_call = time.time()

        # 토큰 만료 체크
        if not self._is_token_valid():
            logger_api.warning("토큰 만료 — 재발급 시도")
            self.access_token = ""  # 캐시 무효화
            if not self.issue_token():
                return None
            headers["authorization"] = f"Bearer {self.access_token}"

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("rt_cd") != "0":
                msg = data.get("msg1", "알 수 없는 오류")
                logger_api.error(f"API 오류 [{data.get('msg_cd')}]: {msg}")
                return None
            return data
        except requests.exceptions.Timeout:
            logger_api.error(f"API 타임아웃: {url}")
            return None
        except Exception as e:
            logger_api.error(f"API 호출 실패: {e}")
            return None

    def _rate_limited_post(self, url: str, headers: dict,
                           body: dict) -> Optional[dict]:
        """Rate Limit 준수 POST 요청"""
        with self._rate_lock:
            elapsed = time.time() - self._last_call
            if elapsed < self.api_delay:
                time.sleep(self.api_delay - elapsed)
            self._last_call = time.time()

        if not self._is_token_valid():
            self.access_token = ""
            if not self.issue_token():
                return None
            headers["authorization"] = f"Bearer {self.access_token}"

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("rt_cd") != "0":
                msg = data.get("msg1", "알 수 없는 오류")
                logger_api.error(f"API 오류 [{data.get('msg_cd')}]: {msg}")
                return None
            return data
        except Exception as e:
            logger_api.error(f"API POST 실패: {e}")
            return None

    # ── 시세 조회 ─────────────────────────────────────────────────
    def get_sector_stocks(self, sector_code: str) -> List[dict]:
        """
        업종별 종목 시세 조회
        TR: FHPUP02100000 (국내주식-업종별주가)
        """
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-member-volume"
        # KIS는 업종별종목조회를 위해 FHPUP02100000 또는
        # 조건검색(FHKST03900000) 활용
        # 실용적 대안: 거래량상위 조회 (업종 필터 포함)
        url = f"{self.base_url}/uapi/domestic-stock/v1/ranking/volume"
        tr_id = "FHPST01710000"
        headers = self._headers(tr_id)
        params = {
            "fid_cond_mrkt_div_code": "J",       # J=주식
            "fid_cond_scr_div_code":  "20171",
            "fid_input_iscd":         sector_code,
            "fid_div_cls_code":       "0",
            "fid_blng_cls_code":      "0",
            "fid_trgt_cls_code":      "111111111",
            "fid_trgt_exls_cls_code": "000000",
            "fid_input_price_1":      str(config.MIN_PRICE),
            "fid_input_price_2":      "",
            "fid_vol_cnt":            "",
            "fid_input_date_1":       "",
        }
        data = self._rate_limited_get(url, headers, params)
        if not data:
            return []
        return data.get("output", [])

    def get_sector_price(self, sector_code: str) -> List[dict]:
        """
        업종별 주가 조회 (FHPUP02100000)
        키움의 OPT20002 대응 API
        """
        url   = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-sectional-index"
        tr_id = "FHPUP02100000"
        headers = self._headers(tr_id)
        params = {
            "fid_cond_mrkt_div_code": "U",           # U=업종
            "fid_input_iscd":         sector_code,
        }
        data = self._rate_limited_get(url, headers, params)
        if not data:
            return []
        return data.get("output2", [])

    def get_volume_ranking(self, sector_code: str = "0000",
                           top_n: int = 30) -> List[dict]:
        """
        거래량 상위 종목 조회 (FHPST01710000)
        섹터 스크리닝의 핵심 데이터소스
        """
        url   = f"{self.base_url}/uapi/domestic-stock/v1/ranking/volume"
        tr_id = "FHPST01710000"
        headers = self._headers(tr_id)
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_cond_scr_div_code":  "20171",
            "fid_input_iscd":         sector_code,
            "fid_div_cls_code":       "0",
            "fid_blng_cls_code":      "0",
            "fid_trgt_cls_code":      "111111111",
            "fid_trgt_exls_cls_code": "000000",
            "fid_input_price_1":      str(config.MIN_PRICE),
            "fid_input_price_2":      "",
            "fid_vol_cnt":            "",
            "fid_input_date_1":       "",
        }
        data = self._rate_limited_get(url, headers, params)
        if not data:
            return []
        return data.get("output", [])[:top_n]

    def get_fluctuation_ranking(self, sector_code: str = "0000",
                                direction: str = "0") -> List[dict]:
        """
        등락률 상위 종목 조회 (FHPST01700000)
        direction: "0"=상위등락, "1"=하위등락
        """
        url   = f"{self.base_url}/uapi/domestic-stock/v1/ranking/fluctuation"
        tr_id = "FHPST01700000"
        headers = self._headers(tr_id)
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_cond_scr_div_code":  "20170",
            "fid_input_iscd":         sector_code,
            "fid_rank_sort_cls_code": direction,
            "fid_input_cnt_1":        "0",
            "fid_prc_cls_code":       "1",
            "fid_input_price_1":      str(config.MIN_PRICE),
            "fid_input_price_2":      "",
            "fid_vol_cnt":            "100000",
            "fid_trgt_cls_code":      "0",
            "fid_trgt_exls_cls_code": "0",
            "fid_div_cls_code":       "0",
            "fid_rsfl_rate1":         "",
            "fid_rsfl_rate2":         "",
        }
        data = self._rate_limited_get(url, headers, params)
        if not data:
            return []
        return data.get("output", [])

    def get_stock_basic_info(self, code: str) -> dict:
        """
        주식 기본 정보 조회 (FHKST01010100)
        키움의 OPT10001 대응 API
        """
        url   = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        tr_id = "FHKST01010100"
        headers = self._headers(tr_id)
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd":         code,
        }
        data = self._rate_limited_get(url, headers, params)
        if not data:
            return {}
        return data.get("output", {})

    def get_investor_trading(self, code: str) -> dict:
        """
        투자자별 매매 동향 조회 (FHKST01010900)
        외국인/기관 수급 데이터 - 키움의 OPT10059 대응
        """
        url   = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-investor"
        tr_id = "FHKST01010900"
        headers = self._headers(tr_id)
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd":         code,
        }
        data = self._rate_limited_get(url, headers, params)
        if not data:
            return {}
        rows = data.get("output", [])
        # 가장 최근(당일) 데이터 반환
        return rows[0] if rows else {}

    def get_daily_chart(self, code: str, period: str = "D") -> List[dict]:
        """
        일봉 차트 조회 (FHKST01010400)
        키움의 OPT10081 대응 API
        period: "D"=일봉, "W"=주봉, "M"=월봉
        """
        url   = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
        tr_id = "FHKST01010400"
        headers = self._headers(tr_id)
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd":         code,
            "fid_input_date_1":       "",
            "fid_input_date_2":       datetime.now().strftime("%Y%m%d"),
            "fid_period_div_code":    period,
            "fid_org_adj_prc":        "0",  # 0=수정주가
        }
        data = self._rate_limited_get(url, headers, params)
        if not data:
            return []
        return data.get("output2", [])

    # ── 계좌 조회 ─────────────────────────────────────────────────
    def get_account_balance(self) -> dict:
        """
        계좌 잔고 조회 (TTTC8434R/VTTC8434R)
        키움의 OPW00018 대응 API
        실서버: TTTC8434R / 모의투자: VTTC8434R
        """
        url   = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        tr_id = "VTTC8434R" if self.is_virtual else "TTTC8434R"
        headers = self._headers(tr_id)
        params = {
            "CANO":            self.account_no,
            "ACNT_PRDT_CD":    self.account_product,
            "AFHR_FLPR_YN":    "N",
            "OFL_YN":          "",
            "INQR_DVSN":       "02",
            "UNPR_DVSN":       "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN":       "01",
            "CTX_AREA_FK100":  "",
            "CTX_AREA_NK100":  "",
        }
        data = self._rate_limited_get(url, headers, params)
        if not data:
            return {}
        return data

    def get_orderable_cash(self) -> int:
        """
        주문 가능 금액 조회 (TTTC8908R/VTTC8908R)
        키움의 OPW00001 대응 API
        """
        url   = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-psbl-order"
        tr_id = "VTTC8908R" if self.is_virtual else "TTTC8908R"
        headers = self._headers(tr_id)
        params = {
            "CANO":         self.account_no,
            "ACNT_PRDT_CD": self.account_product,
            "PDNO":         "005930",   # 더미 종목코드(삼성전자)
            "ORD_UNPR":     "0",
            "ORD_DVSN":     config.ORDER_TYPE_MARKET,
            "CMA_EVLU_AMT_ICLD_YN": "Y",
            "OVRS_ICLD_YN": "N",
        }
        data = self._rate_limited_get(url, headers, params)
        if not data:
            return -1
        output = data.get("output", {})
        raw = output.get("ord_psbl_cash", "0").replace(",", "")
        try:
            val = int(raw)
            logger_api.info(f"주문가능금액: {val:,}원")
            return val
        except ValueError:
            return 0

    # ── 주문 ──────────────────────────────────────────────────────
    def send_order(self, code: str, qty: int, order_type: str,
                   price: int = 0, is_buy: bool = True) -> bool:
        """
        주식 주문 (TTTC0802U/VTTC0802U)
        order_type: "01"=시장가, "00"=지정가
        is_buy: True=매수, False=매도
        """
        if is_buy:
            tr_id = "VTTC0802U" if self.is_virtual else "TTTC0802U"
            side  = "BUY"
        else:
            tr_id = "VTTC0801U" if self.is_virtual else "TTTC0801U"
            side  = "SELL"

        url     = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
        headers = self._headers(tr_id)
        body = {
            "CANO":         self.account_no,
            "ACNT_PRDT_CD": self.account_product,
            "PDNO":         code,
            "ORD_DVSN":     order_type,
            "ORD_QTY":      str(qty),
            "ORD_UNPR":     str(price),   # 시장가일 때는 "0"
        }
        data = self._rate_limited_post(url, headers, body)
        if data:
            odno = data.get("output", {}).get("ODNO", "")
            logger_api.info(
                f"주문성공: {side} {code} {qty}주 | 주문번호: {odno}"
            )
            return True
        logger_api.error(f"주문실패: {side} {code} {qty}주")
        return False

    def buy_market_order(self, code: str, qty: int) -> bool:
        return self.send_order(code, qty, config.ORDER_TYPE_MARKET,
                               price=0, is_buy=True)

    def sell_market_order(self, code: str, qty: int) -> bool:
        return self.send_order(code, qty, config.ORDER_TYPE_MARKET,
                               price=0, is_buy=False)

    # ── WebSocket 실시간 체결 ─────────────────────────────────────
    def _ws_on_open(self, ws):
        logger_api.info("WebSocket 연결 완료")
        self._ws_connected.set()

    def _ws_on_message(self, ws, message: str):
        """
        실시간 데이터 수신 파싱
        메시지 타입:
          "0|TR_ID|count|data^data^..." → 실시간 시세 (비암호화)
          "1|TR_ID|count|data^..."     → 체결통보 (암호화, 별도 처리)
          "{...}"                       → JSON 시스템 메시지 (구독 응답 등)
        """
        try:
            if not message:
                return
            first = message[0]
            if first in ("0", "1"):
                parts = message.split("|")
                if len(parts) < 4:
                    return
                tr_id    = parts[1]
                cnt_str  = parts[2]
                raw_data = parts[3]

                if tr_id == config.WS_TR_PRICE:
                    # H0STCNT0: 실시간 체결가
                    cnt = int(cnt_str) if cnt_str.isdigit() else 1
                    self._parse_realtime_price(raw_data, cnt)

                elif tr_id in (config.WS_TR_NOTICE_REAL,
                               config.WS_TR_NOTICE_VIRTUAL):
                    # H0STCNI0 / H0STCNI9: 체결통보 (AES 암호화)
                    # 실제 복호화는 KIS Developers 공식 샘플 참조 필요
                    # 트레일링 스탑은 H0STCNT0 체결가로 충분하므로 INFO만 기록
                    logger_api.debug(f"체결통보 수신: {tr_id}")

            else:
                # JSON 시스템 메시지 (구독 결과, PINGPONG 등)
                obj   = json.loads(message)
                hdr   = obj.get("header", {})
                body  = obj.get("body",   {})
                tr_id = hdr.get("tr_id", "")
                rt_cd = body.get("rt_cd", "")
                msg1  = body.get("msg1",  "")

                if tr_id == "PINGPONG":
                    ws.send(message)  # 서버 PING에 PONG 응답
                elif rt_cd == "0":
                    logger_api.info(f"WS 구독 성공: {tr_id} | {msg1}")
                else:
                    logger_api.warning(f"WS 시스템 메시지: {tr_id} | {msg1}")

        except json.JSONDecodeError:
            pass  # 비JSON 메시지는 이미 위에서 처리됨
        except Exception as e:
            logger_api.error(f"WS 메시지 파싱 오류: {e}")

    def _parse_realtime_price(self, raw: str, cnt: int):
        """
        H0STCNT0 실시간 체결가 데이터 파싱
        ^ 구분자로 이어진 필드를 분해
        필드 순서(KIS 공식 문서 기준):
          0:종목코드, 1:체결시간, 2:현재가, 3:전일대비부호,
          4:전일대비, 5:전일대비율, 6:가중평균가, 7:시가, 8:고가,
          9:저가, 10:매도호가1, 11:매수호가1, 12:체결거래량,
          13:누적거래량, 14:누적거래대금, ...
        """
        fields = raw.split("^")
        n      = 46  # H0STCNT0 단일 종목 필드 수
        for i in range(cnt):
            base = i * n
            if base + 14 >= len(fields):
                break
            try:
                code       = fields[base + 0].strip()
                tick_time  = fields[base + 1].strip()
                price_raw  = fields[base + 2].strip()
                high_raw   = fields[base + 8].strip()
                vol_raw    = fields[base + 12].strip()
                cum_vol    = fields[base + 13].strip()
                open_raw   = fields[base + 7].strip()

                price = abs(int(price_raw)) if price_raw else 0
                high  = abs(int(high_raw))  if high_raw  else 0
                vol   = abs(int(vol_raw))   if vol_raw   else 0

                if price > 0 and self.real_data_callback:
                    self.real_data_callback({
                        "code":       code,
                        "time":       tick_time,
                        "price":      price,
                        "high":       high,
                        "volume":     vol,
                        "cum_volume": abs(int(cum_vol)) if cum_vol else 0,
                        "open":       abs(int(open_raw)) if open_raw else 0,
                    })
            except (ValueError, IndexError):
                continue

    def _ws_on_error(self, ws, error):
        logger_api.error(f"WebSocket 오류: {error}")

    def _ws_on_close(self, ws, close_status_code, close_msg):
        logger_api.warning(f"WebSocket 종료: {close_status_code} {close_msg}")
        self._ws_connected.clear()

    def _ws_subscribe(self, code: str, tr_id: str, subscribe: bool = True):
        """실시간 구독 등록/해제"""
        if not self._ws or not self._ws_connected.is_set():
            return
        tr_type = "1" if subscribe else "2"
        msg = json.dumps({
            "header": {
                "approval_key": self.approval_key,
                "custtype":     "P",
                "tr_type":      tr_type,
                "content-type": "utf-8",
            },
            "body": {
                "input": {
                    "tr_id":  tr_id,
                    "tr_key": code,
                }
            }
        })
        try:
            self._ws.send(msg)
            action = "구독" if subscribe else "해제"
            logger_api.debug(f"실시간 {action}: {code} ({tr_id})")
        except Exception as e:
            logger_api.error(f"WS 구독 메시지 전송 실패: {e}")

    def start_realtime(self, codes: List[str]):
        """WebSocket 시작 + 종목 구독 (자동 재연결 포함)"""
        if not self.issue_ws_approval_key():
            logger_api.error("WebSocket 접속키 없음 — 실시간 시작 불가")
            return

        def _run():
            reconnect_count = 0
            while reconnect_count < config.WS_RECONNECT_MAX:
                try:
                    self._ws = websocket.WebSocketApp(
                        self.ws_url,
                        on_open    = self._ws_on_open,
                        on_message = self._ws_on_message,
                        on_error   = self._ws_on_error,
                        on_close   = self._ws_on_close,
                    )
                    self._ws.run_forever(
                        ping_interval=30,
                        ping_timeout=10,
                    )
                except Exception as e:
                    logger_api.error(f"WS 스레드 예외: {e}")

                # 연결이 끊긴 경우 재연결
                if not self.is_running if hasattr(self, 'is_running') else False:
                    break
                reconnect_count += 1
                wait = min(5 * reconnect_count, 30)
                logger_api.warning(
                    f"WS 재연결 대기 {wait}초 "
                    f"({reconnect_count}/{config.WS_RECONNECT_MAX})"
                )
                time.sleep(wait)

                # 재연결 시 접속키 갱신
                self.issue_ws_approval_key()

                # 재구독 목록 복원
                if self._ws_subscribed:
                    self._ws_connected.wait(timeout=15)
                    for code, tr_id in list(self._ws_subscribed.items()):
                        self._ws_subscribe(code, tr_id, subscribe=True)
                        time.sleep(0.05)
                    logger_api.info(
                        f"WS 재구독 완료: {len(self._ws_subscribed)}종목"
                    )

        self._ws_thread = threading.Thread(target=_run, daemon=True,
                                           name="KIS-WebSocket")
        self._ws_thread.start()

        # 연결 대기 (최대 15초)
        if not self._ws_connected.wait(timeout=15):
            logger_api.error("WebSocket 연결 타임아웃 (15초)")
            return

        # 종목 구독 등록 (세션당 최대 41개)
        for code in codes[:config.WS_MAX_SUBSCRIPTIONS]:
            self._ws_subscribe(code, config.WS_TR_PRICE, subscribe=True)
            self._ws_subscribed[code] = config.WS_TR_PRICE
            time.sleep(0.05)

        # 체결통보 구독
        notice_tr = (config.WS_TR_NOTICE_VIRTUAL
                     if self.is_virtual
                     else config.WS_TR_NOTICE_REAL)
        self._ws_subscribe(config.ACCOUNT_NO, notice_tr, subscribe=True)

        logger_api.info(
            f"실시간 구독 시작: {len(self._ws_subscribed)}종목 "
            f"+ 체결통보({notice_tr})"
        )

    def add_realtime(self, code: str):
        """실시간 구독 종목 추가"""
        if code in self._ws_subscribed:
            return
        if len(self._ws_subscribed) >= config.WS_MAX_SUBSCRIPTIONS:
            logger_api.warning(f"WS 구독 한도 초과({config.WS_MAX_SUBSCRIPTIONS})")
            return
        self._ws_subscribe(code, config.WS_TR_PRICE, subscribe=True)
        self._ws_subscribed[code] = config.WS_TR_PRICE

    def remove_realtime(self, code: str):
        """실시간 구독 종목 해제"""
        if code not in self._ws_subscribed:
            return
        self._ws_subscribe(code, config.WS_TR_PRICE, subscribe=False)
        del self._ws_subscribed[code]

    def stop_realtime(self):
        """WebSocket 종료"""
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        self._ws_subscribed.clear()
        logger_api.info("실시간 수신 종료")


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Part 2. 산업별 주도주 스크리닝 엔진 (pykrx 하이브리드 v8)          ║
# ╚════════════════════════════════════════════════════════════════════╝
#
# ── 아키텍처 개요 ────────────────────────────────────────────────────
#
#   [스크리닝 단계 — 08:30]
#     pykrx (KRX 공식, Rate Limit 없음)
#       ① get_market_sector_classifications()  → 업종별 종목 목록
#       ② get_market_ohlcv_by_ticker()         → 전종목 OHLCV + 시총
#       ③ get_market_net_purchases_of_equities_by_ticker()
#                                             → 외국인/기관 순매수
#       ④ get_market_ohlcv_by_date()           → N일 모멘텀 일봉
#     KIS API 호출: 0회 (기존 165회 → 0회, 94% 감소)
#
#   [폴백 계층]
#     pykrx 장애 → DB 캐시(전일 스크리닝 결과) → KIS API 폴백
#
#   [주문/잔고 단계 — 09:05 이후]
#     KIS API만 사용 (Part3 PortfolioManager, Part4 TrailingStopEngine)
#
# ── KRX 업종명 ↔ Milka 섹터명 매핑 ──────────────────────────────────
#   config.py 의 PYKRX_SECTOR_MAP 으로 관리.
#   KRX 업종명은 get_market_sector_classifications() 반환값 기준.
# ─────────────────────────────────────────────────────────────────────

logger_scr = logging.getLogger("Screener")


@dataclass
class StockCandidate:
    """주도주 후보 종목"""
    code: str
    name: str
    sector: str
    price: int = 0
    change_rate: float = 0.0
    volume: int = 0
    trading_value: int = 0        # 거래대금 (백만원)
    market_cap: int = 0           # 시가총액 (억원)
    turnover_rate: float = 0.0
    foreign_net_buy: int = 0      # 외국인 순매수거래대금 (원)
    institution_net_buy: int = 0  # 기관 순매수거래대금 (원)
    composite_score: float = 0.0
    rank_details: Dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"[{self.sector}] {self.name}({self.code}) "
            f"점수:{self.composite_score:.2f} "
            f"등락:{self.change_rate:+.2f}% "
            f"거래대금:{self.trading_value:,}백만"
        )


class PykrxDataLoader:
    """
    pykrx 기반 데이터 로더.
    스크리닝에 필요한 모든 시장 데이터를 KRX에서 직접 수집한다.

    설계 원칙:
      - Rate Limit 없음 (KRX 공식 데이터, 별도 인증 불필요)
      - 전일 기준 데이터 (08:30 스크리닝 시점에 완전 유효)
      - 단일 날짜 전종목 일괄 조회 → 종목별 반복 호출 없음
    """

    def __init__(self):
        self._base_date: str = ""           # 오늘 기준 최근 영업일
        self._ohlcv_cache: object = None    # 전종목 OHLCV 캐시
        self._sector_cache: object = None   # 섹터 분류 캐시
        self._loaded_date: str = ""         # 캐시된 날짜

    # ── 영업일 계산 ────────────────────────────────────────────────
    def get_base_date(self) -> str:
        """오늘 기준 최근 영업일 (YYYYMMDD). 주말이면 직전 금요일."""
        today = datetime.now()
        # 장 시작(09:00) 전이면 전일 기준
        if today.hour < 9:
            today -= timedelta(days=1)
        # 주말 건너뜀
        while today.weekday() >= 5:
            today -= timedelta(days=1)
        return today.strftime("%Y%m%d")

    # ── 캐시 초기화 ────────────────────────────────────────────────
    def load_daily_data(self, date: str = "") -> bool:
        """
        전종목 OHLCV + 섹터 분류를 한 번에 로드하고 캐시.
        이미 같은 날짜 데이터가 캐시됐으면 재사용.

        Returns:
            True: 성공, False: 실패 (폴백 필요)
        """
        if not PYKRX_AVAILABLE:
            return False

        target = date or self.get_base_date()
        if self._loaded_date == target and self._ohlcv_cache is not None:
            return True  # 이미 캐시됨

        try:
            logger_scr.info(f"[pykrx] 전종목 데이터 로드 시작: {target}")

            # ① 전종목 OHLCV (시가/고가/저가/종가/거래량/거래대금/등락률/시가총액)
            import pandas as pd
            ohlcv_k = pykrx_stock.get_market_ohlcv_by_ticker(target, market="KOSPI")
            ohlcv_q = pykrx_stock.get_market_ohlcv_by_ticker(target, market="KOSDAQ")
            if ohlcv_k is None or ohlcv_k.empty:
                logger_scr.warning(f"[pykrx] KOSPI OHLCV 빈 데이터: {target}")
                return False
            self._ohlcv_cache = pd.concat([ohlcv_k, ohlcv_q], axis=0)

            # ② 섹터 분류 (종목코드 → 업종명, 종목명)
            sect_k = pykrx_stock.get_market_sector_classifications(target, market="KOSPI")
            sect_q = pykrx_stock.get_market_sector_classifications(target, market="KOSDAQ")
            self._sector_cache = pd.concat(
                [df for df in [sect_k, sect_q] if df is not None and not df.empty],
                axis=0
            )

            self._loaded_date = target
            self._base_date = target
            logger_scr.info(
                f"[pykrx] 로드 완료: OHLCV {len(self._ohlcv_cache)}종목 "
                f"| 섹터분류 {len(self._sector_cache)}종목"
            )
            return True

        except Exception as e:
            logger_scr.error(f"[pykrx] 데이터 로드 실패: {e}")
            self._ohlcv_cache = None
            self._sector_cache = None
            return False

    # ── 섹터별 종목 수집 ───────────────────────────────────────────
    def get_sector_stocks(self, milka_sector: str) -> list:
        """
        Milka 섹터명 → KRX 업종명 → 해당 종목 리스트 반환.

        Args:
            milka_sector: config.PYKRX_SECTOR_MAP 의 키 (예: "반도체")

        Returns:
            StockCandidate 리스트 (가격/거래대금/시총 포함)
        """
        if self._ohlcv_cache is None or self._sector_cache is None:
            return []

        krx_names = config.PYKRX_SECTOR_MAP.get(milka_sector, [])
        if isinstance(krx_names, str):
            krx_names = [krx_names]

        # 업종명 필터링 (복수 업종 OR 조건)
        mask = self._sector_cache["업종명"].isin(krx_names)
        filtered = self._sector_cache[mask]

        if filtered.empty:
            return []

        candidates = []
        for ticker, row in filtered.iterrows():
            code = str(ticker).strip()
            if not code or len(code) != 6:
                continue
            if config.EXCLUDE_ETF and code.startswith(("Q", "1", "2", "3")):
                continue

            # OHLCV 데이터 조인
            if code not in self._ohlcv_cache.index:
                continue
            ohlcv_row = self._ohlcv_cache.loc[code]

            try:
                price       = abs(int(ohlcv_row.get("종가", 0)))
                vol         = abs(int(ohlcv_row.get("거래량", 0)))
                # 거래대금: pykrx는 원 단위 → 백만원으로 변환
                trd_val_raw = abs(int(ohlcv_row.get("거래대금", 0)))
                trd_val     = trd_val_raw // 1_000_000
                chg         = float(ohlcv_row.get("등락률", 0.0))
                # 시가총액: pykrx는 원 단위 → 억원으로 변환
                cap_raw     = abs(int(ohlcv_row.get("시가총액", 0)))
                cap         = cap_raw // 100_000_000
                name        = str(row.get("종목명", code)).strip()

                if price <= 0:
                    continue

                turnover = (trd_val / cap * 100) if cap > 0 else 0.0

                candidates.append(StockCandidate(
                    code=code, name=name, sector=milka_sector,
                    price=price, change_rate=chg,
                    volume=vol, trading_value=trd_val,
                    market_cap=cap, turnover_rate=turnover,
                ))
            except (ValueError, TypeError, KeyError):
                continue

        return candidates

    # ── 외국인/기관 순매수 수집 ────────────────────────────────────
    def enrich_investor_data(self, candidates: list, top_n: int = 10) -> None:
        """
        상위 top_n 종목에 외국인/기관 순매수거래대금 추가.
        pykrx: get_market_net_purchases_of_equities_by_ticker()
               investor="9000" (외국인), "7050" (기관합계)

        Args:
            candidates: StockCandidate 리스트 (in-place 수정)
            top_n: 거래대금 상위 N개만 조회 (전체 조회는 과도한 데이터)
        """
        if not PYKRX_AVAILABLE or not candidates:
            return

        date = self._base_date
        top = sorted(candidates, key=lambda x: x.trading_value, reverse=True)[:top_n]
        top_codes = set(c.code for c in top)

        try:
            # 외국인 순매수 (investor="9000")
            foreign_df = pykrx_stock.get_market_net_purchases_of_equities_by_ticker(
                date, date, "ALL", "9000"
            )
            # 기관합계 순매수 (investor="7050")
            inst_df = pykrx_stock.get_market_net_purchases_of_equities_by_ticker(
                date, date, "ALL", "7050"
            )
        except Exception as e:
            logger_scr.warning(f"[pykrx] 투자자별 순매수 조회 실패: {e}")
            return

        # StockCandidate에 매핑
        for c in top:
            if c.code not in top_codes:
                continue
            try:
                if foreign_df is not None and not foreign_df.empty and c.code in foreign_df.index:
                    # 순매수거래대금 (원 단위)
                    c.foreign_net_buy = int(
                        foreign_df.loc[c.code, "순매수거래대금"]
                    )
            except Exception:
                pass
            try:
                if inst_df is not None and not inst_df.empty and c.code in inst_df.index:
                    c.institution_net_buy = int(
                        inst_df.loc[c.code, "순매수거래대금"]
                    )
            except Exception:
                pass

    # ── 모멘텀 필터용 일봉 조회 ────────────────────────────────────
    def get_daily_prices(self, code: str, days: int = 5) -> list:
        """
        최근 N 영업일 종가 리스트 반환 (최신→과거 순).
        pykrx: get_market_ohlcv_by_date(fromdate, todate, ticker)
        """
        if not PYKRX_AVAILABLE:
            return []
        try:
            end   = self._base_date or self.get_base_date()
            # N 영업일 = 약 N*1.5 캘린더일 (주말/공휴일 여유)
            start_dt = datetime.strptime(end, "%Y%m%d") - timedelta(days=int(days * 2))
            start = start_dt.strftime("%Y%m%d")
            df = pykrx_stock.get_market_ohlcv_by_date(start, end, code)
            if df is None or df.empty:
                return []
            prices = df["종가"].tolist()
            prices.reverse()  # 최신 → 과거
            return prices[:days]
        except Exception as e:
            logger_scr.debug(f"[pykrx] 일봉 조회 실패 {code}: {e}")
            return []


class SectorLeaderScreener:
    """
    산업별 주도주 스크리닝 엔진 (pykrx 하이브리드 v8)

    데이터 흐름:
      pykrx.load_daily_data()         전종목 OHLCV + 섹터분류 일괄 로드
        ↓
      _collect_sector_stocks()        섹터별 종목 필터링
        ↓
      _apply_filters()                최소 시총·가격·거래대금 조건
        ↓
      _enrich_investor_data()         외국인/기관 순매수 추가
        ↓
      _calculate_composite_scores()   6개 지표 가중 순위 합산
        ↓
      _select_leaders()               섹터별 1위 선정
        ↓
      apply_momentum_filter()         5일 모멘텀 통과 여부 검증
        ↓
      _rank_final_portfolio()         전체 상위 MAX_SECTORS개 확정

    폴백 계층:
      pykrx 실패 → _fallback_from_db()   (DB 캐시 전일 결과)
               → _fallback_to_kis_api()  (KIS API 직접 호출)
    """

    def __init__(self, api: KisAPI):
        self.api = api
        self.pykrx = PykrxDataLoader()
        self.candidates: Dict[str, List[StockCandidate]] = {}
        self.leaders: Dict[str, StockCandidate] = {}
        self.screening_time = None
        self._screening_source = "pykrx"   # 실제 사용된 데이터소스 기록

    # ── 메인 진입점 ───────────────────────────────────────────────
    def run_full_screening(self) -> Dict[str, StockCandidate]:
        """전체 스크리닝 프로세스"""
        logger_scr.info("=" * 60)
        logger_scr.info("주도주 스크리닝 시작 (pykrx 하이브리드 v8)")
        logger_scr.info("=" * 60)
        self.screening_time = datetime.now()

        # ── pykrx 데이터 로드 ─────────────────────────────────────
        pykrx_ok = self.pykrx.load_daily_data()
        if pykrx_ok:
            self._screening_source = "pykrx"
            logger_scr.info("[pykrx] 데이터 로드 성공 — pykrx 모드로 스크리닝")
            self._collect_sector_stocks_pykrx()
        else:
            logger_scr.warning(
                "[pykrx] 데이터 로드 실패 — KIS API 폴백 모드로 전환"
            )
            self._screening_source = "kis_api_fallback"
            self._collect_sector_stocks_kis()  # 기존 KIS API 방식

        self._apply_filters()
        self._enrich_investor_data(pykrx_ok)
        self._calculate_composite_scores()
        self._select_leaders()
        final = self._rank_final_portfolio()

        logger_scr.info("=" * 60)
        logger_scr.info(
            f"스크리닝 완료: {len(final)}개 섹터 주도주 선정 "
            f"[소스: {self._screening_source}]"
        )
        for sector, stock in final.items():
            logger_scr.info(f"  {stock}")
        logger_scr.info("=" * 60)
        return final

    # ── pykrx 기반 종목 수집 (신규) ──────────────────────────────
    def _collect_sector_stocks_pykrx(self):
        """pykrx로 섹터별 종목 수집. KIS API 호출 0회."""
        self.candidates = {}
        for milka_name in config.PYKRX_SECTOR_MAP:
            logger_scr.info(f"[수집-pykrx] {milka_name}")
            try:
                cands = self.pykrx.get_sector_stocks(milka_name)
                self.candidates[milka_name] = cands
                logger_scr.info(f"  → {len(cands)}개 종목")
            except Exception as e:
                logger_scr.error(f"  → {milka_name} 수집 실패: {e}")
                self.candidates[milka_name] = []

    # ── KIS API 폴백 종목 수집 (기존 로직 유지) ───────────────────
    def _collect_sector_stocks_kis(self):
        """KIS API로 섹터별 종목 수집 (pykrx 실패 시 폴백)."""
        self.candidates = {}
        sector_codes = getattr(config, "KIS_MARKET_SECTOR_CODES",
                               getattr(config, "SECTOR_CODES", {}))
        for name, code in sector_codes.items():
            logger_scr.info(f"[수집-KIS폴백] {name} (코드: {code})")
            try:
                rows = self.api.get_volume_ranking(sector_code=code, top_n=30)
                cands = []
                for r in rows:
                    try:
                        c = StockCandidate(
                            code=r.get("mksc_shrn_iscd", "").strip(),
                            name=r.get("hts_kor_isnm", "").strip(),
                            sector=name,
                            price=self._parse_price(r.get("stck_prpr", "0")),
                            change_rate=self._parse_float(r.get("prdy_ctrt", "0")),
                            volume=self._parse_price(r.get("acml_vol", "0")),
                            trading_value=self._parse_price(
                                r.get("acml_tr_pbmn", "0")) // 1_000_000,
                            market_cap=self._parse_price(
                                r.get("stck_mrkt_cap", "0")) // 100_000_000,
                        )
                        if c.code and c.price > 0:
                            if c.market_cap > 0:
                                c.turnover_rate = c.trading_value / c.market_cap * 100
                            cands.append(c)
                    except (ValueError, TypeError):
                        continue
                self.candidates[name] = cands
                logger_scr.info(f"  → {len(cands)}개 종목")
            except Exception as e:
                logger_scr.error(f"  → {name} 수집 실패: {e}")
                self.candidates[name] = []

    # ── 공통 유틸 ────────────────────────────────────────────────
    @staticmethod
    def _parse_price(raw) -> int:
        try:
            return abs(int(str(raw).replace(",", "").strip() or "0"))
        except ValueError:
            return 0

    @staticmethod
    def _parse_float(raw) -> float:
        try:
            return float(str(raw).replace(",", "").replace("+", "").strip() or "0")
        except ValueError:
            return 0.0

    # ── 필터·스코어링·선정 (기존 로직 유지) ─────────────────────
    def _apply_filters(self):
        """최소 시총·가격·거래대금 조건 필터링"""
        for name in list(self.candidates.keys()):
            self.candidates[name] = [
                c for c in self.candidates[name]
                if (c.market_cap >= config.MIN_MARKET_CAP
                    and c.trading_value >= config.MIN_TRADING_VOLUME * 100
                    and c.price >= config.MIN_PRICE
                    and len(c.code) == 6)
            ]

    def _enrich_investor_data(self, use_pykrx: bool = True):
        """
        외국인/기관 순매수 데이터 추가.
        use_pykrx=True  → pykrx 일괄 조회 (섹터별 상위 10개)
        use_pykrx=False → KIS API 개별 조회 (상위 5개, Rate Limit 절약)
        """
        if use_pykrx and PYKRX_AVAILABLE:
            # pykrx: 섹터 전체 후보를 한 번에 처리
            for name, cands in self.candidates.items():
                self.pykrx.enrich_investor_data(cands, top_n=10)
        else:
            # KIS API 폴백: 호출 최소화를 위해 상위 5개만
            for name, cands in self.candidates.items():
                top = sorted(cands, key=lambda x: x.trading_value, reverse=True)[:5]
                for c in top:
                    try:
                        inv = self.api.get_investor_trading(c.code)
                        if inv:
                            c.foreign_net_buy = int(
                                inv.get("frgn_ntby_qty", "0")
                                   .replace(",", "") or "0"
                            )
                            c.institution_net_buy = int(
                                inv.get("orgn_ntby_qty", "0")
                                   .replace(",", "") or "0"
                            )
                    except Exception:
                        pass

    def _calculate_composite_scores(self):
        """6개 지표 가중 순위 합산 스코어링 (키움 v7과 동일)"""
        for name, cands in self.candidates.items():
            n = len(cands)
            if n < 2:
                if cands:
                    cands[0].composite_score = 100
                continue

            rankings = {
                "volume_rank":          sorted(cands, key=lambda x: x.trading_value,       reverse=True),
                "price_change_rank":    sorted(cands, key=lambda x: x.change_rate,          reverse=True),
                "foreign_buy_rank":     sorted(cands, key=lambda x: x.foreign_net_buy,      reverse=True),
                "institution_buy_rank": sorted(cands, key=lambda x: x.institution_net_buy,  reverse=True),
                "market_cap_rank":      sorted(cands, key=lambda x: x.market_cap,           reverse=True),
                "turnover_rank":        sorted(cands, key=lambda x: x.turnover_rate,        reverse=True),
            }

            for c in cands:
                score = 0
                for metric, weight in config.SCREENING_WEIGHTS.items():
                    ranked = rankings[metric]
                    try:
                        rank = ranked.index(c) + 1
                    except ValueError:
                        rank = n
                    norm = (n - rank) / max(n - 1, 1) * 100
                    c.rank_details[metric] = {"rank": rank, "score": norm}
                    score += norm * weight
                c.composite_score = score

    def _select_leaders(self):
        """섹터별 최고 점수 종목 선정"""
        self.leaders = {}
        for name, cands in self.candidates.items():
            if cands:
                self.leaders[name] = max(cands, key=lambda x: x.composite_score)

    def _rank_final_portfolio(self) -> Dict[str, StockCandidate]:
        """전체 주도주 상위 MAX_SECTORS개 선정"""
        if not self.leaders:
            return {}
        ranked = sorted(
            self.leaders.items(),
            key=lambda x: x[1].composite_score, reverse=True
        )
        return dict(ranked[:config.MAX_SECTORS])

    # ── 모멘텀 필터 ──────────────────────────────────────────────
    def apply_momentum_filter(
        self, leaders: Dict[str, StockCandidate], days: int = 5
    ) -> Dict[str, StockCandidate]:
        """
        최근 N일 모멘텀 필터.
        현재 종가 ≥ N일 평균 × 0.98 인 종목만 통과.

        데이터소스:
          pykrx 모드  → pykrx.get_daily_prices() (API 호출 없음)
          KIS 폴백    → KIS API get_daily_chart()
        """
        filtered = {}
        for sector, leader in leaders.items():
            try:
                # pykrx 우선, 실패 시 KIS API
                if PYKRX_AVAILABLE and self._screening_source == "pykrx":
                    prices = self.pykrx.get_daily_prices(leader.code, days)
                else:
                    daily = self.api.get_daily_chart(leader.code)
                    prices = []
                    for d in daily[:days]:
                        raw = d.get("stck_clpr", "0").replace(",", "")
                        p = abs(int(raw or "0"))
                        if p > 0:
                            prices.append(p)

                if len(prices) >= 2:
                    avg = sum(prices) / len(prices)
                    if prices[0] >= avg * 0.98:
                        filtered[sector] = leader
                    else:
                        logger_scr.info(
                            f"[모멘텀 필터] {leader.name} 제외 "
                            f"(현재:{prices[0]:,} < 평균:{avg:.0f}×0.98)"
                        )
                else:
                    filtered[sector] = leader  # 데이터 부족 → 통과
            except Exception:
                filtered[sector] = leader
        return filtered

    # ── 리포트 ───────────────────────────────────────────────────
    def generate_report(self) -> str:
        """스크리닝 결과 리포트"""
        lines = ["=" * 70]
        lines.append(
            f"  주도주 스크리닝 리포트 | {self.screening_time} "
            f"[소스: {self._screening_source}]"
        )
        lines.append("=" * 70)
        for sector, ld in self.leaders.items():
            lines.append(f"\n[{sector}]")
            lines.append(f"  종목: {ld.name} ({ld.code})")
            lines.append(f"  현재가: {ld.price:,}원 | 등락률: {ld.change_rate:+.2f}%")
            lines.append(f"  거래대금: {ld.trading_value:,}백만 | 시총: {ld.market_cap:,}억")
            lines.append(f"  외국인: {ld.foreign_net_buy:,} | 기관: {ld.institution_net_buy:,}")
            lines.append(f"  복합점수: {ld.composite_score:.2f}")
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

# ╔════════════════════════════════════════════════════════════════════╗
# ║  Part 3. 포트폴리오 매니저                                         ║
# ╚════════════════════════════════════════════════════════════════════╝

logger_pf = logging.getLogger("Portfolio")


class OrderAction(Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Position:
    """보유 포지션 (키움 v7과 동일 구조)"""
    code: str
    name: str
    sector: str
    quantity: int
    avg_price: float
    current_price: float = 0
    day_high: float = 0
    day_open: float = 0
    buy_time: str = ""
    unrealized_pnl: float = 0
    unrealized_pnl_pct: float = 0
    trailing_high: float = 0
    target_reached: bool = False

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_price

    def update_price(self, price: float, high: float = 0):
        self.current_price = price
        if high > 0:
            self.day_high = max(self.day_high, high)
        self.day_high = max(self.day_high, price)
        self.trailing_high = max(self.trailing_high, price)
        if self.avg_price > 0:
            self.unrealized_pnl = (price - self.avg_price) * self.quantity
            self.unrealized_pnl_pct = (
                (price - self.avg_price) / self.avg_price
            )
            if self.unrealized_pnl_pct >= config.PROFIT_TARGET_PERCENT:
                self.target_reached = True


@dataclass
class RebalanceOrder:
    """리밸런싱 주문 (키움 v7과 동일)"""
    action: OrderAction
    code: str
    name: str
    sector: str
    quantity: int
    target_weight: float
    reason: str
    priority: int = 0


class PortfolioManager:
    """포트폴리오 관리 엔진 (키움 v7과 동일 로직)"""

    def __init__(self, api: KisAPI):
        self.api = api
        self.positions: Dict[str, Position] = {}
        self.total_assets    = 0
        self.available_cash  = 0
        self.total_invested  = 0
        self.daily_start_value = 0
        self.trade_history   = []

    # ── 계좌 동기화 ───────────────────────────────────────────────
    def sync_account(self) -> bool:
        """
        계좌 동기화
        KIS의 TTTC8434R(실전)/VTTC8434R(모의) 활용
        """
        logger_pf.info("계좌 동기화 (KIS)...")
        try:
            balance = self.api.get_account_balance()
            if not balance:
                logger_pf.error("계좌 잔고 조회 실패")
                return False

            output2 = balance.get("output2", [{}])
            summary = output2[0] if output2 else {}

            # 추정 총자산
            self.total_assets = int(
                summary.get("tot_evlu_amt", "0")
                       .replace(",", "") or "0"
            )
            self.total_invested = int(
                summary.get("pchs_amt_smtl_amt", "0")
                       .replace(",", "") or "0"
            )

            # 주문가능금액 별도 조회
            orderable = self.api.get_orderable_cash()
            if orderable >= 0:
                self.available_cash = orderable
            else:
                # 폴백: 총자산 - 총평가금액
                eval_amt = int(
                    summary.get("evlu_amt_smtl_amt", "0")
                           .replace(",", "") or "0"
                )
                self.available_cash = max(0, self.total_assets - eval_amt)
                logger_pf.warning(
                    f"주문가능금액 조회 실패 — 폴백: {self.available_cash:,}원"
                )

            # 보유 종목 파싱
            current_codes = set()
            for row in balance.get("output1", []):
                code = row.get("pdno", "").strip()
                if not code:
                    continue
                current_codes.add(code)
                qty = int(
                    row.get("hldg_qty", "0").replace(",", "") or "0"
                )
                avg = int(
                    row.get("pchs_avg_pric", "0").replace(",", "") or "0"
                )
                cur = int(
                    row.get("prpr", "0").replace(",", "") or "0"
                )
                name = row.get("prdt_name", "").strip()

                if code in self.positions:
                    p = self.positions[code]
                    p.quantity, p.avg_price = qty, avg
                    p.update_price(cur)
                else:
                    self.positions[code] = Position(
                        code=code, name=name,
                        sector="미분류", quantity=qty,
                        avg_price=avg, current_price=cur,
                        trailing_high=cur,
                    )

            for code in set(self.positions.keys()) - current_codes:
                del self.positions[code]

            logger_pf.info(
                f"동기화 완료 | 총자산:{self.total_assets:,} "
                f"| 현금:{self.available_cash:,} "
                f"| 보유:{len(self.positions)}종목"
            )
            return True
        except Exception as e:
            logger_pf.error(f"계좌 동기화 실패: {e}\n{traceback.format_exc()}")
            return False

    # ── 리밸런싱 계산 (키움 v7과 100% 동일) ──────────────────────
    def calculate_rebalance(
        self, new_leaders: Dict[str, StockCandidate]
    ) -> List[RebalanceOrder]:
        orders = []
        new_codes = {ld.code for ld in new_leaders.values()}

        # 1) 탈락 종목 매도
        for code, pos in list(self.positions.items()):
            if code not in new_codes:
                orders.append(RebalanceOrder(
                    OrderAction.SELL, code, pos.name, pos.sector,
                    pos.quantity, 0.0, "주도주 탈락 - 전량매도", 10
                ))

        # 2) 목표 비중 계산
        n = len(new_leaders)
        if n == 0:
            return orders

        total_score = sum(ld.composite_score for ld in new_leaders.values())
        weights = {}
        for sector, ld in new_leaders.items():
            if total_score > 0:
                eq  = 1.0 / n
                sc  = ld.composite_score / total_score
                raw = eq * 0.7 + sc * 0.3
            else:
                raw = 1.0 / n
            weights[ld.code] = max(
                config.MIN_WEIGHT_PER_STOCK,
                min(config.MAX_WEIGHT_PER_STOCK, raw)
            )

        wsum = sum(weights.values())
        if wsum > 0:
            for c in weights:
                weights[c] = (
                    weights[c] / wsum * config.TOTAL_INVESTMENT_RATIO
                )

        # 3) 매수/비중조절
        for sector, ld in new_leaders.items():
            tw         = weights.get(ld.code, 0)
            target_val = self.total_assets * tw

            if ld.code in self.positions:
                pos  = self.positions[ld.code]
                diff = target_val - pos.market_value
                if abs(diff) < ld.price * 5:
                    continue
                if diff > 0:
                    add_qty = int(diff / ld.price)
                    if add_qty > 0:
                        orders.append(RebalanceOrder(
                            OrderAction.BUY, ld.code, ld.name, sector,
                            add_qty, tw, "비중확대", 5
                        ))
                else:
                    sell_qty = int(abs(diff) / ld.price)
                    if sell_qty > 0:
                        orders.append(RebalanceOrder(
                            OrderAction.SELL, ld.code, ld.name, sector,
                            sell_qty, tw, "비중축소", 7
                        ))
            else:
                buy_qty = (
                    int(target_val / ld.price) if ld.price > 0 else 0
                )
                if buy_qty > 0:
                    orders.append(RebalanceOrder(
                        OrderAction.BUY, ld.code, ld.name, sector,
                        buy_qty, tw, f"신규편입 (비중:{tw:.1%})", 3
                    ))

        orders.sort(key=lambda x: x.priority, reverse=True)

        # 매수 자금 검증
        sell_proceeds = sum(
            self.positions[o.code].current_price * o.quantity
            for o in orders
            if o.action == OrderAction.SELL
            and o.code in self.positions
        )
        avail     = self.available_cash + sell_proceeds
        validated = []
        for o in orders:
            if o.action == OrderAction.BUY:
                price = self._get_price(o.code)
                cost  = o.quantity * price
                if cost <= avail:
                    validated.append(o)
                    avail -= cost
                else:
                    reduced = int(avail / price) if price > 0 else 0
                    if reduced > 0:
                        o.quantity = reduced
                        o.reason  += f" (수량축소:{reduced}주)"
                        validated.append(o)
                        avail -= reduced * price
            else:
                validated.append(o)

        logger_pf.info(f"리밸런싱 주문: {len(validated)}건")
        for o in validated:
            logger_pf.info(
                f"  {o.action.value} {o.name}({o.code}) "
                f"{o.quantity}주 | {o.reason}"
            )
        return validated

    def _get_price(self, code: str) -> int:
        if code in self.positions:
            return int(self.positions[code].current_price)
        info = self.api.get_stock_basic_info(code)
        raw  = info.get("stck_prpr", "0").replace(",", "")
        return abs(int(raw or "0"))

    def execute_rebalance(self, orders: List[RebalanceOrder]):
        """주문 실행 (키움 v7과 동일 구조)"""
        for order in orders:
            try:
                if order.action == OrderAction.SELL:
                    ret = self.api.sell_market_order(order.code, order.quantity)
                elif order.action == OrderAction.BUY:
                    ret = self.api.buy_market_order(order.code, order.quantity)
                else:
                    continue

                if ret:
                    self.trade_history.append({
                        "time":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "action": order.action.value,
                        "code":   order.code,
                        "name":   order.name,
                        "sector": order.sector,
                        "quantity": order.quantity,
                        "reason": order.reason,
                    })
                time.sleep(0.3)
            except Exception as e:
                logger_pf.error(f"주문 실행 실패: {order.name} — {e}")

    def update_position_price(self, code: str, price: int, high: int = 0):
        if code in self.positions:
            self.positions[code].update_price(price, high)

    def get_portfolio_summary(self) -> dict:
        total_mv   = sum(p.market_value for p in self.positions.values())
        total_cost = sum(p.cost_basis   for p in self.positions.values())
        pnl        = total_mv - total_cost
        pnl_pct    = (pnl / total_cost * 100) if total_cost > 0 else 0
        return {
            "total_assets":       self.total_assets,
            "cash":               self.available_cash,
            "invested":           total_mv,
            "cost_basis":         total_cost,
            "unrealized_pnl":     pnl,
            "unrealized_pnl_pct": pnl_pct,
            "position_count":     len(self.positions),
            "positions": {
                code: {
                    "name":          p.name,
                    "sector":        p.sector,
                    "qty":           p.quantity,
                    "avg_price":     p.avg_price,
                    "current_price": p.current_price,
                    "pnl":           p.unrealized_pnl,
                    "pnl_pct":       p.unrealized_pnl_pct * 100,
                    "day_high":      p.day_high,
                    "weight": (
                        p.market_value / self.total_assets * 100
                        if self.total_assets > 0 else 0
                    ),
                }
                for code, p in self.positions.items()
            }
        }

    def print_portfolio(self):
        s = self.get_portfolio_summary()
        logger_pf.info("=" * 70)
        logger_pf.info(f"  총자산: {s['total_assets']:>15,}원")
        logger_pf.info(f"  현  금: {s['cash']:>15,}원")
        logger_pf.info(
            f"  손  익: {s['unrealized_pnl']:>+15,}원 "
            f"({s['unrealized_pnl_pct']:+.2f}%)"
        )
        for code, info in s["positions"].items():
            logger_pf.info(
                f"  [{info['sector']}] {info['name']}({code}) "
                f"{info['qty']}주 | 평균:{info['avg_price']:,.0f} "
                f"현재:{info['current_price']:,.0f} | "
                f"손익:{info['pnl']:+,.0f}원({info['pnl_pct']:+.2f}%)"
            )
        logger_pf.info("=" * 70)


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Part 4. 트레일링 스탑 매도 엔진                                   ║
# ╚════════════════════════════════════════════════════════════════════╝

logger_sell = logging.getLogger("SellEngine")


class SellSignal(Enum):
    HOLD             = "HOLD"
    TRAILING_STOP    = "TRAILING_STOP"
    ENHANCED_TRAILING = "ENHANCED_TRAILING"
    STOP_LOSS        = "STOP_LOSS"
    FORCE_CLOSE      = "FORCE_CLOSE"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"


class TrailingStopEngine:
    """
    트레일링 스탑 매도 엔진 (키움 v7과 동일 전략)

    [키움 v7 대비 변경점]
    - COM 이벤트(OnReceiveRealData) → KIS WebSocket 콜백(real_data_callback)
    - SetRealReg/SetRealRemove → api.add_realtime/remove_realtime
    - QEventLoop 없음 → 순수 Python 스레드 기반
    """

    def __init__(self, api: KisAPI, portfolio: PortfolioManager):
        self.api       = api
        self.portfolio = portfolio
        self.sold_today: Dict[str, dict] = {}
        self.monitoring_active = False
        self.on_sell_executed: Optional[Callable] = None
        self.daily_sell_profit = 0
        self.daily_sell_count  = 0

        # WebSocket 콜백 등록
        self.api.real_data_callback = self.on_real_tick

    def on_real_tick(self, data: dict):
        """
        KIS WebSocket 실시간 체결 수신 → 매도 판정
        (키움의 _on_receive_real_data 대응)
        """
        if not self.monitoring_active:
            return
        code       = data.get("code", "")
        price      = data.get("price", 0)
        high       = data.get("high", 0)
        tick_time  = data.get("time", "")

        if code not in self.portfolio.positions:
            return

        pos = self.portfolio.positions[code]
        pos.update_price(price, high)
        signal = self._evaluate(pos, tick_time)
        if signal != SellSignal.HOLD:
            self._execute_sell(pos, signal)

    def _evaluate(self, pos: Position, tick_time: str) -> SellSignal:
        """매도 신호 판정 (키움 v7과 동일 우선순위)"""
        if tick_time and tick_time < config.SELL_START_TIME:
            return SellSignal.HOLD

        # 일일 손실 한도
        if self._check_daily_loss():
            return SellSignal.DAILY_LOSS_LIMIT

        # 강제 청산 시각
        if tick_time and tick_time >= config.FORCE_SELL_TIME:
            return SellSignal.FORCE_CLOSE

        # 손절매
        if pos.unrealized_pnl_pct <= config.STOP_LOSS_PERCENT:
            logger_sell.warning(
                f"[손절매] {pos.name} | 손실:{pos.unrealized_pnl_pct:.2%}"
            )
            return SellSignal.STOP_LOSS

        # 트레일링 스탑
        if pos.trailing_high > 0 and pos.current_price > 0:
            drop = (
                (pos.trailing_high - pos.current_price) / pos.trailing_high
            )
            if pos.target_reached:
                if drop >= config.ENHANCED_TRAILING_STOP:
                    logger_sell.info(
                        f"[강화트레일링] {pos.name} | "
                        f"고점:{pos.trailing_high:,}→현재:{pos.current_price:,}"
                    )
                    return SellSignal.ENHANCED_TRAILING
            else:
                if drop >= config.TRAILING_STOP_PERCENT:
                    logger_sell.info(
                        f"[트레일링스탑] {pos.name} | "
                        f"고점:{pos.trailing_high:,}→현재:{pos.current_price:,}"
                    )
                    return SellSignal.TRAILING_STOP

        return SellSignal.HOLD

    def _execute_sell(self, pos: Position, signal: SellSignal):
        if pos.quantity <= 0 or pos.code in self.sold_today:
            return

        logger_sell.info(
            f"매도실행: {pos.name} | 신호:{signal.value} | "
            f"{pos.quantity}주 | 매수가:{pos.avg_price:,} "
            f"현재가:{pos.current_price:,} | "
            f"고가:{pos.trailing_high:,} | 손익:{pos.unrealized_pnl_pct:.2%}"
        )

        ret = self.api.sell_market_order(pos.code, pos.quantity)
        if ret:
            self.sold_today[pos.code] = {
                "name":          pos.name,
                "signal":        signal.value,
                "qty":           pos.quantity,
                "sell_price":    pos.current_price,
                "avg_price":     pos.avg_price,
                "day_high":      pos.day_high,
                "trailing_high": pos.trailing_high,
                "pnl_pct":       pos.unrealized_pnl_pct,
                "pnl_amount":    pos.unrealized_pnl,
                "time":          datetime.now().strftime("%H:%M:%S"),
            }
            self.daily_sell_count  += 1
            self.daily_sell_profit += pos.unrealized_pnl
            if self.on_sell_executed:
                self.on_sell_executed(pos.code, signal, pos)

    def _check_daily_loss(self) -> bool:
        if self.portfolio.daily_start_value <= 0:
            return False
        total = (
            sum(p.market_value for p in self.portfolio.positions.values())
            + self.portfolio.available_cash
        )
        ret = (
            (total - self.portfolio.daily_start_value)
            / self.portfolio.daily_start_value
        )
        if ret <= config.MAX_DAILY_LOSS_PERCENT:
            logger_sell.critical(f"[일일 손실 한도 초과] {ret:.2%}")
            return True
        return False

    def force_sell_all(self, reason: str = "강제청산"):
        logger_sell.warning(f"전량 청산: {reason}")
        for code, pos in list(self.portfolio.positions.items()):
            if pos.quantity > 0 and code not in self.sold_today:
                self._execute_sell(pos, SellSignal.FORCE_CLOSE)

    def start_monitoring(self):
        """실시간 모니터링 시작 — KIS WebSocket 구독"""
        self.monitoring_active = True
        if not self.portfolio.positions:
            return
        codes = list(self.portfolio.positions.keys())
        self.api.start_realtime(codes)
        logger_sell.info(
            f"트레일링 스탑 모니터링 시작 ({len(codes)}종목)"
        )

    def add_position_monitoring(self, code: str):
        """새 포지션 실시간 구독 추가"""
        self.api.add_realtime(code)

    def stop_monitoring(self):
        self.monitoring_active = False
        self.api.stop_realtime()
        logger_sell.info("트레일링 스탑 모니터링 중지")

    def get_sell_report(self) -> str:
        lines = ["=" * 60, "  당일 매도 성과", "=" * 60]
        if not self.sold_today:
            lines.append("  매도 없음")
        else:
            total = 0
            for code, info in self.sold_today.items():
                pnl   = info["pnl_amount"]
                total += pnl
                eff   = 0
                if info["day_high"] > info["avg_price"]:
                    mx = (info["day_high"] - info["avg_price"]) * info["qty"]
                    if mx > 0:
                        eff = pnl / mx * 100
                lines.append(
                    f"  {info['name']} | {info['time']} | "
                    f"신호:{info['signal']} | "
                    f"매수:{info['avg_price']:,}→매도:{info['sell_price']:,} | "
                    f"손익:{pnl:+,}원({info['pnl_pct']:.2%}) | "
                    f"고가효율:{eff:.1f}%"
                )
            lines.append(
                f"\n  총 {len(self.sold_today)}건 | 총손익: {total:+,}원"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def reset_daily(self):
        self.sold_today        = {}
        self.daily_sell_profit = 0
        self.daily_sell_count  = 0


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Part 5. 리포트 & DB (키움 v7과 동일 구조)                        ║
# ╚════════════════════════════════════════════════════════════════════╝

logger_rpt = logging.getLogger("Reporter")


class TradingDB:
    """SQLite 거래 데이터베이스 (키움 v7과 동일)"""

    def __init__(self, db_path: str = config.DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS daily_portfolio (
                date TEXT, code TEXT, name TEXT, sector TEXT,
                quantity INTEGER, avg_price REAL, close_price REAL,
                day_high REAL, pnl_amount REAL, pnl_pct REAL,
                weight REAL, composite_score REAL,
                PRIMARY KEY (date, code)
            );
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                datetime TEXT, action TEXT, code TEXT, name TEXT,
                sector TEXT, quantity INTEGER, price REAL,
                amount REAL, signal_type TEXT, reason TEXT
            );
            CREATE TABLE IF NOT EXISTS daily_summary (
                date TEXT PRIMARY KEY, total_assets REAL,
                cash REAL, invested REAL, daily_pnl REAL,
                daily_pnl_pct REAL, cumulative_pnl REAL,
                cumulative_pnl_pct REAL, trade_count INTEGER,
                win_count INTEGER, loss_count INTEGER,
                max_drawdown REAL, sector_count INTEGER,
                screening_report TEXT
            );
            CREATE TABLE IF NOT EXISTS screening_history (
                date TEXT, sector TEXT, code TEXT, name TEXT,
                composite_score REAL, selected INTEGER DEFAULT 0,
                PRIMARY KEY (date, sector, code)
            );
        """)
        self.conn.commit()

    def log_trade(self, trade: dict):
        with self._lock:
            self.conn.execute(
                "INSERT INTO trades "
                "(datetime,action,code,name,sector,quantity,price,"
                "amount,signal_type,reason) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (trade.get("datetime", datetime.now().isoformat()),
                 trade.get("action"),  trade.get("code"),
                 trade.get("name"),    trade.get("sector"),
                 trade.get("quantity"), trade.get("price", 0),
                 trade.get("amount", 0), trade.get("signal_type", ""),
                 trade.get("reason", ""))
            )
            self.conn.commit()

    def log_daily_summary(self, s: dict):
        today = datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO daily_summary "
                "(date,total_assets,cash,invested,daily_pnl,daily_pnl_pct,"
                "cumulative_pnl,cumulative_pnl_pct,trade_count,win_count,"
                "loss_count,max_drawdown,sector_count,screening_report) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (today, s.get("total_assets",0), s.get("cash",0),
                 s.get("invested",0), s.get("daily_pnl",0),
                 s.get("daily_pnl_pct",0), s.get("cumulative_pnl",0),
                 s.get("cumulative_pnl_pct",0), s.get("trade_count",0),
                 s.get("win_count",0), s.get("loss_count",0),
                 s.get("max_drawdown",0), s.get("sector_count",0),
                 s.get("screening_report",""))
            )
            self.conn.commit()

    def log_portfolio_snapshot(self, positions: dict):
        today = datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            for code, info in positions.items():
                self.conn.execute(
                    "INSERT OR REPLACE INTO daily_portfolio "
                    "(date,code,name,sector,quantity,avg_price,close_price,"
                    "day_high,pnl_amount,pnl_pct,weight,composite_score) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (today, code, info.get("name"), info.get("sector"),
                     info.get("qty"), info.get("avg_price"),
                     info.get("current_price"), info.get("day_high", 0),
                     info.get("pnl"), info.get("pnl_pct"),
                     info.get("weight"), info.get("composite_score", 0))
                )
            self.conn.commit()

    def get_cumulative_stats(self) -> dict:
        cur = self.conn.execute(
            "SELECT COUNT(*), SUM(daily_pnl), AVG(daily_pnl_pct), "
            "MAX(daily_pnl_pct), MIN(daily_pnl_pct), "
            "SUM(CASE WHEN daily_pnl>0 THEN 1 ELSE 0 END), "
            "SUM(trade_count) FROM daily_summary"
        )
        r = cur.fetchone()
        if r:
            return {
                "trading_days":     r[0],
                "total_pnl":        r[1] or 0,
                "avg_daily_return": r[2] or 0,
                "best_day":         r[3] or 0,
                "worst_day":        r[4] or 0,
                "win_days":         r[5] or 0,
                "win_rate":         (r[5] or 0) / max(r[0], 1) * 100,
                "total_trades":     r[6] or 0,
            }
        return {}

    def close(self):
        self.conn.close()


class DailyReporter:
    """일일 리포트 생성 (키움 v7과 동일)"""

    def __init__(self, db: TradingDB):
        self.db = db

    def generate_daily_report(
        self, portfolio_summary: dict, sell_report: str,
        screening_report: str, trade_history: list
    ) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        cum   = self.db.get_cumulative_stats()
        ps    = portfolio_summary

        lines = []
        lines.append("╔" + "═" * 68 + "╗")
        lines.append(
            "║" + f"  KIS 일일 자동매매 리포트 | {today}".center(68) + "║"
        )
        lines.append("╚" + "═" * 68 + "╝")

        lines.append(f"\n◆ 계좌현황")
        lines.append(f"  총자산: {ps['total_assets']:>15,}원")
        lines.append(f"  투자금: {ps['invested']:>15,}원")
        lines.append(f"  현  금: {ps['cash']:>15,}원")
        lines.append(
            f"  손  익: {ps.get('unrealized_pnl',0):>+15,}원 "
            f"({ps.get('unrealized_pnl_pct',0):+.2f}%)"
        )

        lines.append(f"\n◆ 보유종목")
        for code, info in ps.get("positions", {}).items():
            lines.append(
                f"  [{info['sector']}] {info['name']}({code}) "
                f"{info['qty']}주 | {info['avg_price']:,.0f}→"
                f"{info['current_price']:,.0f} | "
                f"{info['pnl']:+,.0f}원({info['pnl_pct']:+.2f}%)"
            )

        lines.append(f"\n{sell_report}")

        if cum:
            lines.append(f"\n◆ 누적성과")
            lines.append(
                f"  운용:{cum['trading_days']}일 | "
                f"총손익:{cum['total_pnl']:+,.0f}원 | "
                f"승률:{cum['win_rate']:.1f}%"
            )

        if trade_history:
            lines.append(f"\n◆ 당일거래")
            for t in trade_history:
                lines.append(
                    f"  {t['time']} {t['action']} {t['name']} "
                    f"{t['quantity']}주 | {t['reason']}"
                )

        lines.append("\n" + "═" * 70)
        report = "\n".join(lines)

        os.makedirs(config.REPORT_DIR, exist_ok=True)
        path = os.path.join(config.REPORT_DIR, f"kis_report_{today}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        logger_rpt.info(f"리포트 저장: {path}")
        return report


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Part 6. 메인 엔진 & 실행                                         ║
# ╚════════════════════════════════════════════════════════════════════╝

logger_main = logging.getLogger("MainEngine")


def setup_logging():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    fmt = logging.Formatter(
        "[%(asctime)s] %(name)-12s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S"
    )
    fh = logging.FileHandler(
        os.path.join(config.LOG_DIR, f"kis_trading_{today}.log"),
        encoding="utf-8"
    )
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(getattr(logging, config.LOG_LEVEL))
    root.addHandler(fh)
    root.addHandler(ch)


class TradingEngine:
    """
    KIS 메인 자동매매 엔진

    [키움 v7 대비 구조 변경]
    - QTimer → threading.Timer (표준 Python, Qt 의존성 없음)
    - 스케줄러가 1초 주기 스레드에서 돌아감
    - WebSocket은 별도 daemon 스레드 (블로킹 없음)

    ┌──────────────────────────────────────────────────┐
    │ 08:30  스크리닝 → 섹터별 주도주 선정             │
    │ 09:05  리밸런싱 → 탈락종목 매도, 신규종목 매수   │
    │ 09:10  모니터링 → 트레일링 스탑 매도 감시         │
    │ 15:20  강제청산 → 미매도 종목 정리               │
    │ 16:00  리포트   → 일일 성과 저장                 │
    └──────────────────────────────────────────────────┘
    """

    def __init__(self):
        self.api         = KisAPI()
        self.screener    = SectorLeaderScreener(self.api)
        self.portfolio   = PortfolioManager(self.api)
        self.sell_engine = TrailingStopEngine(self.api, self.portfolio)
        self.db          = TradingDB()
        self.reporter    = DailyReporter(self.db)

        self.today_leaders: Dict[str, StockCandidate] = {}
        self.is_running    = False
        self.day_trading_mode = True  # True=당일청산

        # 스케줄 실행 여부 추적 (중복 실행 방지)
        self._tasks_done: Dict[str, str] = {}

        self.sell_engine.on_sell_executed = self._on_sell_complete

    # ── 시작 / 종료 ───────────────────────────────────────────────
    def start(self) -> bool:
        logger_main.info("╔════════════════════════════════════════╗")
        logger_main.info("║  KIS 산업별 주도주 자동매매 시스템 시작   ║")
        logger_main.info("╚════════════════════════════════════════╝")

        # [1/3] 토큰 발급
        logger_main.info("[1/3] KIS 액세스 토큰 발급...")
        if not self.api.issue_token():
            logger_main.critical("  ❌ 토큰 발급 실패 — 시스템 종료")
            logger_main.critical(
                "  확인사항:\n"
                "    kis_config.py → APP_KEY, APP_SECRET 입력 여부\n"
                "    KIS Developers 서비스 신청 완료 여부"
            )
            return False
        logger_main.info("  ✅ 토큰 발급 성공")

        # [2/3] 서버 검증
        mode = "🟡 모의투자" if self.api.is_virtual else "🔴 실서버"
        logger_main.info(f"[2/3] 서버 모드: {mode}")
        if not self.api.is_virtual:
            logger_main.warning(
                "  ⚠️  실서버 모드입니다. 실제 자금으로 거래됩니다!\n"
                "  모의투자 검증 2주 이상 완료 후에만 사용하세요."
            )

        # [3/3] 계좌 동기화
        logger_main.info("[3/3] 계좌 정보 확인...")
        if not self.portfolio.sync_account():
            logger_main.critical(
                "  ❌ 계좌 동기화 실패 — 시스템 종료\n"
                "  확인사항:\n"
                "    kis_config.py → ACCOUNT_NO, ACCOUNT_PRODUCT 입력 여부\n"
                "    KIS Developers 계좌 연결 여부"
            )
            return False
        self.portfolio.daily_start_value = self.portfolio.total_assets
        logger_main.info(f"  시작자산: {self.portfolio.total_assets:,}원")

        # 스케줄러 시작
        self.is_running = True
        self._initial_check()
        self._start_scheduler()
        logger_main.info(
            "  ✅ 시스템 가동 중 — 스케줄러 대기 중...\n"
            "  종료: Ctrl+C"
        )
        return True

    def stop(self):
        logger_main.info("시스템 종료...")
        self.is_running = False
        self.sell_engine.stop_monitoring()
        self.api.revoke_token()
        self.db.close()

    # ── 스케줄러 ──────────────────────────────────────────────────
    def _start_scheduler(self):
        """1초 주기 스케줄러 스레드 시작"""
        def _loop():
            while self.is_running:
                try:
                    self._check_schedule()
                except Exception as e:
                    logger_main.error(f"스케줄러 오류: {e}")
                time.sleep(1)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()

        # 메인 스레드 대기 (Ctrl+C 대기)
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    def _check_schedule(self):
        if not self.is_running or config.EMERGENCY_STOP:
            return
        now = datetime.now().strftime("%H%M%S")
        hm  = now[:4]
        today = datetime.now().strftime("%Y%m%d")

        def _once(key: str, func):
            """하루 한 번만 실행"""
            full_key = f"{today}_{key}"
            if full_key not in self._tasks_done:
                self._tasks_done[full_key] = now
                threading.Thread(target=func, daemon=True).start()

        if hm == config.SCHEDULE["screening"] and now[4:6] == "00":
            _once("screening", self._task_screening)
        elif now[:6] == config.BUY_START_TIME:
            _once("rebalance", self._task_rebalance)
        elif hm == config.SCHEDULE["monitoring_start"] and now[4:6] == "00":
            _once("monitoring", self._task_start_monitoring)
        elif hm == config.SCHEDULE["force_sell"] and now[4:6] == "00":
            if self.day_trading_mode:
                _once("force_sell", self._task_force_close)
        elif hm == config.SCHEDULE["market_close"] and now[4:6] == "00":
            _once("market_close", self._task_market_close)
        elif hm == config.SCHEDULE["daily_report"] and now[4:6] == "00":
            _once("daily_report", self._task_daily_report)

    def _initial_check(self):
        """시작 시각에 따라 즉시 실행할 태스크 결정"""
        now = datetime.now().strftime("%H%M")
        if "0830" <= now < "0900":
            threading.Thread(
                target=self._task_screening, daemon=True
            ).start()
        elif "0900" <= now < "0930":
            threading.Thread(
                target=self._task_screening, daemon=True
            ).start()
            time.sleep(5)
            threading.Thread(
                target=self._task_rebalance, daemon=True
            ).start()
            threading.Thread(
                target=self._task_start_monitoring, daemon=True
            ).start()
        elif "0930" <= now < "1520":
            self.portfolio.sync_account()
            threading.Thread(
                target=self._task_start_monitoring, daemon=True
            ).start()

    # ── 태스크 ────────────────────────────────────────────────────
    def _task_screening(self):
        logger_main.info("━━ 주도주 스크리닝 시작 (KIS) ━━")
        try:
            leaders = self.screener.run_full_screening()
            leaders = self.screener.apply_momentum_filter(leaders)
            self.today_leaders = leaders
            logger_main.info(self.screener.generate_report())

            today = datetime.now().strftime("%Y-%m-%d")
            for sec, ld in leaders.items():
                self.db.conn.execute(
                    "INSERT OR REPLACE INTO screening_history "
                    "(date,sector,code,name,composite_score,selected) "
                    "VALUES (?,?,?,?,?,1)",
                    (today, sec, ld.code, ld.name, ld.composite_score)
                )
            self.db.conn.commit()
        except Exception as e:
            logger_main.error(f"스크리닝 실패: {e}", exc_info=True)

    def _task_rebalance(self):
        logger_main.info("━━ 리밸런싱 시작 ━━")
        if not self.today_leaders:
            self._task_screening()
            if not self.today_leaders:
                return
        try:
            self.portfolio.sync_account()
            orders = self.portfolio.calculate_rebalance(self.today_leaders)
            if orders:
                self.portfolio.execute_rebalance(orders)
                time.sleep(3)
                self.portfolio.sync_account()
                self.portfolio.print_portfolio()
                for o in orders:
                    self.db.log_trade({
                        "datetime": datetime.now().isoformat(),
                        "action":   o.action.value,
                        "code":     o.code,
                        "name":     o.name,
                        "sector":   o.sector,
                        "quantity": o.quantity,
                        "reason":   o.reason,
                    })
        except Exception as e:
            logger_main.error(f"리밸런싱 실패: {e}", exc_info=True)

    def _task_start_monitoring(self):
        logger_main.info("━━ 트레일링 스탑 모니터링 시작 ━━")
        self.portfolio.sync_account()
        if not self.portfolio.positions:
            logger_main.info("  보유 종목 없음 — 모니터링 생략")
            return
        for p in self.portfolio.positions.values():
            p.trailing_high = max(p.current_price, p.day_high)
        self.sell_engine.start_monitoring()

    def _task_force_close(self):
        logger_main.info("━━ 강제 청산 (15:20) ━━")
        self.sell_engine.force_sell_all("장마감 전 강제청산")

    def _task_market_close(self):
        logger_main.info("━━ 장 마감 ━━")
        self.sell_engine.stop_monitoring()
        self.portfolio.sync_account()

    def _task_daily_report(self):
        logger_main.info("━━ 일일 리포트 생성 ━━")
        try:
            ps  = self.portfolio.get_portfolio_summary()
            sr  = self.sell_engine.get_sell_report()
            scr = (self.screener.generate_report()
                   if self.today_leaders else "미실행")
            report = self.reporter.generate_daily_report(
                ps, sr, scr, self.portfolio.trade_history
            )
            logger_main.info(report)

            pnl = (
                self.portfolio.total_assets
                - self.portfolio.daily_start_value
            )
            pnl_pct = (
                pnl / self.portfolio.daily_start_value * 100
                if self.portfolio.daily_start_value > 0 else 0
            )
            cum = self.db.get_cumulative_stats()
            self.db.log_daily_summary({
                "total_assets": self.portfolio.total_assets,
                "cash":         self.portfolio.available_cash,
                "invested":     self.portfolio.total_invested,
                "daily_pnl":    pnl,
                "daily_pnl_pct": pnl_pct,
                "cumulative_pnl": cum.get("total_pnl", 0) + pnl,
                "trade_count":  len(self.portfolio.trade_history),
                "sector_count": len(self.today_leaders),
                "screening_report": scr,
            })
            self.db.log_portfolio_snapshot(ps.get("positions", {}))
            self.sell_engine.reset_daily()
            self.portfolio.trade_history = []
        except Exception as e:
            logger_main.error(f"리포트 실패: {e}", exc_info=True)

    # ── 콜백 ─────────────────────────────────────────────────────
    def _on_sell_complete(self, code: str, signal: SellSignal,
                          position: Position):
        logger_main.info(
            f"[매도완료] {position.name} | {signal.value} | "
            f"손익:{position.unrealized_pnl:+,}원"
        )
        self.db.log_trade({
            "datetime":   datetime.now().isoformat(),
            "action":     "SELL",
            "code":       code,
            "name":       position.name,
            "sector":     position.sector,
            "quantity":   position.quantity,
            "price":      position.current_price,
            "amount":     position.market_value,
            "signal_type": signal.value,
            "reason": (
                f"고점:{position.trailing_high:,}"
                f"→매도:{position.current_price:,}"
            ),
        })


# ══════════════════════════════════════════════════════════════════════
#  실행 진입점
# ══════════════════════════════════════════════════════════════════════
def main():
    setup_logging()
    engine = TradingEngine()

    def on_exit(sig, frame):
        logger_main.info("종료 시그널 수신")
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)

    if engine.start():
        logger_main.info("자동매매 종료")
    else:
        logger_main.critical("시스템 시작 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
