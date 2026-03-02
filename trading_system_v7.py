"""
╔══════════════════════════════════════════════════════════════════════╗
║         키움증권 산업별 주도주 자동매매 시스템 (통합본)                ║
║                                                                      ║
║   config.py  ← 설정 파일 (별도)                                      ║
║   trading_system.py  ← 이 파일 (전체 시스템 통합)                     ║
║                                                                      ║
║   실행: python trading_system.py                                     ║
╚══════════════════════════════════════════════════════════════════════╝

구조:
  Part 1. 키움 OpenAPI+ 래퍼 (KiwoomAPI)
  Part 2. 산업별 주도주 스크리닝 엔진 (SectorLeaderScreener)
  Part 3. 포트폴리오 매니저 (PortfolioManager)
  Part 4. 트레일링 스탑 매도 엔진 (TrailingStopEngine)
  Part 5. 리포트 & DB (TradingDB, DailyReporter)
  Part 6. 메인 엔진 & 실행 (TradingEngine, main)
"""

import sys
import os
import time
import signal
import sqlite3
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum

from PyQt5.QAxContainer import QAxWidget
from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication

import config


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Part 1. 키움 OpenAPI+ 래퍼                                       ║
# ╚════════════════════════════════════════════════════════════════════╝

logger_api = logging.getLogger("KiwoomAPI")


class KiwoomAPI:
    """키움증권 OpenAPI+ COM 인터페이스 래퍼"""

    def __init__(self):
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication(sys.argv)

        self.ocx = QAxWidget("KHOPENAPI.KHOpenAPICtrl.1")
        self.connected = False
        self.account_no = ""
        self.accounts = []
        self.is_simul_server: Optional[bool] = None  # True=모의투자, False=실서버

        # 이벤트 루프 (비동기 → 동기 변환용)
        self.login_loop = QEventLoop()
        self.tr_loop = QEventLoop()

        # TR 데이터 저장
        self.tr_data = {}
        self.tr_remained = False

        # 실시간 데이터 콜백
        self.real_data_callback = None

        # 주문 결과
        self.order_result = {}

        # 이벤트 연결
        self._connect_events()

    # ── 이벤트 슬롯 연결 ──────────────────────────────────────────
    def _connect_events(self):
        self.ocx.OnEventConnect.connect(self._on_event_connect)
        self.ocx.OnReceiveTrData.connect(self._on_receive_tr_data)
        self.ocx.OnReceiveRealData.connect(self._on_receive_real_data)
        self.ocx.OnReceiveChejanData.connect(self._on_receive_chejan_data)
        self.ocx.OnReceiveMsg.connect(self._on_receive_msg)

    # ── 로그인 ────────────────────────────────────────────────────
    def login(self):
        """키움 로그인 (팝업 창) + 계좌비밀번호 입력창 자동 처리"""
        logger_api.info("키움증권 로그인 시도...")
        ret = self.ocx.dynamicCall("CommConnect()")
        if ret == 0:
            self.login_loop.exec_()
        else:
            logger_api.error(f"로그인 요청 실패: {ret}")
            return False

        if not self.connected:
            return False

        # ── 계좌비밀번호 저장 창 자동 호출 ────────────────────────────
        # KOA_Functions("ShowAccountWindow", "") 를 호출하면
        # 영웅문의 [계좌비밀번호 저장] 다이얼로그가 열림.
        # 사용자가 여기서 비밀번호를 저장하면 이후 TR 요청 시
        # 팝업(에러코드 44)이 더 이상 뜨지 않음.
        #
        # config.ACCOUNT_PASSWORD 가 설정돼 있으면 창을 열지 않음.
        if not config.ACCOUNT_PASSWORD:
            logger_api.info(
                "계좌비밀번호가 config에 없습니다. "
                "영웅문 [계좌비밀번호저장] 창을 엽니다 (직접 입력 후 저장 클릭)..."
            )
            self.ocx.dynamicCall(
                "KOA_Functions(QString, QString)",
                "ShowAccountWindow", ""
            )
            # 사용자가 비밀번호를 입력하고 창을 닫을 때까지 대기
            loop = QEventLoop()
            QTimer.singleShot(3000, loop.exit)
            loop.exec_()
            logger_api.info("계좌비밀번호 창 처리 완료")
        else:
            logger_api.info(
                f"config.ACCOUNT_PASSWORD 사용 | 계좌: {self.account_no}"
            )

        return True

    def _on_event_connect(self, err_code):
        if err_code == 0:
            self.connected = True
            self.accounts = (
                self.ocx.dynamicCall("GetLoginInfo(QString)", "ACCNO")
                .strip().rstrip(";").split(";")
            )
            self.account_no = self.accounts[0] if self.accounts else ""

            # ── GetServerGubun 반환값 정의 ──────────────────────────
            # "1"  → 모의투자 서버
            # ""   → 실서버  (빈 문자열 또는 "0")
            # ※ 키움 공식 문서 기준: 1=모의투자, 그 외=실서버
            server_type = self.ocx.dynamicCall(
                "GetLoginInfo(QString)", "GetServerGubun"
            ).strip()
            self.is_simul_server = (server_type == "1")   # True=모의투자
            server_name = "모의투자" if self.is_simul_server else "실서버"

            logger_api.info(
                f"로그인 성공 | 서버: {server_name} "
                f"(GetServerGubun='{server_type}') | 계좌: {self.account_no}"
            )
        else:
            self.connected = False
            self.is_simul_server = None
            logger_api.error(f"로그인 실패: 에러코드 {err_code}")
        self.login_loop.exit()

    # ── TR 데이터 요청 ────────────────────────────────────────────
    def set_input_value(self, key, value):
        self.ocx.dynamicCall("SetInputValue(QString, QString)", key, value)

    def comm_rq_data(self, rq_name, tr_code, prev_next, screen_no):
        self.tr_data = {}
        ret = self.ocx.dynamicCall(
            "CommRqData(QString, QString, int, QString)",
            rq_name, tr_code, prev_next, screen_no
        )
        if ret == 0:
            self.tr_loop.exec_()
        else:
            # -202 : 계좌 비밀번호 미입력 / 비밀번호 오류
            # -200 : 시세 조회 제한
            if ret == -202:
                logger_api.error(
                    f"TR 요청 실패: {rq_name}, 코드: {ret} "
                    f"— 계좌 비밀번호 오류. "
                    f"config.py의 ACCOUNT_PASSWORD를 확인하거나 "
                    f"영웅문 [계좌→계좌비밀번호저장] 에서 비밀번호를 등록하세요."
                )
            else:
                logger_api.error(f"TR 요청 실패: {rq_name}, 코드: {ret}")
            self.tr_data["_error_code"] = ret  # 호출자가 오류 여부 확인용
        return self.tr_data

    def _on_receive_tr_data(self, screen_no, rq_name, tr_code,
                            record_name, prev_next, *args):
        self.tr_remained = (prev_next == "2")
        cnt = self.ocx.dynamicCall(
            "GetRepeatCnt(QString, QString)", tr_code, rq_name
        )
        self.tr_data = {
            "rq_name": rq_name, "tr_code": tr_code,
            "record_count": cnt, "has_next": self.tr_remained, "rows": []
        }

        # rq_name 별 파싱
        parsers = {
            "업종별주가요청": (
                ["종목코드", "종목명", "현재가", "전일대비",
                 "등락율", "거래량", "거래대금", "시가총액"], cnt
            ),
            "종목기본정보": (
                ["종목코드", "종목명", "현재가", "등락율", "거래량",
                 "거래대금", "시가총액", "PER", "PBR", "외국인비율"],
                max(1, cnt)
            ),
            "일봉차트": (
                ["일자", "시가", "고가", "저가", "현재가",
                 "거래량", "거래대금"], cnt
            ),
            "투자자별매매": (
                ["일자", "개인투자자", "외국인투자자", "기관계",
                 "금융투자", "보험", "투신", "은행", "기타금융",
                 "연기금등"], cnt
            ),
        }

        if rq_name == "계좌평가잔고":
            self._parse_account_balance(tr_code, rq_name, cnt)
        elif rq_name == "예수금상세현황":
            self._parse_deposit_detail(tr_code, rq_name)
        elif rq_name in parsers:
            fields, loop_cnt = parsers[rq_name]
            for i in range(loop_cnt):
                row = {}
                for f in fields:
                    row[f] = self.ocx.dynamicCall(
                        "GetCommData(QString, QString, int, QString)",
                        tr_code, rq_name, i, f
                    ).strip()
                self.tr_data["rows"].append(row)
        else:
            for i in range(cnt):
                row = {}
                for f in ["종목코드", "종목명", "현재가", "거래량", "등락율"]:
                    row[f] = self.ocx.dynamicCall(
                        "GetCommData(QString, QString, int, QString)",
                        tr_code, rq_name, i, f
                    ).strip()
                self.tr_data["rows"].append(row)

        self.tr_loop.exit()

    def _parse_account_balance(self, tr_code, rq_name, cnt):
        # ── OPW00018 싱글 필드 ─────────────────────────────────────────
        # OPW00018은 예수금/주문가능금액을 제공하지 않는다.
        # 실제 주문가능금액은 get_orderable_cash() → OPW00001 TR로 별도 조회.
        single_fields = [
            "총매입금액", "총평가금액", "총평가손익금액",
            "총수익률(%)", "추정예탁자산",
        ]
        summary = {}
        for f in single_fields:
            summary[f] = self.ocx.dynamicCall(
                "GetCommData(QString, QString, int, QString)",
                tr_code, rq_name, 0, f
            ).strip()
        self.tr_data["summary"] = summary

        multi_fields = [
            "종목번호", "종목명", "보유수량", "매입가", "현재가",
            "평가손익", "수익률(%)", "매입금액", "평가금액"
        ]
        for i in range(cnt):
            row = {}
            for f in multi_fields:
                row[f] = self.ocx.dynamicCall(
                    "GetCommData(QString, QString, int, QString)",
                    tr_code, rq_name, i, f
                ).strip()
            self.tr_data["rows"].append(row)

    def _parse_deposit_detail(self, tr_code, rq_name):
        """OPW00001 예수금상세현황 싱글 데이터 파싱.
        d+2추정예수금, 주문가능금액 등 실제 현금 관련 필드를 파싱한다.
        """
        # KOA Studio 기준 OPW00001 주요 싱글 필드
        single_fields = [
            "주문가능금액",       # 실제 매수에 쓸 수 있는 금액 (핵심)
            "d+2추정예수금",      # 결제 완료 후 예수금 추정치
            "예수금총액",         # 현재 예수금 총액
            "출금가능금액",       # 당일 출금 가능 금액
        ]
        deposit = {}
        for f in single_fields:
            deposit[f] = self.ocx.dynamicCall(
                "GetCommData(QString, QString, int, QString)",
                tr_code, rq_name, 0, f
            ).strip()
        self.tr_data["deposit"] = deposit

    # ── 실시간 데이터 ─────────────────────────────────────────────
    def set_real_reg(self, screen_no, code_list, fid_list, opt_type="0"):
        ret = self.ocx.dynamicCall(
            "SetRealReg(QString, QString, QString, QString)",
            screen_no, code_list, fid_list, opt_type
        )
        logger_api.info(f"실시간 등록: {code_list} -> {ret}")
        return ret

    def set_real_remove(self, screen_no, code):
        self.ocx.dynamicCall(
            "SetRealRemove(QString, QString)", screen_no, code
        )

    def _on_receive_real_data(self, code, real_type, real_data):
        if real_type == "주식체결":
            data = {
                "code": code,
                "time": self._get_comm_real_data(code, 20),
                "price": abs(int(self._get_comm_real_data(code, 10) or "0")),
                "change": self._get_comm_real_data(code, 11),
                "change_rate": self._get_comm_real_data(code, 12),
                "volume": abs(int(self._get_comm_real_data(code, 15) or "0")),
                "cum_volume": abs(int(self._get_comm_real_data(code, 13) or "0")),
                "open": abs(int(self._get_comm_real_data(code, 16) or "0")),
                "high": abs(int(self._get_comm_real_data(code, 17) or "0")),
                "low": abs(int(self._get_comm_real_data(code, 18) or "0")),
            }
            if self.real_data_callback:
                self.real_data_callback(data)

    def _get_comm_real_data(self, code, fid):
        return self.ocx.dynamicCall(
            "GetCommRealData(QString, int)", code, fid
        ).strip()

    # ── 주문 ──────────────────────────────────────────────────────
    def send_order(self, rq_name, screen_no, acc_no, order_type,
                   code, qty, price, hoga_type, org_order_no=""):
        """
        order_type: 1=신규매수, 2=신규매도
        hoga_type: 00=지정가, 03=시장가
        """
        ret = self.ocx.dynamicCall(
            "SendOrder(QString, QString, QString, int, QString, "
            "int, int, QString, QString)",
            rq_name, screen_no, acc_no, order_type, code,
            qty, price, hoga_type, org_order_no
        )
        action = "매수" if order_type == 1 else "매도"
        if ret == 0:
            logger_api.info(
                f"주문성공: {rq_name} | {code} | {action} | "
                f"수량:{qty} | 가격:{price}"
            )
        else:
            logger_api.error(f"주문실패: {rq_name}, 에러:{ret}")
        return ret

    def _on_receive_chejan_data(self, gubun, item_cnt, fid_list):
        if gubun == "0":  # 체결
            code = self._get_chejan_data(9001).strip().replace("A", "")
            status = self._get_chejan_data(913).strip()
            filled_qty = self._get_chejan_data(911).strip()
            filled_price = self._get_chejan_data(910).strip()
            logger_api.info(
                f"체결: {code} | 상태:{status} | "
                f"체결수량:{filled_qty} | 체결가:{filled_price}"
            )
            self.order_result = {
                "code": code, "status": status,
                "filled_qty": filled_qty, "filled_price": filled_price,
            }
        elif gubun == "1":  # 잔고
            code = self._get_chejan_data(9001).strip().replace("A", "")
            qty = self._get_chejan_data(930).strip()
            avg = self._get_chejan_data(931).strip()
            logger_api.info(f"잔고변경: {code} | 보유:{qty} | 평균가:{avg}")

    def _get_chejan_data(self, fid):
        return self.ocx.dynamicCall("GetChejanData(int)", fid)

    def _on_receive_msg(self, screen_no, rq_name, tr_code, msg):
        logger_api.info(f"[MSG] {rq_name}: {msg}")

    # ── 유틸리티 ──────────────────────────────────────────────────
    def get_code_list_by_market(self, market_code):
        codes = self.ocx.dynamicCall(
            "GetCodeListByMarket(QString)", str(market_code)
        )
        return codes.strip().rstrip(";").split(";") if codes else []

    def get_master_code_name(self, code):
        return self.ocx.dynamicCall(
            "GetMasterCodeName(QString)", code
        ).strip()

    def get_connect_state(self):
        return self.ocx.dynamicCall("GetConnectState()") == 1

    # ── 고수준 조회 함수 ──────────────────────────────────────────
    def get_sector_stocks(self, sector_code):
        self.set_input_value("업종코드", sector_code)
        self.set_input_value("표시순서", "0")
        data = self.comm_rq_data("업종별주가요청", "OPT20002", 0, "2000")
        time.sleep(0.25)
        return data.get("rows", [])

    def get_stock_basic_info(self, code):
        self.set_input_value("종목코드", code)
        data = self.comm_rq_data("종목기본정보", "OPT10001", 0, "2001")
        time.sleep(0.25)
        return data.get("rows", [{}])[0] if data.get("rows") else {}

    def get_account_balance(self):

        self.set_input_value("계좌번호", self.account_no)

        if self.is_simul_server:
            # 모의투자: config 설정값 우선, 없으면 "0000"
            pwd = config.ACCOUNT_PASSWORD if config.ACCOUNT_PASSWORD else "0000"
        else:
            # 실서버: config 설정값 우선, 없으면 빈 문자열(영웅문 저장 비번 사용)
            pwd = config.ACCOUNT_PASSWORD
            if not pwd:
                logger_api.warning(
                    "실서버 ACCOUNT_PASSWORD 미설정 — 영웅문 저장 비밀번호로 시도합니다."
                )

        self.set_input_value("비밀번호", pwd)
        self.set_input_value("비밀번호입력매체구분", "03")
        self.set_input_value("조회구분", "1")
        data = self.comm_rq_data("계좌평가잔고", "OPW00018", 0, "2002")
        time.sleep(0.25)

        # ── -202 발생 시 공통 처리 (모의투자 + 실서버 모두 적용) ──────────
        # 원인: 영웅문 [계좌비밀번호저장]에 비밀번호가 등록되지 않았거나
        #        config.ACCOUNT_PASSWORD 값이 실제 비밀번호와 다른 경우.
        # 처리: ShowAccountWindow 팝업을 열어 사용자가 직접 저장하게 한 후 재시도.
        if data.get("_error_code") == -202:
            server_label = "모의투자" if self.is_simul_server else "실서버"
            logger_api.warning(
                f"[{server_label}] 계좌 비밀번호 오류(-202) 발생.\n"
                "  영웅문 [계좌비밀번호저장] 창을 엽니다.\n"
                "  창에서 계좌 비밀번호를 입력하고 [등록] 버튼을 누른 뒤 닫아주세요.\n"
                "  (최대 30초 대기)"
            )
            self.ocx.dynamicCall(
                "KOA_Functions(QString, QString)",
                "ShowAccountWindow", ""
            )
            # 사용자가 비밀번호를 입력하고 창을 닫을 때까지 최대 30초 대기
            loop = QEventLoop()
            QTimer.singleShot(30000, loop.exit)
            loop.exec_()
            logger_api.info("비밀번호 저장 창 닫힘 — 계좌조회 재시도...")

            # 재시도: 저장된 비밀번호(빈 문자열)로 키움이 내부적으로 처리
            self.set_input_value("계좌번호", self.account_no)
            self.set_input_value("비밀번호", "")
            self.set_input_value("비밀번호입력매체구분", "03")
            self.set_input_value("조회구분", "1")
            data = self.comm_rq_data("계좌평가잔고", "OPW00018", 0, "2002")
            time.sleep(0.25)

            if data.get("_error_code") == -202:
                logger_api.error(
                    "재시도 후에도 -202 오류 지속.\n"
                    "  영웅문 HTS에서 [계좌] → [계좌비밀번호저장]을\n"
                    "  직접 열어 비밀번호를 다시 저장한 후 재실행해주세요."
                )

        return data

    def get_orderable_cash(self) -> int:
        """OPW00001 예수금상세현황으로 실제 주문가능금액 조회.

        OPW00018(계좌평가잔고)은 예수금/주문가능금액을 제공하지 않으므로
        이 함수를 별도로 호출해야 정확한 현금 잔고를 얻을 수 있다.

        Returns:
            주문가능금액 (원). 조회 실패 시 -1 반환.
        """
        self.set_input_value("계좌번호", self.account_no)
        self.set_input_value("비밀번호", "")         # 영웅문 저장 비밀번호 사용
        self.set_input_value("비밀번호입력매체구분", "00")
        self.set_input_value("조회구분", "2")         # 1=추정조회, 2=일반조회
        data = self.comm_rq_data("예수금상세현황", "OPW00001", 0, "2005")
        time.sleep(0.25)

        if "_error_code" in data:
            logger_api.error(
                f"OPW00001 조회 실패: 코드 {data['_error_code']} — "
                "주문가능금액을 가져올 수 없습니다."
            )
            return -1

        deposit = data.get("deposit", {})
        raw = deposit.get("주문가능금액", "").strip()
        if raw:
            val = abs(int(raw))
            logger_api.info(f"OPW00001 주문가능금액: {val:,}원")
            return val

        logger_api.warning("OPW00001 주문가능금액 필드 빈값 — 0 반환")
        return 0

    def get_daily_chart(self, code, date=""):
        self.set_input_value("종목코드", code)
        self.set_input_value(
            "기준일자", date or datetime.now().strftime("%Y%m%d")
        )
        self.set_input_value("수정주가구분", "1")
        data = self.comm_rq_data("일봉차트", "OPT10081", 0, "2003")
        time.sleep(0.25)
        return data.get("rows", [])

    def get_investor_trading(self, code):
        self.set_input_value("종목코드", code)
        self.set_input_value("시작일자", "")
        self.set_input_value("종료일자", "")
        data = self.comm_rq_data("투자자별매매", "OPT10059", 0, "2004")
        time.sleep(0.25)
        return data.get("rows", [])

    def buy_market_order(self, code, qty):
        return self.send_order(
            "시장가매수", "3001", self.account_no, 1, code, qty, 0, "03"
        )

    def sell_market_order(self, code, qty):
        return self.send_order(
            "시장가매도", "3002", self.account_no, 2, code, qty, 0, "03"
        )


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Part 2. 산업별 주도주 스크리닝 엔진                                ║
# ╚════════════════════════════════════════════════════════════════════╝

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
    foreign_net_buy: int = 0
    institution_net_buy: int = 0
    composite_score: float = 0.0
    rank_details: Dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"[{self.sector}] {self.name}({self.code}) "
            f"점수:{self.composite_score:.2f} "
            f"등락:{self.change_rate:+.2f}% "
            f"거래대금:{self.trading_value:,}백만"
        )


class SectorLeaderScreener:
    """산업별 주도주 스크리닝 엔진"""

    def __init__(self, kiwoom_api: KiwoomAPI):
        self.api = kiwoom_api
        self.candidates: Dict[str, List[StockCandidate]] = {}
        self.leaders: Dict[str, StockCandidate] = {}
        self.screening_time = None

    def run_full_screening(self) -> Dict[str, StockCandidate]:
        """전체 스크리닝 프로세스"""
        logger_scr.info("=" * 60)
        logger_scr.info("주도주 스크리닝 시작")
        logger_scr.info("=" * 60)
        self.screening_time = datetime.now()

        self._collect_sector_stocks()
        self._apply_filters()
        self._enrich_investor_data()
        self._calculate_composite_scores()
        self._select_leaders()
        final = self._rank_final_portfolio()

        logger_scr.info("=" * 60)
        logger_scr.info(f"스크리닝 완료: {len(final)}개 섹터 주도주 선정")
        for sector, stock in final.items():
            logger_scr.info(f"  {stock}")
        logger_scr.info("=" * 60)
        return final

    def _collect_sector_stocks(self):
        """각 섹터 종목 수집"""
        self.candidates = {}
        for name, code in config.SECTOR_CODES.items():
            logger_scr.info(f"[수집] {name} (코드: {code})")
            try:
                stocks = self.api.get_sector_stocks(code)
                cands = []
                for s in stocks[:30]:
                    try:
                        c = StockCandidate(
                            code=s.get("종목코드", "").strip(),
                            name=s.get("종목명", "").strip(),
                            sector=name,
                            price=abs(int(s.get("현재가", "0").strip() or "0")),
                            change_rate=float(
                                s.get("등락율", "0").strip() or "0"
                            ),
                            volume=abs(int(
                                s.get("거래량", "0").strip() or "0"
                            )),
                            trading_value=abs(int(
                                s.get("거래대금", "0").strip() or "0"
                            )),
                            market_cap=abs(int(
                                s.get("시가총액", "0").strip() or "0"
                            )),
                        )
                        if c.code and c.price > 0:
                            if c.market_cap > 0:
                                c.turnover_rate = (
                                    c.trading_value / c.market_cap * 100
                                )
                            cands.append(c)
                    except (ValueError, TypeError):
                        continue
                self.candidates[name] = cands
                logger_scr.info(f"  → {len(cands)}개 종목 수집")
            except Exception as e:
                logger_scr.error(f"  → 수집 실패: {e}")
                self.candidates[name] = []

    def _apply_filters(self):
        """최소 조건 필터링"""
        for name in list(self.candidates.keys()):
            self.candidates[name] = [
                c for c in self.candidates[name]
                if (c.market_cap >= config.MIN_MARKET_CAP
                    and c.trading_value >= config.MIN_TRADING_VOLUME * 100
                    and c.price >= config.MIN_PRICE
                    and len(c.code) == 6)
            ]

    def _enrich_investor_data(self):
        """외국인/기관 수급 보강 (상위 10개만)"""
        for name, cands in self.candidates.items():
            top = sorted(
                cands, key=lambda x: x.trading_value, reverse=True
            )[:10]
            for c in top:
                try:
                    data = self.api.get_investor_trading(c.code)
                    if data:
                        c.foreign_net_buy = int(
                            data[0].get("외국인투자자", "0").strip() or "0"
                        )
                        c.institution_net_buy = int(
                            data[0].get("기관계", "0").strip() or "0"
                        )
                except Exception:
                    pass

    def _calculate_composite_scores(self):
        """복합 스코어링"""
        for name, cands in self.candidates.items():
            n = len(cands)
            if n < 2:
                if cands:
                    cands[0].composite_score = 100
                continue

            rankings = {
                "volume_rank": sorted(
                    cands, key=lambda x: x.trading_value, reverse=True),
                "price_change_rank": sorted(
                    cands, key=lambda x: x.change_rate, reverse=True),
                "foreign_buy_rank": sorted(
                    cands, key=lambda x: x.foreign_net_buy, reverse=True),
                "institution_buy_rank": sorted(
                    cands, key=lambda x: x.institution_net_buy, reverse=True),
                "market_cap_rank": sorted(
                    cands, key=lambda x: x.market_cap, reverse=True),
                "turnover_rank": sorted(
                    cands, key=lambda x: x.turnover_rate, reverse=True),
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
                best = max(cands, key=lambda x: x.composite_score)
                self.leaders[name] = best

    def _rank_final_portfolio(self) -> Dict[str, StockCandidate]:
        """전체 주도주 상위 N개 선정"""
        if not self.leaders:
            return {}
        ranked = sorted(
            self.leaders.items(),
            key=lambda x: x[1].composite_score, reverse=True
        )
        return dict(ranked[:config.MAX_SECTORS])

    def apply_momentum_filter(
        self, leaders: Dict[str, StockCandidate], days: int = 5
    ) -> Dict[str, StockCandidate]:
        """최근 N일 모멘텀 필터"""
        filtered = {}
        for sector, leader in leaders.items():
            try:
                daily = self.api.get_daily_chart(leader.code)
                if len(daily) >= days:
                    prices = [
                        abs(int(d.get("현재가", "0").strip() or "0"))
                        for d in daily[:days]
                    ]
                    prices = [p for p in prices if p > 0]
                    if len(prices) >= 2:
                        avg = sum(prices) / len(prices)
                        if prices[0] >= avg * 0.98:
                            filtered[sector] = leader
                            continue
                        logger_scr.info(
                            f"[모멘텀 필터] {leader.name} 제외"
                        )
                        continue
                filtered[sector] = leader
            except Exception:
                filtered[sector] = leader
        return filtered

    def generate_report(self) -> str:
        """스크리닝 리포트"""
        lines = ["=" * 70]
        lines.append(f"  주도주 스크리닝 리포트 | {self.screening_time}")
        lines.append("=" * 70)
        for sector, ld in self.leaders.items():
            lines.append(f"\n[{sector}]")
            lines.append(f"  종목: {ld.name} ({ld.code})")
            lines.append(f"  현재가: {ld.price:,}원 | 등락률: {ld.change_rate:+.2f}%")
            lines.append(f"  거래대금: {ld.trading_value:,}백만 | 시총: {ld.market_cap:,}억")
            lines.append(f"  외국인: {ld.foreign_net_buy:,}주 | 기관: {ld.institution_net_buy:,}주")
            lines.append(f"  복합점수: {ld.composite_score:.2f}")
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Part 3. 포트폴리오 매니저                                         ║
# ╚════════════════════════════════════════════════════════════════════╝

logger_pf = logging.getLogger("Portfolio")


class OrderAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Position:
    """보유 포지션"""
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
            self.unrealized_pnl_pct = (price - self.avg_price) / self.avg_price
            if self.unrealized_pnl_pct >= config.PROFIT_TARGET_PERCENT:
                self.target_reached = True


@dataclass
class RebalanceOrder:
    """리밸런싱 주문"""
    action: OrderAction
    code: str
    name: str
    sector: str
    quantity: int
    target_weight: float
    reason: str
    priority: int = 0


class PortfolioManager:
    """포트폴리오 관리 엔진"""

    def __init__(self, kiwoom_api: KiwoomAPI):
        self.api = kiwoom_api
        self.positions: Dict[str, Position] = {}
        self.total_assets = 0
        self.available_cash = 0
        self.total_invested = 0
        self.daily_start_value = 0
        self.trade_history = []

    # ── 계좌 동기화 ───────────────────────────────────────────────
    def sync_account(self) -> bool:
        """계좌 동기화. 성공 시 True, 실패 시 False 반환."""
        logger_pf.info("계좌 동기화...")
        try:
            balance = self.api.get_account_balance()

            # ── TR 요청 자체가 실패한 경우 감지 ──────────────────────
            if "_error_code" in balance:
                err = balance["_error_code"]
                if err == -202:
                    logger_pf.critical(
                        "계좌 동기화 실패 (오류코드 -202) — 계좌 비밀번호가 올바르지 않습니다.\n"
                        "  해결방법 ①: config.py → ACCOUNT_PASSWORD = '계좌비밀번호4자리'\n"
                        "  해결방법 ②: 영웅문 HTS → [계좌] → [계좌비밀번호저장] 에서 등록"
                    )
                else:
                    logger_pf.error(f"계좌 동기화 실패 (오류코드 {err})")
                return False

            summary = balance.get("summary", {})
            self.total_assets = abs(int(
                summary.get("추정예탁자산", "0").strip() or "0"
            ))
            self.total_invested = abs(int(
                summary.get("총매입금액", "0").strip() or "0"
            ))

            # ── 주문가능금액: OPW00001 TR로 정확한 현금 조회 ──────────────
            # OPW00018은 예수금/주문가능금액 미제공.
            # 폴백: 추정예탁자산 - 총평가금액(현재가 기준)
            #   → 총매입금액(최초 매입가 기준) 대신 총평가금액(현재가 기준)을 사용해
            #      주가 변동에 의한 오차를 줄임.
            orderable = self.api.get_orderable_cash()
            if orderable >= 0:
                self.available_cash = orderable
                cash_source = "OPW00001(주문가능금액)"
            else:
                # OPW00001 조회 실패 시 폴백
                total_eval = abs(int(
                    summary.get("총평가금액", "0").strip() or "0"
                ))
                if total_eval > 0:
                    self.available_cash = max(0, self.total_assets - total_eval)
                    cash_source = "추정예탁자산-총평가금액(폴백)"
                else:
                    self.available_cash = max(0, self.total_assets - self.total_invested)
                    cash_source = "추정예탁자산-총매입금액(폴백2)"
                logger_pf.warning(
                    f"OPW00001 조회 실패 — {cash_source}으로 현금 계산: "
                    f"{self.available_cash:,}원"
                )

            logger_pf.info(
                f"현금 상세 | {cash_source}: {self.available_cash:,}원 "
                f"| 추정예탁자산: {self.total_assets:,}원 "
                f"| 총매입금액(최초매입가): {self.total_invested:,}원"
            )

            current_codes = set()
            for row in balance.get("rows", []):
                code = row.get("종목번호", "").strip().replace("A", "")
                if not code:
                    continue
                current_codes.add(code)
                qty = abs(int(row.get("보유수량", "0").strip() or "0"))
                avg = abs(int(row.get("매입가", "0").strip() or "0"))
                cur = abs(int(row.get("현재가", "0").strip() or "0"))

                if code in self.positions:
                    p = self.positions[code]
                    p.quantity, p.avg_price = qty, avg
                    p.update_price(cur)
                else:
                    self.positions[code] = Position(
                        code=code,
                        name=row.get("종목명", "").strip(),
                        sector="미분류", quantity=qty,
                        avg_price=avg, current_price=cur,
                        trailing_high=cur,
                    )

            for code in set(self.positions.keys()) - current_codes:
                del self.positions[code]

            logger_pf.info(
                f"동기화 완료 | 총자산:{self.total_assets:,} | "
                f"현금:{self.available_cash:,} | "
                f"보유:{len(self.positions)}종목"
            )
            return True
        except Exception as e:
            logger_pf.error(f"계좌 동기화 실패: {e}")
            return False

    # ── 리밸런싱 계산 ─────────────────────────────────────────────
    def calculate_rebalance(
        self, new_leaders: Dict[str, StockCandidate]
    ) -> List[RebalanceOrder]:
        """새 주도주 기반 리밸런싱 주문 생성 (매도 우선)"""
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
                eq = 1.0 / n
                sc = ld.composite_score / total_score
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
                weights[c] = weights[c] / wsum * config.TOTAL_INVESTMENT_RATIO

        # 3) 매수/비중조절
        for sector, ld in new_leaders.items():
            tw = weights.get(ld.code, 0)
            target_val = self.total_assets * tw

            if ld.code in self.positions:
                pos = self.positions[ld.code]
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
                buy_qty = int(target_val / ld.price) if ld.price > 0 else 0
                if buy_qty > 0:
                    orders.append(RebalanceOrder(
                        OrderAction.BUY, ld.code, ld.name, sector,
                        buy_qty, tw,
                        f"신규편입 (비중:{tw:.1%})", 3
                    ))

        orders.sort(key=lambda x: x.priority, reverse=True)

        # 매수 자금 검증
        sell_proceeds = sum(
            self.positions[o.code].current_price * o.quantity
            for o in orders
            if o.action == OrderAction.SELL and o.code in self.positions
        )
        avail = self.available_cash + sell_proceeds
        validated = []
        for o in orders:
            if o.action == OrderAction.BUY:
                price = self._get_price(o.code)
                cost = o.quantity * price
                if cost <= avail:
                    validated.append(o)
                    avail -= cost
                else:
                    reduced = int(avail / price) if price > 0 else 0
                    if reduced > 0:
                        o.quantity = reduced
                        o.reason += f" (축소:{reduced}주)"
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
            return self.positions[code].current_price
        info = self.api.get_stock_basic_info(code)
        return abs(int(info.get("현재가", "0").strip() or "0"))

    # ── 주문 실행 ─────────────────────────────────────────────────
    def execute_rebalance(self, orders: List[RebalanceOrder]):
        for order in orders:
            try:
                if order.action == OrderAction.SELL:
                    ret = self.api.sell_market_order(order.code, order.quantity)
                elif order.action == OrderAction.BUY:
                    ret = self.api.buy_market_order(order.code, order.quantity)
                else:
                    continue

                if ret == 0:
                    self.trade_history.append({
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "action": order.action.value,
                        "code": order.code, "name": order.name,
                        "sector": order.sector, "quantity": order.quantity,
                        "reason": order.reason,
                    })
                time.sleep(0.3)
            except Exception as e:
                logger_pf.error(f"주문실행 실패: {order.name} - {e}")

    def update_position_price(self, code: str, price: int, high: int = 0):
        if code in self.positions:
            self.positions[code].update_price(price, high)

    # ── 현황 조회 ─────────────────────────────────────────────────
    def get_portfolio_summary(self) -> Dict:
        total_mv = sum(p.market_value for p in self.positions.values())
        total_cost = sum(p.cost_basis for p in self.positions.values())
        pnl = total_mv - total_cost
        pnl_pct = (pnl / total_cost * 100) if total_cost > 0 else 0
        return {
            "total_assets": self.total_assets,
            "cash": self.available_cash,
            "invested": total_mv,
            "cost_basis": total_cost,
            "unrealized_pnl": pnl,
            "unrealized_pnl_pct": pnl_pct,
            "position_count": len(self.positions),
            "positions": {
                code: {
                    "name": p.name, "sector": p.sector,
                    "qty": p.quantity, "avg_price": p.avg_price,
                    "current_price": p.current_price,
                    "pnl": p.unrealized_pnl,
                    "pnl_pct": p.unrealized_pnl_pct * 100,
                    "day_high": p.day_high,
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
                f"{info['qty']}주 | 평균:{info['avg_price']:,} "
                f"현재:{info['current_price']:,} | "
                f"손익:{info['pnl']:+,}원({info['pnl_pct']:+.2f}%)"
            )
        logger_pf.info("=" * 70)


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Part 4. 트레일링 스탑 매도 엔진                                    ║
# ╚════════════════════════════════════════════════════════════════════╝

logger_sell = logging.getLogger("SellEngine")


class SellSignal(Enum):
    HOLD = "HOLD"
    TRAILING_STOP = "TRAILING_STOP"
    ENHANCED_TRAILING = "ENHANCED_TRAILING"
    STOP_LOSS = "STOP_LOSS"
    FORCE_CLOSE = "FORCE_CLOSE"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"


class TrailingStopEngine:
    """
    트레일링 스탑 매도 엔진 (당일 최고가 추적 → 고점 대비 하락 시 매도)

    전략:
      1) 실시간 체결가로 trailing_high 갱신
      2) 고점 대비 TRAILING_STOP_PERCENT 이상 하락 → 매도
      3) 목표수익 도달 후 → 더 타이트한 ENHANCED_TRAILING
      4) 손절매 라인 → 즉시 매도
      5) 15:20 강제 청산 (데이트레이딩 모드)
    """

    def __init__(self, kiwoom_api: KiwoomAPI, portfolio: PortfolioManager):
        self.api = kiwoom_api
        self.portfolio = portfolio
        self.sold_today: Dict[str, dict] = {}
        self.monitoring_active = False
        self.on_sell_executed: Optional[Callable] = None
        self.daily_sell_profit = 0
        self.daily_sell_count = 0

    def on_real_tick(self, data: dict):
        """실시간 체결 수신 → 매도 판정"""
        if not self.monitoring_active:
            return
        code = data.get("code", "")
        price = data.get("price", 0)
        high = data.get("high", 0)
        tick_time = data.get("time", "")

        if code not in self.portfolio.positions:
            return

        pos = self.portfolio.positions[code]
        pos.update_price(price, high)
        signal = self._evaluate(pos, tick_time)
        if signal != SellSignal.HOLD:
            self._execute_sell(pos, signal)

    def _evaluate(self, pos: Position, tick_time: str) -> SellSignal:
        """매도 신호 판정 (우선순위: 일일한도 > 강제청산 > 손절 > 트레일링)"""
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
                f"[손절매] {pos.name} | 손실: {pos.unrealized_pnl_pct:.2%}"
            )
            return SellSignal.STOP_LOSS

        # 트레일링 스탑
        if pos.trailing_high > 0 and pos.current_price > 0:
            drop = (pos.trailing_high - pos.current_price) / pos.trailing_high
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
        if ret == 0:
            self.sold_today[pos.code] = {
                "name": pos.name, "signal": signal.value,
                "qty": pos.quantity,
                "sell_price": pos.current_price,
                "avg_price": pos.avg_price,
                "day_high": pos.day_high,
                "trailing_high": pos.trailing_high,
                "pnl_pct": pos.unrealized_pnl_pct,
                "pnl_amount": pos.unrealized_pnl,
                "time": datetime.now().strftime("%H:%M:%S"),
            }
            self.daily_sell_count += 1
            self.daily_sell_profit += pos.unrealized_pnl
            if self.on_sell_executed:
                self.on_sell_executed(pos.code, signal, pos)

    def _check_daily_loss(self) -> bool:
        if self.portfolio.daily_start_value <= 0:
            return False
        total = sum(
            p.market_value for p in self.portfolio.positions.values()
        ) + self.portfolio.available_cash
        ret = (total - self.portfolio.daily_start_value) / self.portfolio.daily_start_value
        if ret <= config.MAX_DAILY_LOSS_PERCENT:
            logger_sell.critical(f"[일일 손실 한도 초과] {ret:.2%}")
            return True
        return False

    def force_sell_all(self, reason="강제청산"):
        logger_sell.warning(f"전량 청산: {reason}")
        for code, pos in list(self.portfolio.positions.items()):
            if pos.quantity > 0 and code not in self.sold_today:
                self._execute_sell(pos, SellSignal.FORCE_CLOSE)

    def start_monitoring(self):
        self.monitoring_active = True
        if self.portfolio.positions:
            codes = ";".join(self.portfolio.positions.keys())
            fids = ";".join(
                str(f) for f in config.REAL_TIME_FIDS.values()
            )
            self.api.set_real_reg("5000", codes, fids, "0")
        logger_sell.info(
            f"모니터링 시작 ({len(self.portfolio.positions)}종목)"
        )

    def stop_monitoring(self):
        self.monitoring_active = False
        for code in self.portfolio.positions:
            self.api.set_real_remove("5000", code)
        logger_sell.info("모니터링 중지")

    def get_sell_report(self) -> str:
        lines = ["=" * 60, "  당일 매도 성과", "=" * 60]
        if not self.sold_today:
            lines.append("  매도 없음")
        else:
            total = 0
            for code, info in self.sold_today.items():
                pnl = info["pnl_amount"]
                total += pnl
                eff = 0
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
            lines.append(f"\n  총 {len(self.sold_today)}건 | 총손익: {total:+,}원")
        lines.append("=" * 60)
        return "\n".join(lines)

    def reset_daily(self):
        self.sold_today = {}
        self.daily_sell_profit = 0
        self.daily_sell_count = 0


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Part 5. 리포트 & DB                                              ║
# ╚════════════════════════════════════════════════════════════════════╝

logger_rpt = logging.getLogger("Reporter")


class TradingDB:
    """SQLite 거래 데이터베이스"""

    def __init__(self, db_path: str = config.DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
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
        self.conn.execute(
            "INSERT INTO trades (datetime,action,code,name,sector,"
            "quantity,price,amount,signal_type,reason) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (trade.get("datetime", datetime.now().isoformat()),
             trade.get("action"), trade.get("code"),
             trade.get("name"), trade.get("sector"),
             trade.get("quantity"), trade.get("price", 0),
             trade.get("amount", 0), trade.get("signal_type", ""),
             trade.get("reason", ""))
        )
        self.conn.commit()

    def log_daily_summary(self, s: dict):
        today = datetime.now().strftime("%Y-%m-%d")
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
        for code, info in positions.items():
            self.conn.execute(
                "INSERT OR REPLACE INTO daily_portfolio "
                "(date,code,name,sector,quantity,avg_price,close_price,"
                "day_high,pnl_amount,pnl_pct,weight,composite_score) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (today, code, info.get("name"), info.get("sector"),
                 info.get("qty"), info.get("avg_price"),
                 info.get("current_price"), info.get("day_high",0),
                 info.get("pnl"), info.get("pnl_pct"),
                 info.get("weight"), info.get("composite_score",0))
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
                "trading_days": r[0], "total_pnl": r[1] or 0,
                "avg_daily_return": r[2] or 0,
                "best_day": r[3] or 0, "worst_day": r[4] or 0,
                "win_days": r[5] or 0,
                "win_rate": (r[5] or 0) / max(r[0], 1) * 100,
                "total_trades": r[6] or 0,
            }
        return {}

    def close(self):
        self.conn.close()


class DailyReporter:
    """일일 리포트 생성"""

    def __init__(self, db: TradingDB):
        self.db = db

    def generate_daily_report(self, portfolio_summary, sell_report,
                               screening_report, trade_history) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        cum = self.db.get_cumulative_stats()
        ps = portfolio_summary

        lines = []
        lines.append("╔" + "═" * 68 + "╗")
        lines.append("║" + f"  일일 자동매매 리포트 | {today}".center(68) + "║")
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
                f"{info['qty']}주 | {info['avg_price']:,}→"
                f"{info['current_price']:,} | "
                f"{info['pnl']:+,}원({info['pnl_pct']:+.2f}%)"
            )

        lines.append(f"\n{sell_report}")

        if cum:
            lines.append(f"\n◆ 누적성과")
            lines.append(f"  운용:{cum['trading_days']}일 | "
                          f"총손익:{cum['total_pnl']:+,.0f}원 | "
                          f"승률:{cum['win_rate']:.1f}%")

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
        path = os.path.join(config.REPORT_DIR, f"report_{today}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        logger_rpt.info(f"리포트 저장: {path}")
        return report


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Part 6. 메인 엔진 & 실행                                         ║
# ╚════════════════════════════════════════════════════════════════════╝

logger_main = logging.getLogger("MainEngine")


def setup_logging():
    """로깅 초기화"""
    os.makedirs(config.LOG_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    fmt = logging.Formatter(
        "[%(asctime)s] %(name)-12s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S"
    )
    fh = logging.FileHandler(
        os.path.join(config.LOG_DIR, f"trading_{today}.log"),
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
    메인 자동매매 엔진

    ┌──────────────────────────────────────────────────┐
    │ 08:30  스크리닝 → 섹터별 주도주 선정             │
    │ 09:05  리밸런싱 → 탈락종목 매도, 신규종목 매수   │
    │ 09:10  모니터링 → 트레일링 스탑 매도 감시         │
    │ 15:20  강제청산 → 미매도 종목 정리               │
    │ 16:00  리포트   → 일일 성과 저장                 │
    └──────────────────────────────────────────────────┘
    """

    def __init__(self):
        self.api = KiwoomAPI()
        self.screener = SectorLeaderScreener(self.api)
        self.portfolio = PortfolioManager(self.api)
        self.sell_engine = TrailingStopEngine(self.api, self.portfolio)
        self.db = TradingDB()
        self.reporter = DailyReporter(self.db)

        self.today_leaders: Dict[str, StockCandidate] = {}
        self.is_running = False
        self.day_trading_mode = True  # True=당일청산, False=스윙

        self.timer = QTimer()
        self.timer.timeout.connect(self._check_schedule)

        self.api.real_data_callback = self._on_real_data
        self.sell_engine.on_sell_executed = self._on_sell_complete

    # ── 시작 / 종료 ───────────────────────────────────────────────
    def start(self):
        logger_main.info("╔════════════════════════════════════════╗")
        logger_main.info("║  산업별 주도주 자동매매 시스템 시작      ║")
        logger_main.info("╚════════════════════════════════════════╝")

        # [1/4] 로그인
        logger_main.info("[1/4] 키움증권 로그인 중...")
        if not self.api.login():
            logger_main.critical("  ❌ 로그인 실패 — 시스템 종료")
            return False
        logger_main.info("  ✅ 로그인 성공")

        # [2/4] 서버 이중검증 ─────────────────────────────────────────
        # GetServerGubun: "1" = 모의투자, "" 또는 "0" = 실서버
        # config.IS_REAL_SERVER: True = 실서버 의도, False = 모의투자 의도
        logger_main.info("[2/4] 서버 이중 검증 중...")

        actual_is_simul = self.api.is_simul_server   # True=모의투자, False=실서버
        config_is_real  = config.IS_REAL_SERVER       # True=실서버 의도

        # actual_is_simul=True(모의)  ↔ config_is_real=False(모의 의도) → 일치
        # actual_is_simul=False(실서버) ↔ config_is_real=True(실서버 의도) → 일치
        matched = (actual_is_simul != config_is_real)

        actual_label = "🟡 모의투자 (SIMUL)" if actual_is_simul else "🔴 실서버 (REAL)"
        config_label = (
            "IS_REAL_SERVER = False (모의투자 의도)"
            if not config_is_real
            else "IS_REAL_SERVER = True  (실서버 의도)"
        )

        logger_main.info(f"  실제 연결 서버 : {actual_label}")
        logger_main.info(f"  config 설정값  : {config_label}")

        if not matched:
            logger_main.critical(
                "  ❌ 서버 불일치! config.IS_REAL_SERVER 설정을 확인하세요.\n"
                "     • 모의투자로 로그인했다면 → IS_REAL_SERVER = False\n"
                "     • 실서버로 로그인했다면   → IS_REAL_SERVER = True\n"
                "  시스템을 안전하게 종료합니다."
            )
            return False

        logger_main.info("  ✅ config 설정과 실제 서버가 일치합니다.")

        if not actual_is_simul:  # 실서버 진입 시 경고
            logger_main.warning(
                "  ⚠️  실서버 모드입니다. 실제 자금으로 거래됩니다!"
            )

        # [3/4] 계좌 동기화
        logger_main.info("[3/4] 계좌 정보 확인 중...")
        if not self.portfolio.sync_account():
            logger_main.critical(
                "  ❌ 계좌 동기화 실패 — 시스템 종료\n"
                "  계좌 비밀번호를 확인하세요:\n"
                "    방법 ①: config.py → ACCOUNT_PASSWORD = '비밀번호4자리'\n"
                "    방법 ②: 영웅문 HTS → [계좌] → [계좌비밀번호저장] 등록 (권장)"
            )
            return False
        self.portfolio.daily_start_value = self.portfolio.total_assets
        logger_main.info(f"  시작자산: {self.portfolio.total_assets:,}원")

        # [4/4] 시스템 가동
        logger_main.info("[4/4] 자동매매 시스템 가동...")
        self.is_running = True
        self.timer.start(1000)
        self._initial_check()
        logger_main.info("  ✅ 시스템 가동 중 — 스케줄러 대기 중...")
        return True

    def stop(self):
        logger_main.info("시스템 종료...")
        self.is_running = False
        self.timer.stop()
        self.sell_engine.stop_monitoring()
        self.db.close()

    # ── 스케줄러 ──────────────────────────────────────────────────
    def _check_schedule(self):
        if not self.is_running or config.EMERGENCY_STOP:
            return
        now = datetime.now().strftime("%H%M%S")
        hm = now[:4]

        if hm == config.SCHEDULE["screening"] and now[4:6] == "00":
            self._task_screening()
        elif now == config.BUY_START_TIME:
            self._task_rebalance()
        elif hm == config.SCHEDULE["monitoring_start"] and now[4:6] == "00":
            self._task_start_monitoring()
        elif hm == config.SCHEDULE["force_sell"] and now[4:6] == "00":
            if self.day_trading_mode:
                self._task_force_close()
        elif hm == config.SCHEDULE["market_close"] and now[4:6] == "00":
            self._task_market_close()
        elif hm == config.SCHEDULE["daily_report"] and now[4:6] == "00":
            self._task_daily_report()

    def _initial_check(self):
        now = datetime.now().strftime("%H%M")
        if "0830" <= now < "0900":
            self._task_screening()
        elif "0900" <= now < "0930":
            self._task_screening()
            self._task_rebalance()
            self._task_start_monitoring()
        elif "0930" <= now < "1520":
            self.portfolio.sync_account()
            self._task_start_monitoring()

    # ── 태스크들 ──────────────────────────────────────────────────
    def _task_screening(self):
        logger_main.info("━━ 주도주 스크리닝 시작 ━━")
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
                        "action": o.action.value,
                        "code": o.code, "name": o.name,
                        "sector": o.sector, "quantity": o.quantity,
                        "reason": o.reason,
                    })
        except Exception as e:
            logger_main.error(f"리밸런싱 실패: {e}", exc_info=True)

    def _task_start_monitoring(self):
        logger_main.info("━━ 매도 모니터링 시작 ━━")
        self.portfolio.sync_account()
        if not self.portfolio.positions:
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
        logger_main.info("━━ 일일 리포트 ━━")
        try:
            ps = self.portfolio.get_portfolio_summary()
            sr = self.sell_engine.get_sell_report()
            scr = (self.screener.generate_report()
                   if self.today_leaders else "미실행")
            report = self.reporter.generate_daily_report(
                ps, sr, scr, self.portfolio.trade_history
            )
            logger_main.info(report)

            pnl = self.portfolio.total_assets - self.portfolio.daily_start_value
            pnl_pct = (
                pnl / self.portfolio.daily_start_value * 100
                if self.portfolio.daily_start_value > 0 else 0
            )
            cum = self.db.get_cumulative_stats()
            self.db.log_daily_summary({
                "total_assets": self.portfolio.total_assets,
                "cash": self.portfolio.available_cash,
                "invested": self.portfolio.total_invested,
                "daily_pnl": pnl, "daily_pnl_pct": pnl_pct,
                "cumulative_pnl": cum.get("total_pnl", 0) + pnl,
                "trade_count": len(self.portfolio.trade_history),
                "sector_count": len(self.today_leaders),
                "screening_report": scr,
            })
            self.db.log_portfolio_snapshot(ps.get("positions", {}))
            self.sell_engine.reset_daily()
            self.portfolio.trade_history = []
        except Exception as e:
            logger_main.error(f"리포트 실패: {e}", exc_info=True)

    # ── 콜백 ─────────────────────────────────────────────────────
    def _on_real_data(self, data: dict):
        code = data.get("code")
        if code:
            self.portfolio.update_position_price(
                code, data.get("price", 0), data.get("high", 0)
            )
            self.sell_engine.on_real_tick(data)

    def _on_sell_complete(self, code, signal, position):
        logger_main.info(
            f"[매도완료] {position.name} | {signal.value} | "
            f"손익:{position.unrealized_pnl:+,}원"
        )
        self.db.log_trade({
            "datetime": datetime.now().isoformat(),
            "action": "SELL", "code": code,
            "name": position.name, "sector": position.sector,
            "quantity": position.quantity,
            "price": position.current_price,
            "amount": position.market_value,
            "signal_type": signal.value,
            "reason": f"고점:{position.trailing_high:,}→매도:{position.current_price:,}",
        })


# ═══════════════════════════════════════════════════════════════════
#  실행 진입점
# ═══════════════════════════════════════════════════════════════════
def main():
    setup_logging()
    app = QApplication.instance() or QApplication(sys.argv)
    engine = TradingEngine()

    def on_exit(sig, frame):
        logger_main.info("종료 시그널 수신")
        engine.stop()
        app.quit()

    signal.signal(signal.SIGINT, on_exit)

    if engine.start():
        logger_main.info("이벤트 루프 시작 - 자동매매 운영 중...")
        sys.exit(app.exec_())
    else:
        logger_main.critical("시스템 시작 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
