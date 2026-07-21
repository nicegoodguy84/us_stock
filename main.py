import base64
import io
import os
import platform
import re
import sys
import warnings
from datetime import datetime, timedelta, timezone

import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

warnings.filterwarnings("ignore")

# OS별 폰트 동적 설정 (GitHub Actions 리눅스 환경 대응)
system_name = platform.system()
if system_name == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
elif system_name == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    # Linux (GitHub Actions)
    plt.rcParams["font.family"] = "NanumGothic"

plt.rcParams["axes.unicode_minus"] = False


def get_sp500_tickers_with_names():
    """위키피디아에서 S&P 500 종목 티커 및 회사명 가져오기"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            table = pd.read_html(response.text)
            df = table[0]
            df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
            ticker_to_name = dict(zip(df["Symbol"], df["Security"]))
            return df["Symbol"].tolist(), ticker_to_name
        else:
            raise Exception()
    except Exception:
        fallback_tickers = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "AVGO",
            "COST",
            "AMD",
            "PLTR",
            "ANET",
            "NFLX",
        ]
        fallback_names = {t: t for t in fallback_tickers}
        return fallback_tickers, fallback_names


def get_nasdaq100_tickers_with_names():
    """위키피디아에서 NASDAQ 100 종목 티커 및 회사명 가져오기"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            table = pd.read_html(response.text)
            df = table[0]
            df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)
            ticker_to_name = dict(zip(df["Ticker"], df["Company"]))
            return df["Ticker"].tolist(), ticker_to_name
        else:
            raise Exception()
    except Exception:
        fallback_tickers = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "AVGO",
            "COST",
            "AMD",
            "NFLX",
        ]
        fallback_names = {t: t for t in fallback_tickers}
        return fallback_tickers, fallback_names


def get_dynamic_leading_stocks(limit=25):
    """실시간 야후 파이낸스 Trending 주도주 동적 수집"""
    print("🔄 [동적 종목 수집] 미국 실시간 주도 테마주 추적 중...")
    dynamic_tickers = {}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/trending/US"
        response = requests.get(url, headers=headers, timeout=8)
        if response.status_code == 200:
            data = response.json()
            trending_list = (
                data.get("finance", {})
                .get("result", [{}])[0]
                .get("quotes", [])
            )
            for item in trending_list:
                ticker = item.get("symbol")
                if ticker and ticker.isalpha() and len(ticker) <= 5:
                    dynamic_tickers[ticker] = f"[Trending] {ticker}"
                    if len(dynamic_tickers) >= limit:
                        break
    except Exception:
        pass

    if len(dynamic_tickers) < 5:
        return {
            "RDDT": "Reddit, Inc.",
            "COIN": "Coinbase Global",
            "HOOD": "Robinhood Markets",
            "PLTR": "Palantir Technologies",
            "LLY": "Eli Lilly",
            "NVO": "Novo Nordisk",
            "VKTX": "Viking Therapeutics",
            "RKLB": "Rocket Lab USA",
            "GEV": "GE Vernova",
            "VST": "Vistra Corp.",
            "CEG": "Constellation Energy",
            "OKLO": "Oklo Inc.",
        }
    return dict(list(dynamic_tickers.items())[:limit])


def detect_vcp_and_pivot(df, lookback=40):
    """VCP 패턴 및 피벗 돌파 연산"""
    df_recent = df.tail(lookback).copy()
    if len(df_recent) < lookback:
        return False, 1.0, 1.0, "관망"

    std_recent = df_recent["Close"].tail(5).std()
    std_past = df_recent["Close"].iloc[:-10].std()
    vcp_ratio = round(std_recent / std_past, 2) if std_past > 0 else 1.0

    vol_recent_shrink = df_recent["Volume"].tail(3).mean()
    vol_past_shrink = df_recent["Volume"].mean()
    vol_shrink_ratio = (
        round(vol_recent_shrink / vol_past_shrink, 2)
        if vol_past_shrink > 0
        else 1.0
    )

    high_20d = df_recent["High"].iloc[-20:-2].max()
    current_close = df_recent["Close"].iloc[-1]

    if vcp_ratio <= 0.65 and vol_shrink_ratio <= 0.70:
        m_point = "1차 타점 (VCP 수렴 완료)"
    elif (
        current_close >= high_20d
        and df_recent["Volume"].iloc[-1] > vol_past_shrink * 1.5
    ):
        m_point = "2차 타점 (피벗 거래량 돌파)"
    else:
        m_point = "조건 미달 (수렴 진행중)"

    return True, vcp_ratio, vol_shrink_ratio, m_point


def calculate_minervini_base(df_hist):
    """미너비니 베이스 스테이지 계산"""
    if len(df_hist) < 200:
        return 1

    df_hist["MA200_slope"] = df_hist["MA200"].diff(5)
    stage2_df = df_hist[df_hist["MA200_slope"] > 0]

    if len(stage2_df) < 20:
        return 1

    base_count = 1
    highest_price = stage2_df["Close"].iloc[0]
    in_correction = False

    for idx, row in stage2_df.iterrows():
        price = row["Close"]
        if price > highest_price:
            highest_price = price
            if in_correction:
                base_count += 1
                in_correction = False
        elif price < highest_price * 0.88:
            in_correction = True

    return min(base_count, 4)


def generate_chart_image(
    ticker, name, df_hist, w_point, m_point, base_stage, rs_rating
):
    """기술적 스크리닝 차트 이미지 생성"""
    df_plot = df_hist.tail(120).copy()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(
        df_plot.index,
        df_plot["Close"],
        label="현재가",
        color="#1e293b",
        linewidth=2,
    )
    ax1.plot(
        df_plot.index,
        df_plot["MA20"],
        label="20일선",
        color="#ef4444",
        linestyle="--",
        alpha=0.6,
    )
    ax1.plot(
        df_plot.index,
        df_plot["MA50"],
        label="50일선",
        color="#3b82f6",
        linestyle="--",
        alpha=0.5,
    )
    ax1.plot(
        df_plot.index,
        df_plot["MA150"],
        label="150일선(와인스타인)",
        color="#10b981",
        linewidth=2,
    )
    ax1.plot(
        df_plot.index,
        df_plot["MA200"],
        label="200일선(미너비니)",
        color="#8b5cf6",
        linewidth=1.5,
        alpha=0.5,
    )

    if len(df_hist) >= 200:
        df_hist["MA200_slope"] = df_hist["MA200"].diff(5)
        stage2_df = df_hist[df_hist["MA200_slope"] > 0]

        if len(stage2_df) >= 20:
            bases_info = []
            current_base = 1
            base_start_idx = stage2_df.index[0]
            highest_price = stage2_df["Close"].iloc[0]
            in_correction = False

            for idx, row in stage2_df.iterrows():
                price_c = row["Close"]
                if price_c > highest_price:
                    if in_correction:
                        bases_info.append(
                            (
                                base_start_idx,
                                idx,
                                highest_price,
                                current_base,
                            )
                        )
                        current_base += 1
                        base_start_idx = idx
                        in_correction = False
                    highest_price = price_c
                elif price_c < highest_price * 0.90:
                    in_correction = True

            bases_info.append(
                (
                    base_start_idx,
                    stage2_df.index[-1],
                    highest_price,
                    current_base,
                )
            )

            for start_dt, end_dt, h_price, b_num in bases_info:
                if (
                    end_dt >= df_plot.index[0]
                    and start_dt <= df_plot.index[-1]
                ):
                    plot_start = max(start_dt, df_plot.index[0])
                    plot_end = min(end_dt, df_plot.index[-1])
                    try:
                        y_start = df_plot.loc[plot_start, "Close"]
                        y_end = df_plot.loc[plot_end, "Close"]
                        y_min = df_plot.loc[plot_start:plot_end, "Close"].min()
                        y_max = df_plot.loc[plot_start:plot_end, "Close"].max()
                    except KeyError:
                        continue

                    x_start = mdates.date2num(plot_start)
                    x_end = mdates.date2num(plot_end)
                    x_mid = (x_start + x_end) / 2
                    width = x_end - x_start

                    if b_num == base_stage:
                        y_control = y_min - (
                            (y_max - y_min) * 0.3
                            if (y_max > y_min)
                            else h_price * 0.05
                        )
                        path_data = [
                            (patches.Path.MOVETO, (x_start, y_start)),
                            (patches.Path.CURVE3, (x_mid, y_control)),
                            (patches.Path.CURVE3, (x_end, y_end)),
                        ]
                        codes, verts = zip(*path_data)
                        path = patches.Path(verts, codes)
                        ax1.add_patch(
                            patches.PathPatch(
                                path,
                                edgecolor="#f59e0b",
                                facecolor="none",
                                lw=2.5,
                                alpha=0.9,
                                zorder=4,
                            )
                        )
                        ax1.text(
                            mdates.num2date(x_mid),
                            y_control,
                            f" 현재 Base {b_num}기 (곡선 수렴) ",
                            color="#d97706",
                            fontsize=9,
                            fontweight="bold",
                            ha="center",
                            va="top",
                        )
                    elif b_num < base_stage:
                        ellipse = patches.Ellipse(
                            xy=(x_mid, (y_max + y_min) / 2),
                            width=width,
                            height=(y_max - y_min) * 1.2,
                            edgecolor="#4338ca",
                            facecolor="#e0e7ff",
                            alpha=0.25,
                            lw=2.0,
                            linestyle="--",
                            zorder=3,
                        )
                        ax1.add_patch(ellipse)
                        ax1.text(
                            mdates.num2date(x_mid),
                            y_min * 0.96,
                            f" 직전 Base {b_num}기 ",
                            color="#4338ca",
                            fontsize=8,
                            fontweight="bold",
                            ha="center",
                            va="top",
                        )

    info_text = f"▶ RS 상대강도: {rs_rating}점\n▶ 미너비니 타점: {m_point} [{base_stage}기]\n▶ 와인스타인 스테이지: {w_point}"
    ax1.text(
        0.02,
        0.92,
        info_text,
        transform=ax1.transAxes,
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#ffffff",
            edgecolor="#cbd5e1",
            alpha=0.9,
        ),
    )
    ax1.set_title(
        f"📈 {name} ({ticker}) 미국 주식 추세 융합 스크리닝 차트",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax1.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        facecolor="white",
        fontsize=9,
    )
    ax1.grid(True, linestyle=":", alpha=0.5)

    colors = [
        "#ef4444" if row["Close"] >= row["Open"] else "#3b82f6"
        for idx, row in df_plot.iterrows()
    ]
    ax2.bar(
        df_plot.index, df_plot["Volume"], color=colors, alpha=0.7, width=0.6
    )
    ax2.grid(True, linestyle=":", alpha=0.5)

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str


def generate_html_report(df_result, stats, today_str, chart_list):
    """통합 HTML 보고서 생성 (상단 면책조항 및 KST 시간 포함)"""
    kst = timezone(timedelta(hours=9))
    now_kst = datetime.now(kst).strftime("%Y년 %m월 %d일 %H시 %M분")

    table_rows = ""
    for idx, row in df_result.iterrows():
        rank = idx + 1
        rating = row["추천등급"]
        badge_style = (
            "bg-danger text-white"
            if rating == "강력매수"
            else (
                "bg-primary text-white"
                if rating == "매수"
                else "bg-warning text-dark"
            )
        )
        vcp_active = (
            "text-success font-bold" if row["변동성축소비율"] <= 0.70 else ""
        )
        vol_active = (
            "text-success font-bold" if row["거래량축소비율"] <= 0.70 else ""
        )

        table_rows += f"""
        <tr>
            <td class="text-center font-bold" style="font-size: 1.1rem; color: #1e293b;">{rank}위</td>
            <td><span class="ticker-badge">{row['종목코드']}</span></td>
            <td><strong>{row['종목명']}</strong></td>
            <td>{row['현재가']}</td>
            <td class="text-center"><span class="badge {badge_style}" style="padding: 6px 12px; border-radius: 20px; font-weight: bold;">{rating}</span></td>
            <td style="font-size: 0.9rem;">
                <strong>와인스타인:</strong> {row['와인스타인지점']}<br>
                <strong>미너비니:</strong> {row['미너비니지점']} <span class="badge bg-secondary">베이스 {row['미너비니베이스']}기</span>
            </td>
            <td class="text-center {vcp_active}">{row['변동성축소비율']}</td>
            <td class="text-center {vol_active}">{row['거래량축소비율']}</td>
            <td class="text-center text-danger"><strong>{row['최근거래량증가(배)']}배</strong></td>
            <td class="text-center font-bold text-primary" style="font-size: 1.1rem; background-color: #f0fdf4;">{row['RS상대강도(백분위)']}점</td>
            <td class="text-center text-dark font-bold" style="font-size: 1.05rem; background-color: #faf5ff;">
                <strong>{row['종합점수']}점</strong>
                <div style="font-size: 0.75rem; color: #6b7280; font-weight: normal; margin-top: 4px;">
                    RS:{row['score_rs']} | VCP:{row['score_vcp']} | Vol:{row['score_vol']} | PV:{row['score_pivot']}
                </div>
            </td>
        </tr>
        """

    chart_sections = ""
    for chart in chart_list:
        chart_sections += f"""
        <div class="row align-items-center border-bottom py-4 bg-white px-3 my-3 rounded-3 shadow-sm" style="display: flex;">
            <div style="flex: 0 0 25%; padding-right: 20px;">
                <h4 class="fw-bold text-dark mb-1" style="margin: 0 0 5px 0; font-size: 1.2rem;">{chart['rank']}위. {chart['name']}</h4>
                <p class="text-muted small mb-3" style="margin: 0 0 15px 0; color: #6c757d; font-size: 0.9rem;">[{chart['ticker']}]</p>
                <div class="p-3 bg-light rounded-3 mb-2" style="background: #f8fafc; padding: 15px; border-radius: 8px; font-size: 0.88rem; border: 1px solid #e2e8f0;">
                    <div style="margin-bottom: 8px;"><strong>추천 등급:</strong> <span class="badge bg-danger" style="background-color:#dc3545; color:white; padding:3px 8px; border-radius:10px;">{chart['rating']}</span></div>
                    <div style="margin-bottom: 8px;"><strong>미너비니 단계:</strong> 베이스 {chart['base_stage']}기 현황</div>
                    <div style="margin-bottom: 8px; color: #2563eb;"><strong>🔥 RS 상대강도:</strong> <strong>{chart['rs_rating']}점</strong></div>
                    <div style="margin-bottom: 8px;"><strong>150일선 이격:</strong> {chart['disparity']}</div>
                    <div class="fw-bold text-primary" style="font-size: 1.05rem; border-top: 1px solid #ddd; padding-top: 5px; margin-top: 5px; font-weight: bold; color: #0d6efd;">종합 스코어: {chart['score']}점 / 100점</div>
                    <div class="mt-2 text-muted" style="font-size: 0.8rem; line-height: 1.4; color:#6c757d; margin-top:8px;">
                        <span style="display:block;">▪ 주도주 RS 점수: {chart['score_rs']}점 / 30</span>
                        <span style="display:block;">▪ VCP 압축 점수: {chart['score_vcp']}점 / 30</span>
                        <span style="display:block;">▪ 거래공백 점수: {chart['score_vol']}점 / 25</span>
                        <span style="display:block;">▪ 피벗매물대 점수: {chart['score_pivot']}점 / 15</span>
                    </div>
                </div>
            </div>
            <div style="flex: 0 0 75%; text-align: center;">
                <img src="data:image/png;base64,{chart['img_base64']}" class="img-fluid rounded border shadow-xs" style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #dee2e6;" alt="차트">
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>미국 주식 미너비니 정통 VCP & 와인스타인 스테이지2 융합 리포트</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ background-color: #f4f6f9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; color: #334155; padding: 20px 0 40px 0; }}
        .card {{ border: none; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); margin-bottom: 30px; }}
        .theory-title {{ border-left: 5px solid #4f46e5; padding-left: 12px; font-weight: 700; }}
        .ticker-badge {{ background-color: #f1f5f9; color: #334155; padding: 6px 10px; border-radius: 6px; font-family: monospace; font-weight: bold; }}
        .stat-card {{ background: linear-gradient(135deg, #4f46e5, #3b82f6); color: white; border-radius: 16px; padding: 25px; text-align: center; }}
        .table th {{ background-color: #f8fafc; color: #64748b; font-weight: 600; text-align: center; }}
        .font-bold {{ font-weight: bold; }}
        .disclaimer-banner {{ background-color: #fef2f2; border: 1px solid #fecaca; color: #991b1b; padding: 12px 20px; border-radius: 12px; font-size: 0.88rem; text-align: center; line-height: 1.5; margin-bottom: 25px; }}
        .update-time {{ background-color: #e2e8f0; color: #475569; display: inline-block; padding: 6px 16px; border-radius: 20px; font-weight: 600; font-size: 0.92rem; }}
    </style>
</head>
<body>
    <div class="container" style="max-width: 1350px;">
        
        <!-- 최상단 면책 조항 (Disclaimer Banner) -->
        <div class="disclaimer-banner">
            ⚠️ <strong>[주의 및 면책 조항]</strong> 해당 내용은 주식 분석 정보 제공 및 정통 기술적 조건 검증을 위한 <strong>참고용</strong> 자료이며, 투자의 최종 책임은 전적으로 본인에게 있습니다.<br>
            <span style="font-size: 0.82rem; opacity: 0.85;">(Disclaimer: The content provided herein is for informational and educational purposes only. All investment decisions are solely the responsibility of the investor.)</span>
        </div>

        <div class="text-center mb-5">
            <h1 class="fw-extrabold" style="color: #0f172a;">📈 정통 추세추종 융합 스크리너 (US Stock Universe)</h1>
            <p class="text-muted fs-5 mb-2">분석 기준일: {today_str[:4]}-{today_str[4:6]}-{today_str[6:]} | S&P500 × NASDAQ100 × US Trending</p>
            <div class="update-time mt-1">⏰ 리포트 자동 산출 일시: {now_kst} (KST)</div>
        </div>

        <div class="row mb-4">
            <div class="col-md-4"><div class="stat-card"><h5>총 스캔 후보군</h5><h2 class="display-5 fw-bold">{stats['total_scanned']}개</h2><p class="mb-0">미국 유니버스 추출 종목</p></div></div>
            <div class="col-md-4"><div class="stat-card" style="background: linear-gradient(135deg, #10b981, #059669);"><h5>융합 추세 필터 통과</h5><h2 class="display-5 fw-bold">{stats['passed']}개</h2><p class="mb-0">중장기 정배열 & RS 만족 종목</p></div></div>
            <div class="col-md-4"><div class="stat-card" style="background: linear-gradient(135deg, #f59e0b, #d97706);"><h5>최종 타점 포착</h5><h2 class="display-5 fw-bold">{len(df_result)}개</h2><p class="mb-0">VCP 압축 또는 피벗 돌파 임박</p></div></div>
        </div>

        <div class="card p-4">
            <h3 class="theory-title mb-4">🔍 대가들의 계량적 조건 만족 주도주 종합 랭킹 리스트</h3>
            <div class="table-responsive">
                <table class="table table-hover align-middle">
                    <thead>
                        <tr>
                            <th>순위</th><th>종목코드</th><th>종목명</th><th>현재가</th><th>추천등급</th>
                            <th>예상 진입 타점</th><th>변동성축소</th><th>거래량축소</th><th>최근 거래량 증가율</th>
                            <th style="background-color: #e8f5e9;">RS 상대강도</th><th>종합점수 (세부항목)</th>
                        </tr>
                    </thead>
                    <tbody>{table_rows}</tbody>
                </table>
            </div>
        </div>

        <div class="card p-4">
            <h3 class="theory-title mb-4">📊 대가들의 기술적 정합성 입체 차트 분석</h3>
            <div class="container-fluid px-0">{chart_sections}</div>
        </div>
    </div>
</body>
</html>
"""
    file_html = f"정통_융합_추세돌파_리포트_US_{today_str}.html"
    with open(file_html, "w", encoding="utf-8-sig") as f:
        f.write(html_content)

    # GitHub Pages 서비스를 위한 index.html 메인 파일 복사 생성
    with open("index.html", "w", encoding="utf-8-sig") as f:
        f.write(html_content)

    return file_html


def get_us_combined_screener(top_n=999):
    print(
        "🚀 [미국 주식] S&P 500 + NASDAQ 100 + 차세대 주도주 [정통 융합 스캔] 가동"
    )

    sp500_tickers, sp500_names = get_sp500_tickers_with_names()
    nasdaq100_tickers, nasdaq100_names = get_nasdaq100_tickers_with_names()
    extra_names = get_dynamic_leading_stocks(limit=25)

    tickers = list(
        set(sp500_tickers + nasdaq100_tickers + list(extra_names.keys()))
    )
    ticker_names = {**sp500_names, **extra_names, **nasdaq100_names}

    total_tickers = len(tickers)
    print(
        f"👉 총 {total_tickers}개 기업 티커 확보 완료. 벌크 다운로드를 진행합니다..."
    )

    today = datetime.today()
    start_date = (today - timedelta(days=600)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    try:
        raw_data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            group_by="ticker",
            progress=True,
        )
    except Exception as e:
        print(f"❌ 데이터 벌크 다운로드 실패: {e}")
        return None

    raw_rs_scores = {}
    history_cache = {}
    valid_targets = []

    print(
        f"\n⚡ 1단계: 미국 유니버스 대상 정통 RS 상대강도 스코어 계산 중..."
    )
    for ticker in tqdm(tickers, desc="RS 연산", file=sys.stdout):
        try:
            if ticker not in raw_data.columns.get_level_values(0):
                continue
            try:
                df = raw_data[ticker].dropna()
            except KeyError:
                df = raw_data.xs(ticker, axis=1, level=0).dropna()

            if len(df) < 200:
                continue

            c_now = float(df["Close"].iloc[-1])
            c_1m = float(df["Close"].iloc[-20])
            c_3m = float(df["Close"].iloc[-60])
            c_6m = float(df["Close"].iloc[-120])
            c_12m = float(df["Close"].iloc[0])

            raw_rs = (
                (((c_now - c_1m) / c_1m) * 4)
                + (((c_now - c_3m) / c_3m) * 2)
                + (((c_now - c_6m) / c_6m) * 2)
                + (((c_now - c_12m) / c_12m) * 2)
            )
            raw_rs_scores[ticker] = raw_rs
            history_cache[ticker] = df
            valid_targets.append(ticker)
        except Exception:
            continue

    if not raw_rs_scores:
        print("❌ 유효한 RS 연산 결과 데이터가 없습니다.")
        return None

    rs_series = pd.Series(raw_rs_scores)
    rs_ratings = (rs_series.rank(pct=True) * 100).round(1).to_dict()

    screener_list = []
    passed_count, failed_count, scanned_count = 0, 0, 0

    print(
        f"\n⚙️ 2단계: 양대 거장의 조건 결합(와인스타인 2Stage ∩ 미너비니 트렌드 템플릿) 검증..."
    )
    for ticker in tqdm(valid_targets, desc="추세 검증", file=sys.stdout):
        scanned_count += 1
        rs_rating = rs_ratings.get(ticker, 0.0)

        if rs_rating < 70.0:
            failed_count += 1
            continue

        try:
            df = history_cache[ticker]

            df["MA20"] = df["Close"].rolling(window=20).mean()
            df["MA50"] = df["Close"].rolling(window=50).mean()
            df["MA150"] = df["Close"].rolling(window=150).mean()
            df["MA200"] = df["Close"].rolling(window=200).mean()

            high_52w = (
                df["High"].rolling(window=250, min_periods=1).max().iloc[-1]
            )
            low_52w = (
                df["Low"].rolling(window=250, min_periods=1).min().iloc[-1]
            )

            last_row = df.iloc[-1]
            current_close = float(last_row["Close"])

            ma20, ma50, ma150, ma200 = (
                last_row["MA20"],
                last_row["MA50"],
                last_row["MA150"],
                last_row["MA200"],
            )
            if pd.isna([current_close, ma20, ma50, ma150, ma200]).any():
                continue

            w_stage2 = (current_close > ma150) and (
                ma150 > df["MA150"].iloc[-20]
            )
            m_template = (
                (current_close > ma50)
                and (ma50 > ma150)
                and (ma150 > ma200)
                and (ma200 > df["MA200"].iloc[-20])
            )
            dist_from_high = ((high_52w - current_close) / high_52w) * 100
            dist_from_low = ((current_close - low_52w) / low_52w) * 100
            trend_safety = (dist_from_high <= 25.0) and (dist_from_low >= 30.0)

            if w_stage2 and m_template and trend_safety:
                passed_count += 1

                success, vcp_ratio, vol_shrink_ratio, m_point = (
                    detect_vcp_and_pivot(df)
                )
                if not success:
                    continue

                disparity_150 = round((current_close / ma150) * 100, 1)
                w_point = (
                    "Stage 2A (돌파 초입 우량)"
                    if disparity_150 <= 112.0
                    else "Stage 2B (추세 확장 국면)"
                )

                box_base_stage = calculate_minervini_base(df)

                score_rs = round((rs_rating / 100) * 30)
                score_vcp = 30 if vcp_ratio <= 0.70 else 10
                score_vol = 25 if vol_shrink_ratio <= 0.70 else 10
                score_pivot = 15 if dist_from_high <= 10.0 else 5

                score = score_rs + score_vcp + score_vol + score_pivot
                rating = (
                    "강력매수"
                    if score >= 80
                    else ("매수" if score >= 55 else "관망/유지")
                )

                screener_list.append(
                    {
                        "종목코드": ticker,
                        "종목명": ticker_names.get(ticker, ticker),
                        "현재가": f"${current_close:,.2f}",
                        "추천등급": rating,
                        "와인스타인지점": w_point,
                        "미너비니지점": m_point,
                        "미너비니베이스": box_base_stage,
                        "변동성축소비율": vcp_ratio,
                        "거래량축소비율": vol_shrink_ratio,
                        "최근거래량증가(배)": round(
                            df["Volume"].iloc[-1] / df["Volume"].iloc[-2], 2
                        ),
                        "150일선이격도": f"{disparity_150}%",
                        "RS상대강도(백분위)": rs_rating,
                        "종합점수": score,
                        "score_rs": score_rs,
                        "score_vcp": score_vcp,
                        "score_vol": score_vol,
                        "score_pivot": score_pivot,
                        "Base_Raw": box_base_stage,
                    }
                )
                tqdm.write(
                    f" 🎯 [타점포착] {ticker:<5} | RS: {rs_rating}점 | VCP비율: {vcp_ratio} | 스코어: {score}점"
                )
            else:
                failed_count += 1
        except Exception:
            failed_count += 1
            continue

    stats = {
        "total_scanned": scanned_count,
        "passed": passed_count,
        "failed": failed_count,
    }
    if not screener_list:
        print(
            "\n❌ 양대 대가의 정통 조건식을 완벽히 만족하는 종목이 오늘 미국 시장에 없습니다."
        )
        return None

    df_result = pd.DataFrame(screener_list)
    df_result = (
        df_result.sort_values(
            by=["종합점수", "RS상대강도(백분위)"], ascending=[False, False]
        )
        .head(top_n)
        .reset_index(drop=True)
    )
    df_result_for_charts = df_result.copy()

    df_result.index = df_result.index + 1
    df_result.index.name = "순위"

    today_str = datetime.today().strftime("%Y%m%d")
    df_result.to_csv(
        f"정통_융합_추세돌파_스크리닝_US_{today_str}.csv",
        index=True,
        encoding="utf-8-sig",
    )

    print(
        f"\n📊 조건 만족 종목 ({len(df_result_for_charts)}개) 기술적 연동 차트 빌드 중..."
    )
    chart_list = []

    for idx, row in df_result_for_charts.iterrows():
        ticker = row["종목코드"]
        df_hist_target = history_cache.get(ticker)
        if df_hist_target is not None:
            img_base64 = generate_chart_image(
                ticker=ticker,
                name=row["종목명"],
                df_hist=df_hist_target,
                w_point=row["와인스타인지점"],
                m_point=row["미너비니지점"],
                base_stage=row["Base_Raw"],
                rs_rating=row["RS상대강도(백분위)"],
            )

            chart_list.append(
                {
                    "rank": idx + 1,
                    "ticker": ticker,
                    "name": row["종목명"],
                    "rating": row["추천등급"],
                    "base_stage": row["Base_Raw"],
                    "disparity": row["150일선이격도"],
                    "rs_rating": row["RS상대강도(백분위)"],
                    "score": row["종합점수"],
                    "img_base64": img_base64,
                    "score_rs": row["score_rs"],
                    "score_vcp": row["score_vcp"],
                    "score_vol": row["score_vol"],
                    "score_pivot": row["score_pivot"],
                }
            )

    file_html = generate_html_report(
        df_result.reset_index(drop=True), stats, today_str, chart_list
    )
    print(
        f"\n🎉 [엔진 종료] 미국 주식 스크리닝 출력 완료!\n🌐 HTML 종합 보고서: {file_html}"
    )
    return df_result


if __name__ == "__main__":
    result = get_us_combined_screener(top_n=999)
