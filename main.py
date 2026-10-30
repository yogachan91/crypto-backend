"""
Crypto Statistical Trend Forecast API v1.9
+ Natural Forecast Path (curved / non-straight future line)
+ BUY/SELL markers
+ Multi-interval (1m → 1d)
+ Multi-TF + Order Book + BIAS + Strength + Confidence

All previous features kept. Demo only — NOT financial advice.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
import httpx
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime, timezone, timedelta
import os
from dotenv import load_dotenv

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Float, Integer, DateTime

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/crypto_forecast"
)

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class ForecastLog(Base):
    __tablename__ = "forecast_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(50))
    interval: Mapped[str] = mapped_column(String(10))
    slope: Mapped[float] = mapped_column(Float)
    target_price: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    bias: Mapped[str] = mapped_column(String(20), default="SIDEWAYS")
    strength: Mapped[str] = mapped_column(String(20), default="Weak")
    recommendation: Mapped[str] = mapped_column(String(10), default="WAIT")
    ob_imbalance: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


app = FastAPI(
    title="Crypto Statistical Forecast API v1.9",
    description="Natural forecast path + BUY/SELL markers + Multi-interval",
    version="1.9.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYMBOLS = {
    "BTCUSDT": "BTC/USDT Perpetual",
    "1000SHIBUSDT": "1000SHIB/USDT Perpetual",
    "1000PEPEUSDT": "1000PEPE/USDT Perpetual",
}

SUPPORTED_INTERVALS = ["1m", "3m", "5m", "15m", "1h", "6h", "1d"]

HIGHER_TFS = {
    "1m":  ["5m", "15m"],
    "3m":  ["15m", "1h"],
    "5m":  ["15m", "1h"],
    "15m": ["1h", "4h"],
    "1h":  ["4h", "1d"],
    "6h":  ["1d", "3d"],
    "1d":  ["3d", "1w"],
}

INTERVAL_DELTA = {
    "1m":  timedelta(minutes=1),
    "3m":  timedelta(minutes=3),
    "5m":  timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h":  timedelta(hours=1),
    "4h":  timedelta(hours=4),
    "6h":  timedelta(hours=6),
    "1d":  timedelta(days=1),
    "3d":  timedelta(days=3),
    "1w":  timedelta(weeks=1),
}


class Candle(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class ForecastPoint(BaseModel):
    time: int
    value: float
    upper: Optional[float] = None
    lower: Optional[float] = None


class Metrics(BaseModel):
    lookback_length: int
    forecast_horizon: int
    trend_slope: float
    current_bag_price: float
    target_price_end: float
    visual_range_pct: float
    alert_trigger_z: float
    r_squared: float
    multi_tf_alignment: str
    confidence_score: float
    bias: str
    strength: str
    recommendation: str          # strict panel: BUY/SELL/WAIT
    chart_marker: str            # lenient chart marker: BUY/SELL/WAIT
    orderbook_imbalance: float
    bid_ask_ratio: float
    ob_signal: str
    largest_bid_wall: float
    largest_ask_wall: float


class ForecastResponse(BaseModel):
    symbol: str
    interval: str
    candles: List[Candle]
    forecast_line: List[ForecastPoint]       # linear mean path
    path_line: List[ForecastPoint]           # natural curved path (NEW)
    upper_band: List[ForecastPoint]
    lower_band: List[ForecastPoint]
    metrics: Metrics
    generated_at: str


@app.on_event("startup")
async def startup():
    try:
        await init_db()
        print("✅ PostgreSQL connected & tables ready (v1.9)")
    except Exception as e:
        print(f"⚠️  Database connection failed: {e}")
        print("   App still runs, logging disabled.")


async def fetch_binance_klines(symbol: str, interval: str = "1m", limit: int = 200) -> pd.DataFrame:
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, params=params)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Binance klines error ({interval}): {resp.text}")
        data = resp.json()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


async def fetch_order_book(symbol: str, limit: int = 20) -> dict:
    url = "https://fapi.binance.com/fapi/v1/depth"
    params = {"symbol": symbol, "limit": limit}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, params=params)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Binance depth error: {resp.text}")
        return resp.json()


def analyze_order_book(depth: dict) -> dict:
    bids = depth.get("bids", [])
    asks = depth.get("asks", [])

    if not bids or not asks:
        return {
            "imbalance": 0.0,
            "bid_ask_ratio": 1.0,
            "ob_signal": "Neutral",
            "largest_bid_wall": 0.0,
            "largest_ask_wall": 0.0,
        }

    bid_volumes = [float(q) for _, q in bids]
    ask_volumes = [float(q) for _, q in asks]

    total_bid = sum(bid_volumes)
    total_ask = sum(ask_volumes)
    total = total_bid + total_ask + 1e-12

    imbalance = (total_bid - total_ask) / total
    bid_ask_ratio = total_bid / (total_ask + 1e-12)

    largest_bid = max(bid_volumes) if bid_volumes else 0.0
    largest_ask = max(ask_volumes) if ask_volumes else 0.0

    if imbalance > 0.18 and bid_ask_ratio > 1.25:
        ob_signal = "Buy Pressure"
    elif imbalance < -0.18 and bid_ask_ratio < 0.80:
        ob_signal = "Sell Pressure"
    else:
        ob_signal = "Neutral"

    return {
        "imbalance": round(imbalance, 4),
        "bid_ask_ratio": round(bid_ask_ratio, 3),
        "ob_signal": ob_signal,
        "largest_bid_wall": round(largest_bid, 4),
        "largest_ask_wall": round(largest_ask, 4),
    }


def fit_regression(closes: np.ndarray, lookback: int):
    y = closes[-lookback:]
    X = np.arange(lookback).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    slope = float(model.coef_[0])
    r2 = float(r2_score(y, preds))
    resid = y - preds
    std = float(np.std(resid))
    if std < 1e-12:
        std = abs(closes[-1]) * 0.001
    return slope, r2, preds, std


def generate_natural_path(
    last_price: float,
    linear_targets: np.ndarray,
    recent_closes: np.ndarray,
    std: float,
    horizon: int
) -> np.ndarray:
    """
    Create a more natural-looking path toward the linear forecast.
    - Starts with short-term momentum
    - Gradually blends into the linear regression target
    - Adds mild smooth oscillation so it doesn't look like a ruler
    This is for visualization; it does NOT increase real accuracy.
    """
    if horizon <= 0:
        return np.array([])

    # Recent momentum (last few bars)
    k = min(5, len(recent_closes) - 1)
    if k >= 1:
        momentum = (recent_closes[-1] - recent_closes[-1 - k]) / k
    else:
        momentum = 0.0

    path = np.zeros(horizon)
    price = last_price

    for i in range(horizon):
        t = (i + 1) / horizon  # 0 → 1 progress

        # Blend: early steps follow momentum more, later steps follow linear target
        linear_val = float(linear_targets[i])
        momentum_step = price + momentum * (1 - t) * 0.85

        # Smooth blend weight
        w = t ** 1.2  # more weight to linear as we go further
        blended = (1 - w) * momentum_step + w * linear_val

        # Mild smooth wave (visual only) — amplitude decays over time
        wave_amp = std * 0.35 * (1 - t) ** 1.5
        wave = wave_amp * np.sin(i * 0.55 + 0.3)

        price = blended + wave
        path[i] = price

        # Softly pull momentum toward residual to linear
        residual = (linear_val - price) * 0.08
        momentum = momentum * 0.92 + residual

    return path


def compute_statistical_forecast(
    df: pd.DataFrame,
    interval: str,
    lookback: int = 30,
    horizon: int = 60,
    confidence_band: float = 1.5
) -> dict:
    closes = df["close"].values
    times = df["open_time"].values

    if len(closes) < lookback + 5:
        raise ValueError("Not enough data")

    slope, r2, hist_preds, std = fit_regression(closes, lookback)

    y = closes[-lookback:]
    X = np.arange(lookback).reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    future_X = np.arange(lookback, lookback + horizon).reshape(-1, 1)
    future_preds = model.predict(future_X)

    last_ts = pd.Timestamp(times[-1])
    delta = INTERVAL_DELTA.get(interval, timedelta(minutes=1))

    # Natural path
    natural = generate_natural_path(
        last_price=float(closes[-1]),
        linear_targets=future_preds,
        recent_closes=closes,
        std=std,
        horizon=horizon
    )

    forecast_points = []
    path_points = []
    upper_points = []
    lower_points = []

    for i in range(horizon):
        t = last_ts + delta * (i + 1)
        ts = int(t.timestamp())
        lin_val = float(future_preds[i])
        path_val = float(natural[i])
        band = std * confidence_band * (1 + i * 0.015)

        forecast_points.append({"time": ts, "value": lin_val, "upper": lin_val + band, "lower": lin_val - band})
        path_points.append({"time": ts, "value": path_val})
        upper_points.append({"time": ts, "value": lin_val + band})
        lower_points.append({"time": ts, "value": lin_val - band})

    hist_upper = []
    hist_lower = []
    hist_times = times[-lookback:]
    for i, t in enumerate(hist_times):
        val = float(hist_preds[i])
        band = std * confidence_band
        ts = int(pd.Timestamp(t).timestamp())
        hist_upper.append({"time": ts, "value": val + band})
        hist_lower.append({"time": ts, "value": val - band})

    current_price = float(closes[-1])
    target_price = float(future_preds[-1])
    visual_range_pct = abs(target_price - current_price) / current_price * 100
    alert_z = (target_price - current_price) / (std + 1e-9)

    return {
        "forecast_line": forecast_points,
        "path_line": path_points,
        "upper_band": hist_upper + upper_points,
        "lower_band": hist_lower + lower_points,
        "slope": slope,
        "r2": r2,
        "std": std,
        "current_price": current_price,
        "target_price": target_price,
        "visual_range_pct": visual_range_pct,
        "alert_z": alert_z,
    }


def compute_confidence(
    r2: float,
    abs_slope_norm: float,
    alignment_score: float,
    visual_range_pct: float,
    ob_imbalance: float,
    ob_signal: str
) -> float:
    r2_s = max(0.0, min(1.0, r2))
    slope_s = max(0.0, min(1.0, abs_slope_norm))
    align_s = max(0.0, min(1.0, alignment_score))
    move_s = 1.0 if visual_range_pct >= 0.15 else max(0.0, visual_range_pct / 0.15)

    ob_s = min(1.0, abs(ob_imbalance) * 2.2)
    if ob_signal == "Neutral":
        ob_s *= 0.5

    score = (
        r2_s * 0.28 +
        slope_s * 0.20 +
        align_s * 0.22 +
        move_s * 0.08 +
        ob_s * 0.22
    ) * 100
    return round(float(score), 1)


def decide_bias_and_strength(
    slope: float,
    r2: float,
    alignment: str,
    ob_signal: str,
    ob_imbalance: float,
    confidence: float
) -> Tuple[str, str]:
    slope_up = slope > 0
    slope_down = slope < 0

    ob_bullish = ob_signal == "Buy Pressure" or ob_imbalance > 0.12
    ob_bearish = ob_signal == "Sell Pressure" or ob_imbalance < -0.12

    tf_ok = alignment in ("Aligned", "Mostly Aligned")

    if slope_up and (ob_bullish or not ob_bearish) and tf_ok:
        bias = "LONG"
    elif slope_down and (ob_bearish or not ob_bullish) and tf_ok:
        bias = "SHORT"
    elif slope_up and ob_bullish:
        bias = "LONG"
    elif slope_down and ob_bearish:
        bias = "SHORT"
    else:
        bias = "SIDEWAYS"

    if (slope_up and ob_bearish) or (slope_down and ob_bullish):
        bias = "SIDEWAYS"

    if bias == "SIDEWAYS":
        strength = "No Trade" if confidence < 40 else "Weak"
    else:
        if confidence >= 70 and r2 >= 0.50 and alignment == "Aligned":
            strength = "Strong"
        elif confidence >= 55:
            strength = "Moderate"
        elif confidence >= 40:
            strength = "Weak"
        else:
            strength = "No Trade"

    return bias, strength


def decide_recommendation(bias: str, strength: str, confidence: float) -> str:
    """
    Strict recommendation for the panel (more conservative).
    """
    if bias == "LONG" and strength == "Strong" and confidence >= 70:
        return "BUY"
    if bias == "SHORT" and strength == "Strong" and confidence >= 70:
        return "SELL"
    return "WAIT"


def decide_chart_marker(
    bias: str,
    strength: str,
    confidence: float,
    recent_momentum: float,
    ob_signal: str,
    ob_imbalance: float,
    visual_range_pct: float,
    current_price: float,
) -> str:
    """
    Chart marker focused on short-term scalp opportunity.
    More reactive than panel recommendation.

    Goals from user feedback:
    - If short-term down + sell pressure and move is not tiny -> SELL
    - If short-term up + buy pressure and move is not tiny -> BUY
    - Only WAIT when move is very small or signals are too mixed
    """
    # Ignore tiny expected/observed movement
    min_move_pct = 0.08  # below this -> too small, no marker
    if visual_range_pct < min_move_pct and abs(recent_momentum) / max(current_price, 1e-12) * 100 < min_move_pct:
        return "WAIT"

    mom_up = recent_momentum > 0
    mom_down = recent_momentum < 0

    ob_buy = ob_signal == "Buy Pressure" or ob_imbalance > 0.10
    ob_sell = ob_signal == "Sell Pressure" or ob_imbalance < -0.10

    # Strong short-term agreement
    if mom_down and ob_sell:
        return "SELL"
    if mom_up and ob_buy:
        return "BUY"

    # Medium agreement with existing bias/strength
    if mom_down and bias == "SHORT" and strength in ("Strong", "Moderate", "Weak") and confidence >= 45:
        return "SELL"
    if mom_up and bias == "LONG" and strength in ("Strong", "Moderate", "Weak") and confidence >= 45:
        return "BUY"

    # Path/bias only, if confidence not too low and move not tiny
    if bias == "SHORT" and strength in ("Strong", "Moderate") and confidence >= 50:
        return "SELL"
    if bias == "LONG" and strength in ("Strong", "Moderate") and confidence >= 50:
        return "BUY"

    return "WAIT"


async def get_multi_tf_slopes(symbol: str, primary_interval: str, lookback: int = 30) -> Dict[str, float]:
    higher = HIGHER_TFS.get(primary_interval, ["15m", "1h"])
    slopes = {}
    for tf in higher:
        try:
            lim = min(lookback + 30, 200)
            df = await fetch_binance_klines(symbol, interval=tf, limit=lim)
            closes = df["close"].values
            lb = min(lookback, len(closes) - 2)
            if lb < 5:
                slopes[tf] = 0.0
                continue
            slope, _, _, _ = fit_regression(closes, lb)
            slopes[tf] = slope
        except Exception as e:
            print(f"Higher TF {tf} failed: {e}")
            slopes[tf] = 0.0
    return slopes


def evaluate_alignment(primary_slope: float, tf_slopes: Dict[str, float]) -> Tuple[str, float]:
    signs = []
    if primary_slope != 0:
        signs.append(1 if primary_slope > 0 else -1)
    for s in tf_slopes.values():
        if s != 0:
            signs.append(1 if s > 0 else -1)

    if not signs:
        return "Unknown", 0.0

    if len(set(signs)) == 1:
        return "Aligned", 1.0

    pos = sum(1 for x in signs if x > 0)
    neg = len(signs) - pos
    ratio = max(pos, neg) / len(signs)
    if ratio >= 0.66:
        return "Mostly Aligned", 0.65
    return "Mixed", 0.25


@app.get("/")
async def root():
    return {
        "message": "Crypto Statistical Forecast API v1.9 — Natural path + BUY/SELL markers",
        "docs": "/docs",
        "symbols": list(SYMBOLS.keys()),
        "intervals": SUPPORTED_INTERVALS,
        "version": "1.9.0"
    }


@app.get("/symbols")
async def get_symbols():
    return [{"symbol": k, "name": v} for k, v in SYMBOLS.items()]


@app.get("/intervals")
async def get_intervals():
    return SUPPORTED_INTERVALS


@app.get("/forecast/{symbol}", response_model=ForecastResponse)
async def get_forecast(
    symbol: str,
    interval: str = Query("1m", description="1m, 3m, 5m, 15m, 1h, 6h, 1d"),
    lookback: int = Query(30, ge=10, le=100),
    horizon: int = Query(24, ge=5, le=120),
    limit: int = Query(200, ge=50, le=500)
):
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        raise HTTPException(400, detail=f"Supported: {list(SYMBOLS.keys())}")
    if interval not in SUPPORTED_INTERVALS:
        raise HTTPException(400, detail=f"Interval must be one of: {SUPPORTED_INTERVALS}")

    if interval in ["6h", "1d"] and limit > 200:
        limit = 200

    try:
        df = await fetch_binance_klines(symbol, interval=interval, limit=limit)
    except Exception as e:
        raise HTTPException(502, detail=str(e))

    try:
        result = compute_statistical_forecast(
            df, interval=interval, lookback=lookback, horizon=horizon
        )
    except Exception as e:
        raise HTTPException(500, detail=f"Forecast error: {e}")

    tf_slopes = await get_multi_tf_slopes(symbol, primary_interval=interval, lookback=lookback)
    alignment_label, alignment_score = evaluate_alignment(result["slope"], tf_slopes)

    try:
        depth = await fetch_order_book(symbol, limit=20)
        ob = analyze_order_book(depth)
    except Exception as e:
        print(f"Order book fetch failed: {e}")
        ob = {
            "imbalance": 0.0,
            "bid_ask_ratio": 1.0,
            "ob_signal": "Neutral",
            "largest_bid_wall": 0.0,
            "largest_ask_wall": 0.0,
        }

    price = result["current_price"]
    abs_slope_norm = min(1.0, abs(result["slope"]) / (price * 0.0008 + 1e-12))

    confidence = compute_confidence(
        r2=result["r2"],
        abs_slope_norm=abs_slope_norm,
        alignment_score=alignment_score,
        visual_range_pct=result["visual_range_pct"],
        ob_imbalance=ob["imbalance"],
        ob_signal=ob["ob_signal"]
    )

    bias, strength = decide_bias_and_strength(
        slope=result["slope"],
        r2=result["r2"],
        alignment=alignment_label,
        ob_signal=ob["ob_signal"],
        ob_imbalance=ob["imbalance"],
        confidence=confidence
    )

    recommendation = decide_recommendation(bias, strength, confidence)

    # Recent short-term momentum from last few closes
    closes_arr = df["close"].values
    k = min(4, len(closes_arr) - 1)
    recent_momentum = float(closes_arr[-1] - closes_arr[-1 - k]) if k >= 1 else 0.0

    chart_marker = decide_chart_marker(
        bias=bias,
        strength=strength,
        confidence=confidence,
        recent_momentum=recent_momentum,
        ob_signal=ob["ob_signal"],
        ob_imbalance=ob["imbalance"],
        visual_range_pct=result["visual_range_pct"],
        current_price=result["current_price"],
    )

    candles = [
        Candle(
            time=int(pd.Timestamp(row["open_time"]).timestamp()),
            open=round(row["open"], 8),
            high=round(row["high"], 8),
            low=round(row["low"], 8),
            close=round(row["close"], 8),
            volume=round(row["volume"], 2)
        )
        for _, row in df.iterrows()
    ]

    metrics = Metrics(
        lookback_length=lookback,
        forecast_horizon=horizon,
        trend_slope=round(result["slope"], 8),
        current_bag_price=round(result["current_price"], 6 if price < 1 else 2),
        target_price_end=round(result["target_price"], 6 if price < 1 else 2),
        visual_range_pct=round(result["visual_range_pct"], 3),
        alert_trigger_z=round(result["alert_z"], 2),
        r_squared=round(result["r2"], 3),
        multi_tf_alignment=alignment_label,
        confidence_score=confidence,
        bias=bias,
        strength=strength,
        recommendation=recommendation,
        chart_marker=chart_marker,
        orderbook_imbalance=ob["imbalance"],
        bid_ask_ratio=ob["bid_ask_ratio"],
        ob_signal=ob["ob_signal"],
        largest_bid_wall=ob["largest_bid_wall"],
        largest_ask_wall=ob["largest_ask_wall"],
    )

    try:
        async with AsyncSessionLocal() as session:
            log = ForecastLog(
                symbol=symbol,
                interval=interval,
                slope=result["slope"],
                target_price=result["target_price"],
                confidence=confidence,
                bias=bias,
                strength=strength,
                recommendation=recommendation,
                ob_imbalance=ob["imbalance"],
                created_at=datetime.now(timezone.utc)
            )
            session.add(log)
            await session.commit()
    except Exception as e:
        print(f"DB log failed: {e}")

    return ForecastResponse(
        symbol=symbol,
        interval=interval,
        candles=candles,
        forecast_line=[ForecastPoint(**p) for p in result["forecast_line"]],
        path_line=[ForecastPoint(time=p["time"], value=p["value"]) for p in result["path_line"]],
        upper_band=[ForecastPoint(time=p["time"], value=p["value"]) for p in result["upper_band"]],
        lower_band=[ForecastPoint(time=p["time"], value=p["value"]) for p in result["lower_band"]],
        metrics=metrics,
        generated_at=datetime.now(timezone.utc).isoformat()
    )


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.9.0", "time": datetime.now(timezone.utc).isoformat()}
