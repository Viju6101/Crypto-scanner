"""
Crypto futures scanner -> Telegram alerts.
Signals only. No order execution. Not financial advice.

Strategy (simple, transparent, meant to be backtested and tuned):
  - 1h trend filter: EMA50 vs EMA200
  - 15m trigger: RSI pullback recovery (long) / rejection (short) + volume spike
  - SL = 1.5 x ATR, TP = 3 x ATR (2R)
"""
import os
import time
import sqlite3
import logging

import ccxt
import pandas as pd
import requests

# ---------- Config (set via environment variables) ----------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EXCHANGE = os.getenv("EXCHANGE", "bybit")      # bybit, okx, binanceusdm, ...
TOP_N = int(os.getenv("TOP_N", "40"))          # scan top N pairs by 24h volume
SCAN_EVERY_SEC = int(os.getenv("SCAN_EVERY_SEC", "300"))
COOLDOWN_HOURS = float(os.getenv("COOLDOWN_HOURS", "4"))
VOL_SPIKE = float(os.getenv("VOL_SPIKE", "1.3"))
SL_ATR, TP_ATR = 1.5, 3.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scanner")

ex = getattr(ccxt, EXCHANGE)({"enableRateLimit": True, "options": {"defaultType": "swap"}})

# ---------- Database ----------
db = sqlite3.connect("signals.db")
db.execute(
    """CREATE TABLE IF NOT EXISTS signals (
        ts REAL, symbol TEXT, side TEXT, entry REAL, sl REAL, tp REAL,
        rsi REAL, vol_ratio REAL)"""
)
db.commit()


def recently_signaled(symbol, side):
    cutoff = time.time() - COOLDOWN_HOURS * 3600
    row = db.execute(
        "SELECT 1 FROM signals WHERE symbol=? AND side=? AND ts>? LIMIT 1",
        (symbol, side, cutoff),
    ).fetchone()
    return row is not None


def log_signal(s):
    db.execute(
        "INSERT INTO signals VALUES (?,?,?,?,?,?,?,?)",
        (time.time(), s["symbol"], s["side"], s["entry"], s["sl"], s["tp"], s["rsi"], s["vol_ratio"]),
    )
    db.commit()


# ---------- Indicators ----------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ---------- Data ----------
def get_df(symbol, timeframe, limit=250):
    raw = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    return df.iloc[:-1].reset_index(drop=True)  # drop still-forming candle


def top_symbols():
    ex.load_markets()
    tickers = ex.fetch_tickers()
    rows = []
    for sym, t in tickers.items():
        m = ex.markets.get(sym)
        if not m or not m.get("swap") or not m.get("linear") or m.get("quote") != "USDT":
            continue
        if not m.get("active", True):
            continue
        rows.append((sym, t.get("quoteVolume") or 0))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in rows[:TOP_N]]


# ---------- Signal logic ----------
def analyze(symbol):
    h1 = get_df(symbol, "1h")
    if len(h1) < 210:
        return None
    e50, e200 = ema(h1["close"], 50).iloc[-1], ema(h1["close"], 200).iloc[-1]
    price1h = h1["close"].iloc[-1]
    if e50 > e200 and price1h > e50:
        trend = "up"
    elif e50 < e200 and price1h < e50:
        trend = "down"
    else:
        return None

    m15 = get_df(symbol, "15m")
    r = rsi(m15["close"])
    prev_rsi, cur_rsi = r.iloc[-2], r.iloc[-1]
    vol_ratio = m15["volume"].iloc[-1] / m15["volume"].rolling(20).mean().iloc[-1]
    a = atr(m15).iloc[-1]
    entry = m15["close"].iloc[-1]

    side = None
    if trend == "up" and prev_rsi < 40 <= cur_rsi and vol_ratio >= VOL_SPIKE:
        side = "LONG"
    elif trend == "down" and prev_rsi > 60 >= cur_rsi and vol_ratio >= VOL_SPIKE:
        side = "SHORT"
    if not side:
        return None

    sign = 1 if side == "LONG" else -1
    return {
        "symbol": symbol,
        "side": side,
        "entry": entry,
        "sl": entry - sign * SL_ATR * a,
        "tp": entry + sign * TP_ATR * a,
        "rsi": float(cur_rsi),
        "vol_ratio": float(vol_ratio),
        "trend": trend,
    }


# ---------- Telegram ----------
def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram not configured. Message:\n%s", text)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=15,
        )
    except requests.RequestException as e:
        log.error("Telegram error: %s", e)


def fmt(s):
    icon = "🟢" if s["side"] == "LONG" else "🔴"
    return (
        f"{icon} <b>{s['side']}</b> {s['symbol']}\n"
        f"Entry: <code>{s['entry']:.6g}</code>\n"
        f"Stop:  <code>{s['sl']:.6g}</code>\n"
        f"Target: <code>{s['tp']:.6g}</code> (2R)\n"
        f"1h trend: {s['trend']} | RSI(15m): {s['rsi']:.1f} | Vol: {s['vol_ratio']:.1f}x\n"
        f"<i>Signal only, not financial advice.</i>"
    )


# ---------- Main loop ----------
def scan_once():
    symbols = top_symbols()
    log.info("Scanning %d symbols on %s", len(symbols), EXCHANGE)
    for sym in symbols:
        try:
            s = analyze(sym)
            if s and not recently_signaled(sym, s["side"]):
                log_signal(s)
                send_telegram(fmt(s))
                log.info("Signal: %s %s", s["side"], sym)
        except Exception as e:
            log.warning("%s failed: %s", sym, e)


if __name__ == "__main__":
    if os.getenv("ONCE") == "1":  # run a single scan and exit (for GitHub Actions)
        scan_once()
        if os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch":  # manual test run
            send_telegram(f"✅ Manual scan complete on {EXCHANGE}. Alerts will appear here when setups trigger.")
        raise SystemExit(0)
    send_telegram(f"✅ Scanner started on {EXCHANGE} (top {TOP_N} pairs, every {SCAN_EVERY_SEC}s)")
    while True:
        try:
            scan_once()
        except Exception as e:
            log.error("Scan failed: %s", e)
        time.sleep(SCAN_EVERY_SEC)
