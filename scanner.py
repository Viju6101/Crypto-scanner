"""
Crypto futures scanner -> Telegram alerts + automatic paper trading.
Signals only. No real orders are ever placed. Not financial advice.

Strategy (simple, transparent, meant to be tuned):
  - 1h trend filter: EMA50 vs EMA200
  - 15m trigger: RSI pullback recovery (long) / rejection (short) + volume spike
  - SL = 1.5 x ATR, TP = 3 x ATR (2R)

Paper trading: every signal is saved as an "open" trade. On each run the scanner
checks later candles to see whether the stop or target was hit first, records the
result in R multiples (after estimated fees), and sends a daily stats summary.
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
TOP_N = int(os.getenv("TOP_N", "300"))         # scan top N pairs by 24h volume
SCAN_EVERY_SEC = int(os.getenv("SCAN_EVERY_SEC", "300"))
COOLDOWN_HOURS = float(os.getenv("COOLDOWN_HOURS", "4"))
VOL_SPIKE = float(os.getenv("VOL_SPIKE", "1.1"))
RSI_LO = float(os.getenv("RSI_LO", "45"))      # long trigger: RSI crosses up through this
RSI_HI = float(os.getenv("RSI_HI", "55"))      # short trigger: RSI crosses down through this
SL_ATR, TP_ATR = 1.5, 3.0
EXPIRY_HOURS = float(os.getenv("EXPIRY_HOURS", "24"))   # close paper trade at market after this
FEE_PCT = float(os.getenv("FEE_PCT", "0.001"))          # est. round-trip fees+slippage (0.1%)
TF_MS = 15 * 60 * 1000

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
db.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
# Upgrade older databases (restored from cache) with the paper-trading columns
_cols = {r[1] for r in db.execute("PRAGMA table_info(signals)")}
for _name, _ddl in [
    ("bar_close_ms", "REAL"),
    ("status", "TEXT DEFAULT 'open'"),
    ("closed_ts", "REAL"),
    ("r_result", "REAL"),
]:
    if _name not in _cols:
        db.execute(f"ALTER TABLE signals ADD COLUMN {_name} {_ddl}")
db.commit()


def recently_signaled(symbol, side):
    cutoff = time.time() - COOLDOWN_HOURS * 3600
    row = db.execute(
        "SELECT 1 FROM signals WHERE symbol=? AND side=? AND (ts>? OR status='open') LIMIT 1",
        (symbol, side, cutoff),
    ).fetchone()
    return row is not None


def log_signal(s):
    db.execute(
        "INSERT INTO signals (ts,symbol,side,entry,sl,tp,rsi,vol_ratio,bar_close_ms,status) "
        "VALUES (?,?,?,?,?,?,?,?,?,'open')",
        (time.time(), s["symbol"], s["side"], s["entry"], s["sl"], s["tp"],
         s["rsi"], s["vol_ratio"], s["bar_close_ms"]),
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
    if trend == "up" and prev_rsi < RSI_LO <= cur_rsi and vol_ratio >= VOL_SPIKE:
        side = "LONG"
    elif trend == "down" and prev_rsi > RSI_HI >= cur_rsi and vol_ratio >= VOL_SPIKE:
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
        "bar_close_ms": int(m15["ts"].iloc[-1]) + TF_MS,  # when the signal candle closed
    }


# ---------- Paper trading ----------
def update_open_trades():
    """Check later candles for each open paper trade: stop hit, target hit, or expired."""
    rows = db.execute(
        "SELECT rowid, ts, symbol, side, entry, sl, tp, bar_close_ms FROM signals WHERE status='open'"
    ).fetchall()
    now = time.time()
    for rid, ts, sym, side, entry, sl, tp, bar_close in rows:
        try:
            start = bar_close if bar_close else ts * 1000
            df = get_df(sym, "15m", limit=150)
            df = df[df["ts"] >= start]
            risk = abs(entry - sl)
            cost = entry * FEE_PCT / risk       # fees/slippage expressed in R
            sign = 1 if side == "LONG" else -1
            status, res = None, None
            for _, c in df.iterrows():
                if side == "LONG":
                    hit_sl, hit_tp = c["low"] <= sl, c["high"] >= tp
                else:
                    hit_sl, hit_tp = c["high"] >= sl, c["low"] <= tp
                if hit_sl:                       # if both hit in one candle, assume stop first
                    status, res = "sl", -1.0 - cost
                    break
                if hit_tp:
                    status, res = "tp", TP_ATR / SL_ATR - cost
                    break
            if status is None and now - ts >= EXPIRY_HOURS * 3600:
                if df.empty:
                    status, res = "unknown", None   # candle history no longer available
                else:
                    last = float(df["close"].iloc[-1])
                    status, res = "expired", sign * (last - entry) / risk - cost
            if status:
                db.execute(
                    "UPDATE signals SET status=?, closed_ts=?, r_result=? WHERE rowid=?",
                    (status, now, res, rid),
                )
                db.commit()
                log.info("Paper trade closed: %s %s -> %s", side, sym, status)
        except Exception as e:
            log.warning("update %s failed: %s", sym, e)


def stats_text():
    rows = db.execute(
        "SELECT side, status, r_result FROM signals WHERE status IN ('tp','sl','expired') AND r_result IS NOT NULL"
    ).fetchall()
    open_n = db.execute("SELECT COUNT(*) FROM signals WHERE status='open'").fetchone()[0]
    if not rows:
        return f"📊 Paper-trade stats: no closed trades yet ({open_n} open)."

    def block(rs):
        n = len(rs)
        wins = sum(1 for r in rs if r[2] > 0)
        tot = sum(r[2] for r in rs)
        return n, wins, tot

    n, wins, tot = block(rows)
    tp = sum(1 for r in rows if r[1] == "tp")
    sl = sum(1 for r in rows if r[1] == "sl")
    ex_n = sum(1 for r in rows if r[1] == "expired")
    lines = [
        "📊 <b>Paper-trade stats</b> (after est. fees)",
        f"Closed: {n} | Open: {open_n}",
        f"Win rate: {wins / n * 100:.0f}% | Avg: {tot / n:+.2f}R | Total: {tot:+.1f}R",
        f"Target hit: {tp} | Stop hit: {sl} | Expired: {ex_n}",
    ]
    for side in ("LONG", "SHORT"):
        sub = [r for r in rows if r[0] == side]
        if sub:
            sn, sw, st = block(sub)
            lines.append(f"{side}: {sn} trades, {sw / sn * 100:.0f}% win, {st / sn:+.2f}R avg")
    if n < 50:
        lines.append(f"<i>Only {n} closed trades: too few to judge. Aim for 100+.</i>")
    return "\n".join(lines)


def maybe_send_summary(force=False):
    row = db.execute("SELECT v FROM meta WHERE k='last_summary'").fetchone()
    last = float(row[0]) if row else 0
    if force or time.time() - last >= 86400:
        send_telegram(stats_text())
        db.execute("INSERT OR REPLACE INTO meta VALUES ('last_summary', ?)", (str(time.time()),))
        db.commit()


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
        f"<i>Paper-tracked automatically. Not financial advice.</i>"
    )


# ---------- Main loop ----------
def scan_once():
    update_open_trades()                      # settle earlier paper trades first
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
    maybe_send_summary()


if __name__ == "__main__":
    if os.getenv("ONCE") == "1":  # run a single scan and exit (for GitHub Actions)
        scan_once()
        if os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch":  # manual test run
            send_telegram(f"✅ Manual scan complete on {EXCHANGE}.")
            maybe_send_summary(force=True)
        raise SystemExit(0)
    send_telegram(f"✅ Scanner started on {EXCHANGE} (top {TOP_N} pairs, every {SCAN_EVERY_SEC}s)")
    while True:
        try:
            scan_once()
        except Exception as e:
            log.error("Scan failed: %s", e)
        time.sleep(SCAN_EVERY_SEC)
