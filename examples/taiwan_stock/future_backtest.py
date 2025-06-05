import sqlite3
from collections import deque

DB_PATH = 'examples/taiwan_stock/futures.db'
SHORT_WINDOW = 15  # minutes
LONG_WINDOW = 60   # minutes


def iterate_prices(db_path: str):
    """Yield (timestamp, close) ordered by time."""
    conn = sqlite3.connect(db_path)
    cur = conn.execute('SELECT timestamp, close FROM futures_kbars ORDER BY timestamp')
    for ts, close in cur:
        yield ts, close
    conn.close()


def backtest():
    short_q = deque(maxlen=SHORT_WINDOW)
    long_q = deque(maxlen=LONG_WINDOW)

    position = 0  # 1 for long, -1 for short, 0 for flat
    entry_price = 0.0
    cash = 0.0

    for ts, price in iterate_prices(DB_PATH):
        short_q.append(price)
        long_q.append(price)
        if len(long_q) < LONG_WINDOW:
            continue  # wait until enough data for moving averages

        short_avg = sum(short_q) / len(short_q)
        long_avg = sum(long_q) / len(long_q)

        # Golden cross: go long
        if short_avg > long_avg and position <= 0:
            if position == -1:
                cash += entry_price - price  # profit from short position
            position = 1
            entry_price = price
        # Death cross: go short
        elif short_avg < long_avg and position >= 0:
            if position == 1:
                cash += price - entry_price  # profit from long position
            position = -1
            entry_price = price

    # close any remaining position at the final price
    if position == 1:
        cash += price - entry_price
    elif position == -1:
        cash += entry_price - price
    print(f"Final PnL: {cash:.2f}")


if __name__ == "__main__":
    backtest()
