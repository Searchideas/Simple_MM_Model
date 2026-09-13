"""Set config.Exchange and print it. Usage: python _set_exchange.py bybit"""
from pathlib import Path
import re
import sys

ex = sys.argv[1]
assert ex in {"binance-futures", "gate-io-futures", "bybit"}, ex
p = Path(__file__).resolve().parent / "config.py"
t = p.read_text(encoding="utf-8")
t2, n = re.subn(r'^Exchange = "[^"]*"', f'Exchange = "{ex}"', t, count=1, flags=re.M)
assert n == 1, "Exchange line not found"
p.write_text(t2, encoding="utf-8")
print(f"Exchange -> {ex}")
