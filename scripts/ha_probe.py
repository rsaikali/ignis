"""One-shot HA recorder probe: sampling rate + retention for NILM-relevant entities.

Run against the HA container (replace user@ha-host with yours):
    ssh user@ha-host 'docker exec -i homeassistant python3' < scripts/ha_probe.py
"""

import sqlite3
import statistics
import time

DB = "/config/home-assistant_v2.db"
PATTERNS = [
    "puissance",
    "four",
    "lave_linge",
    "lave_vaisselle",
    "televis",
    "pc",
    "linky",
    "ballon",
    "east",
]

c = sqlite3.connect(DB)
q = c.cursor()
now = time.time()

# Global recorder span: oldest retained state -> retention in days.
oldest = q.execute("SELECT MIN(last_updated_ts) FROM states").fetchone()[0]
newest = q.execute("SELECT MAX(last_updated_ts) FROM states").fetchone()[0]
if oldest:
    print("RETENTION: states span %.1f days (oldest -> newest raw state)" % ((newest - oldest) / 86400.0))

print("-" * 90)
seen = set()
for p in PATTERNS:
    rows = q.execute(
        "SELECT metadata_id, entity_id FROM states_meta WHERE entity_id LIKE ?",
        ("%" + p + "%",),
    ).fetchall()
    for mid, eid in rows:
        if mid in seen:
            continue
        seen.add(mid)
        ts = [
            r[0]
            for r in q.execute(
                "SELECT last_updated_ts FROM states WHERE metadata_id=? AND last_updated_ts>? ORDER BY last_updated_ts",
                (mid, now - 10800),
            )
        ]
        if len(ts) > 2:
            d = [b - a for a, b in zip(ts, ts[1:])]
            print(
                "%-48s n=%-5d med_dt=%6.1fs min=%5.1f max=%7.1f" % (eid, len(ts), statistics.median(d), min(d), max(d))
            )
        else:
            print("%-48s n=%d  (sparse/none in last 3h)" % (eid, len(ts)))
