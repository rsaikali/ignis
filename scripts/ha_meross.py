"""List all Meross mss315 entities (power W + energy kWh) with last value + 24h count."""

import sqlite3
import time

c = sqlite3.connect("/config/home-assistant_v2.db")
q = c.cursor()
now = time.time()

rows = q.execute(
    "SELECT metadata_id, entity_id FROM states_meta "
    "WHERE entity_id LIKE '%mss315%' ORDER BY entity_id"
).fetchall()
print("total mss315 entities:", len(rows))
for mid, eid in rows:
    r = q.execute(
        "SELECT state, last_updated_ts FROM states "
        "WHERE metadata_id=? ORDER BY last_updated_ts DESC LIMIT 1",
        (mid,),
    ).fetchone()
    n = q.execute(
        "SELECT COUNT(*) FROM states WHERE metadata_id=? AND last_updated_ts>?",
        (mid, now - 86400),
    ).fetchone()[0]
    print("%-52s last=%-10s n24h=%d" % (eid, (r[0] if r else "?"), n))
