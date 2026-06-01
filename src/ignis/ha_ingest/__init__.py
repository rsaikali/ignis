"""ha_ingest: Home Assistant -> Ignis TimescaleDB.

Primary path is MQTT-push via HA ``mqtt_statestream`` (see CLAUDE.md and the
ingestion decision). The subscriber stores raw on-arrival samples at full
resolution; resampling onto the common grid happens later in ``eval``.
"""
