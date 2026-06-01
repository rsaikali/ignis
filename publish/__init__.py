"""publish: Ignis output surfaces.

Two consumers, one broker (see CLAUDE.md "Two output surfaces"):

1. **Portfolio MQTT** -- the frozen ``nilm/...`` contract from the portfolio
   spec section 6.4 (``contract.py`` builds the exact payloads).
2. **HA entity bridge** -- MQTT discovery configs so Home Assistant
   auto-creates sensors without a custom component (``ha_discovery.py``).
   The full custom_components/ignis integration + LitElement cards remain a
   separate TODO (recoverable from git 457cc1b).

Payload construction is pure and testable; the MQTT send is a thin layer.
"""
