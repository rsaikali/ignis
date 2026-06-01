"""eval: NILM prediction vs HA truth -> per-appliance diff/drift -> retrain request.

Pure logic lives here (metrics, drift trigger) so it is testable without a
running model or DB. Data alignment (read ha_samples, bucket to the common
grid, join truth vs prediction) is a thin layer added once inference writes
predictions -- see module TODOs.

Acceptance gate (CLAUDE.md): per-appliance state F1 >= 0.8 AND energy error
<= 10-15% vs HA. The diff is one quantity with three uses: eval metric,
portfolio hero, retrain signal.
"""
