"""training: self-supervised dataset + native (Metal) training.

Runs NATIVE on macOS, never in a container (see CLAUDE.md "Training is
OFF-device"). Reads the lab's TimescaleDB compatibility views, builds an
aligned (aggregate, per-appliance-truth) dataset from the free Meross labels,
trains the Seq2Point multi-output model, and exports a ``.keras`` artifact.

This deliberately bypasses Linkya's manual-signature path (which faked
``appliance_power = aggregate.copy()``): the project goal is the
self-supervised loop where HA per-appliance power IS the label.
"""
