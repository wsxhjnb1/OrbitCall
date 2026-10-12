# Release interface and evidence

After generation, reference ranking, numerical audits, timing, label evaluation,
analysis and two independent artifact reviews were completed, the public API
received one interface correction: `SequenceScorer.score` defaults to `full`
instead of `tree`. Tree/cache execution now requires an explicit method choice.
This avoids implicitly choosing the diagnostic path whose 3B numerical gate
failed. No implementation of a numerical path, kernel, precision, candidate,
prediction, threshold, fixture or comparison changes.

Every recorded experiment already supplied its method explicitly. Original
experiment sources remain byte-identical under each raw run's `source/`
directory. `validate_release_interface.py` checks the original scoring manifests
and verifies that replacing exactly that one default declaration produces the
release source bytes. The complete 230-test suite is rerun on Spark after the
correction. The [release receipt](../metadata/release_interface_validation.json)
records both source hashes and the test-log hash. This test repetition validates
the interface release; it is not a new quality or timing experiment.

The independent quality and systems artifact receipts validate the historical
study before this API correction; their original source hashes remain truthful.
For byte-exact replay of those historical validators or model audits, restore
the original source snapshot listed by the run manifest in a separate checkout.
Do not rewrite manifests to the release hash. A new full study using the release
code freezes its own source receipts and fresh run IDs as described in
[REPRODUCE.md](REPRODUCE.md).

The accepted conclusions remain a scoped resident-kernel benefit, no
mass-over-strong-control accuracy gain, and essentially unchanged full-scorer
latency. No new experimental conclusion follows from the default correction.
