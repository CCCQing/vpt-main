# P1-3a head-only protocol

P1-3a freezes each paired A2 checkpoint, extracts each split's final CLS once,
and trains only a shared candidate-conditioned compatibility head. Every
trainable condition uses all official Seen classes for a predeclared fixed
number of epochs. Post-training evaluation uses the official final-GZSL Seen and Unseen
splits.

The formal conditions are `candidate_raw312`, `candidate_projected768`,
`semantic_permuted_raw312`, `image_constant_raw312`,
`image_only_temperature`, and `temperature_only`. Frozen references are the
current semantic dot-product and the same-feature semantic cosine readout.

The three independent head/training seeds are paired with A2 seeds 0/1/2. The
three fixed Probe selection seeds are post-hoc robustness views and never count
as independent training repetitions. A run is complete only after all three
artifacts pass `src.tools.validate_p13a_compatibility` and the aggregate is
written successfully.
