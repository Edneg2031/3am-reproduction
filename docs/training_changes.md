# 3AM Training Logic and Change Log

This document records the current ScanNet++ training behavior and changes made
for the reproduction workflow. Update it whenever sampling, supervision,
optimization, metrics, visualization, or data paths change.

## Current ScanNet++ Training Flow

The active single-scene configuration is:

```text
configs/scannetpp_continuous.yaml
```

The current sampling logic is:

1. Randomly select a training scene. With only `06b5863f73` in the manifest,
   this always selects that scene.
2. Randomly select the start of a contiguous 8-frame clip.
3. Use the first frame of the clip as the reference and prompt frame.
4. Read all positive instance IDs in the reference mask.
5. Filter these IDs using the configured area, visibility, category, and
   explicit-ID rules.
6. Uniformly select one instance at random from the remaining eligible IDs.
7. Build the reference mask prompt and the target mask for the same instance
   across all eight frames.
8. If no instance passes the filters, discard the clip and sample another
   contiguous clip. Training raises `NoEligibleInstanceError` after 64 failed
   attempts instead of silently training with an empty or noisy target.

Therefore, instance selection is still random, but only inside the filtered
eligible set. It is not currently weighted by object area, category, frequency,
or tracking difficulty.

## Instance Eligibility

The defaults are defined under
`datasets.scannetpp.instance_sampling` in
`configs/scannetpp_continuous.yaml`.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `min_reference_pixels` | `256` | Minimum pixels in the original reference label map |
| `min_reference_area_ratio` | `0.0005` | Minimum fraction of the reference image |
| `max_reference_area_ratio` | `0.10` | Reject instances covering more than 10% of the reference image |
| `min_visible_frames` | `2` | Instance must be visible in at least two frames of the clip |
| `min_visible_pixels_per_frame` | `64` | Pixels required for a frame to count as visible |
| `sample_resample_attempts` | `64` | Maximum clip resampling attempts |

The thresholds are evaluated on the original instance label maps before the
SAM2 resize/letterbox transform.

### Structural Categories

The default excluded categories include wall, floor, ceiling, staircase, and
their common plural forms.

Category filtering works only when an `instances.json` entry contains a field
such as:

```json
{"id": 12, "category": "wall"}
```

The loader also accepts fields such as `label`, `class_name`,
`semantic_label`, and `name`.

The current ScanNet++ preprocessing script normally writes only `id`, `frames`,
and `pixels`. When category names are unavailable, training emits a warning and
continues using area/visibility thresholds. Confirmed structural instance IDs
can be excluded manually:

```yaml
datasets:
  scannetpp:
    instance_sampling:
      excluded_instance_ids: [12, 35]
```

The 10% maximum area threshold removes most wall and floor observations, but it
is not a semantic guarantee. A structural surface occupying less than 10% of a
particular reference frame can still be selected unless its category or ID is
available.

## Model Supervision

For the selected instance:

- The first frame receives a mask prompt.
- The same instance ID is supervised across all eight frames.
- Frames where the instance is absent use an empty target mask.
- SAM2 image-encoder parameters and MUSt3R are frozen.
- Feature Merger, SAM2 memory attention, memory encoder, and mask decoder are
  trainable.
- The loss is:

```text
20 * focal + dice + IoU L1 + occlusion cross-entropy
```

FoV sampling is disabled in the active configuration:

```yaml
fov_sampling_probability: 0.0
```

## Diagnostics and Outputs

Training logs include:

```text
scene=... instance_id=... sampling=continuous
```

Loss and tracking history:

```text
outputs/training_metrics/scannetpp_continuous/training_history.csv
outputs/training_metrics/scannetpp_continuous/training_curves.png
```

Tracking visualizations:

```text
outputs/visualizations/scannetpp_continuous/step_*.png
outputs/visualizations/scannetpp_continuous/step_*.mp4
```

Each visualization compares the reference prompt, ground truth, predicted mask,
overlap/error map, confidence, and per-frame IoU for the sampled 8-frame clip.

## Data Paths

The active configuration matches the defaults used by
`scripts/prepare_scannetpp_training_data.py`:

```text
data/processed/scannetpp/
data/processed/scannetpp_manifest.json
outputs/must3r_features/
```

The preparation command with `--precompute-must3r` already creates both the
manifest and MUSt3R feature cache. They do not need to be regenerated before
each training run.

## Change History

### 2026-06-23

- Added `configs/scannetpp_continuous.yaml`.
- Disabled FoV sampling and enabled contiguous 8-frame sampling.
- Added reference-instance area and visibility thresholds.
- Added structural category and explicit instance-ID exclusion.
- Added clip resampling when no eligible reference instance exists.
- Added training loss/tracking CSV history and curve rendering.
- Added 8-frame tracking visualization output.
- Added `instance_id` and sampling mode to training logs.
- Corrected ScanNet++ paths to `data/processed/...`.

