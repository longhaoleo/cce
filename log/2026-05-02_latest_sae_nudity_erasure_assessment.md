# Latest SAE Nudity Erasure Assessment

## Context

Checkpoint:

- `train/output_time_decorr_stage23_half_no_stage1/checkpoints/stage3_step_0013772`

Recent test:

- `nudity` erasure using the latest `no stage1` SAE
- output includes `image_output/batch_shared_concept_erase_nudity_stage23_half`

## Observation

Compared with the previous SAE, the latest model produces more stable and higher quality images after intervention.

The intervention can now remove the target `nudity` concept very thoroughly. This is a real improvement over the earlier failure mode where the image quality could degrade while the target concept remained visible.

However, the intervention is still too destructive. In many cases the erased image is heavily rewritten, and the remaining content can become hard to identify. This indicates that the model is capable of strong concept suppression, but the selected feature subspace and/or intervention strength still removes too much non-target information.

## Interpretation

This is not a simple "intervention did not apply" problem.

The current failure mode is:

- better image stability than the previous SAE
- strong target concept removal
- excessive semantic and structural drift

So the next optimization target should shift from "can it erase" to "can it erase while preserving non-target content."

## Likely Causes

- The new SAE may have learned a cleaner and more stable representation, but the concept feature set for `nudity` is still too broad.
- The current feature selection may include body/pose/lighting/scene features alongside the actual target concept.
- The intervention strength and time weighting may be too aggressive for the latest checkpoint.
- The blacklist should be regenerated from the latest SAE feature frequency statistics before drawing final conclusions.

## Next Steps

1. Regenerate feature frequency and blacklist for the latest checkpoint:

   - `scripts/feature_frequency_latest_sae.md`

2. Rerun `nudity` locator using the new blacklist.

3. Rerun batch erasure with smaller intervention settings, for example:

   - lower `--int_scale`
   - lower `--int_time_weight_scale`
   - smaller `--int_feature_top_k`

4. Compare against the previous SAE on the same prompts using:

   - target suppression
   - identity/content preservation
   - image stability

## Current Conclusion

The latest SAE is directionally better for generation stability and can erase `nudity`, but the erasure is currently overpowered and harms image semantics. The next work item is to reduce collateral damage, not to increase erasure strength.
