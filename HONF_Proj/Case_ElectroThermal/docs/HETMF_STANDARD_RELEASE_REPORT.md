# HET-MF Standard release, 2026-10-17

The frozen Standard input plan was generated on physical GPU2 in FP64 and packed into verified raw and model-facing processed shards. All 4,608 planned states completed, including the M10, M16, high-contrast, and composite challenge groups. The release is browsable at [Dataset/formal_standard](../Dataset/formal_standard) in the home checkout; its payload remains on the verified `/data2` mount. No HONF training was run.

## Scope and measured result

| Item | Result |
|---|---:|
| Input plan SHA-256 | `a94ddd9e0374fb9a971f4856944b2f8cc475749db6b2f797cddccbd3afa29e6b` |
| Approval SHA-256 | `96b9a45e0f7ea1057ad59f58a34b4bb84097e72e1167f1508caae1d638536da2` |
| States | 4,608 accepted / 4,608 planned; 0 failed |
| Split membership | TRAIN 2,880; DEV 480; TEST 480; audit 256; four panel splits 128 each |
| Native grid and precision | 96 × 96, GPU2 FP64; processed target channels stored as float32 |
| Raw generation wall time | 272.58 s, including solves, writes, and per-case post-storage audits |
| Packed output | 73 raw shards, 73 processed shards; 4,608 FP64 source cases retained |
| Apparent release size | 5,914,570,355 bytes (5.51 GiB) including plans, ledgers, source cases, shards, and selected figures |
| Resource limits | 8 GiB retained, 1 GiB temporary, 100 GiB minimum free; concurrent GPU2 capped at 25% process memory and 16 GiB free-memory floor |

The exact approval record is [formal_approval.json](../Dataset/formal_standard/inputs/formal_approval.json), and the independent completion and packing receipts are [formal_progress.json](../Dataset/formal_standard/diagnostics/formal_progress.json) and [formal_pack_receipt.json](../Dataset/formal_standard/diagnostics/formal_pack_receipt.json). All 4,608 saved FP64 source cases were checked against their frozen input and solver hashes and re-audited for physical balance before packing; each shard was checked after its atomic write. The other user's GPU2 process stayed present throughout the run, so the 272.58-second generation time is a shared-device measurement. The smallest observed GPU2 free memory before a batch was 44,269 MiB, above the approved 16,384 MiB floor.

![Saved Standard FP64 potential, temperature rise, and Joule heat density for ordinary M2, M16, and high-contrast cases](../Dataset/formal_standard/figures/standard_saved_fields.png)

Figure 1 uses saved native 96 × 96 FP64 source arrays for `formal:train:e0000:m2:l0`, `formal:panel:extrapolation:m16:0000`, and `formal:panel:extrapolation:high_contrast:0000`; axes are cell indices and physical quantities use the case's dimensionless units. Their post-storage relative current-balance errors are `1.84e-12`, `4.87e-12`, and `2.51e-12`, respectively. The M16 row shows 16 finite-area contact responses and corresponding localized Joule heating; the image supports saved-field inspection, not any learned-model claim. [PDF master](../Dataset/formal_standard/figures/standard_saved_fields.pdf).

![Post-storage electrical and thermal residuals and current-balance errors for all 4,608 Standard states](../Dataset/formal_standard/figures/standard_post_storage_audit.png)

Figure 2 plots independent post-storage relative electrical residual, thermal residual, and terminal-current balance for every frozen state in plan order. Maxima were `1.00e-10`, `1.00e-10`, and `1.07e-10`, respectively; the maximum relative electrical and thermal power-balance errors were `2.27e-15` and `1.39e-12`. Every state passed the `1e-5` post-storage relative acceptance limit, and saved exterior face flux was exactly zero in the audit. [PDF master](../Dataset/formal_standard/figures/standard_post_storage_audit.pdf).

## Model boundary and next use

The release provides predictor targets and a frozen root-disjoint split; predictor accuracy has not been measured. Organizer interaction accuracy, sparse work, and executor savings have not been measured. Inverse design validity and sample trails have not been measured. Any later HONF campaign must bind this release identity, fit normalization on ordinary TRAIN only, and report those three outcomes separately. The earlier [pilot report](HETMF_PILOT_REPORT.md) remains a dated pilot record and its forecasts are not formal measurements.

## Retention

Retain the frozen plan, exact approval, all source cases and shards, append-only solver and attempt ledgers, selected PDF masters and small Markdown raster companions, and completion receipts. The temporary staging directory was empty at closeout; no source cases or necessary intermediate evidence were removed. The home `Dataset` directory stays real because it contains tracked schemas and documentation; its ignored `formal_standard` link points to the full `/data2` release.

### Retention amendment: 2026-10-24

The retention paragraph above records the original release closeout. Following explicit cleanup approval, the 73 raw and 73 processed packaged shards were retired after every stored array and membership was checked against the retained FP64 originals and current standalone Standard training archive. All original source cases, plans, approvals, ledgers, figures, and original completion/packing receipts remain retained. The original packing counts and timings are historical measurements. See the [current storage index](../Dataset/README.md) and [cleanup receipt](../Dataset/readiness/receipts/tidy_20261024_v1/cleanup_receipt.json) for the current retention state and the exact verified removal list.
