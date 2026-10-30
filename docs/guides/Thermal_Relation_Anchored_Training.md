# Thermal relation-anchored training

The maintained `relation_anchored` and `relation_anchored_active` TRAIN600 recipes both use the detail-preserving joint-field reader. Their access measures are `inventory_unit_total` and `active_domain`, respectively. They use the existing full TRAIN600 membership, train-only normalization, canonical VAL89 and fixed input manifest. The first segment ends at 100 epochs; its 500-epoch optimizer schedule permits the selected candidate's authorized continuation after review. Health observations do not automatically pause the run.

Set `member_routing.model.query_tile_size` in an explicit recipe before training to tune decoder throughput and memory for your inputs. It changes query chunking while retaining all sampled queries and the physical objective. Follow the [query-tiling guide](Query_Tiling.md) for a local forward/backward benchmark, numerical checks and the scope of changing an existing checkpoint's setting.

Choose the public name directly. Set the physical GPU visibility and output path for the intended run before launching; an epoch covers the declared training membership once.

```bash
python tools/honf_research.py train-member-routing --model relation_anchored --device cuda:0 --output /data2/wanglz/ModularDT/relation_anchored/train600/RUN_NAME --dry-run
python tools/honf_research.py train-member-routing --model relation_anchored_active --device cuda:0 --output /data2/wanglz/ModularDT/relation_anchored/train600/RUN_NAME --dry-run
```

Remove `--dry-run` to start an authorized run. Choose a unique run name directly under the configured DATA2 output root; the runner creates a discoverable alias at `Trained_Results/ThermalChannel/HONF_Forward_Runs/RUN_NAME` pointing to that payload and refuses an existing alias. The explicit recipes are [relation anchored](../../configs/research/thermal_relation_anchored_train600.json) and [active domain](../../configs/research/thermal_relation_anchored_active_train600.json); `--config` accepts an explicit recipe. The saved R14 architecture and workflow identifiers remain in these recipes so that their checkpoints use the maintained strict loader. Historical R13 recipes retain their original schemas and behavior.

Resume the same lineage with `python tools/honf_research.py resume-member-routing --run RUN_PATH --device cuda:0 --stop-after-epoch EPOCH`. The continuation from 100 to at most 500 epochs stays within this recipe's configured horizon. Continuing beyond it requires explicit authorization and `--extend-horizon`. Evaluate a saved checkpoint with `python tools/honf_research.py evaluate-member-routing --checkpoint CHECKPOINT --device cpu --output OUTPUT_PATH`.
