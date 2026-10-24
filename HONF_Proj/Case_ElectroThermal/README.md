# HET-MF: heterogeneous electro-thermal contact sheets

This case provides an independent conservative reference generator and a bounded pilot for the proposed `hetmf_contact_sheet_v1` dataset. It does not train HONF. The pilot report, numerical arrays, and selected figures describe observed behavior and the qualification limits. Formal generation requires a separately bound approval record and is not triggered by the pilot commands.

The approved Standard release completed on 2026-10-17; see [the release report](docs/HETMF_STANDARD_RELEASE_REPORT.md). The [dataset and training-readiness report](HETMF_DATASET_AND_TRAINING_READINESS_REPORT.md) covers the reviewed Standard and Expanded archives, exact split memberships, physical equations, saved-field galleries, numerical limits, and the tested case-local connection to the general HONF model. In the home checkout, `Dataset/standard.h5` and `Dataset/expanded.h5` open the standalone processed training files on `/data2`; `Dataset/formal_standard` and `Dataset/formal_expanded` expose physical source references.

Use the verified ModularDT Python environment and set `PYTHONPATH=src` from this directory. See `docs/HETMF_GENERATION_GUIDE.md` for commands and `Dataset/PHYSICS_AND_DATA.md` for the input/target contract.

## Start here

| Need | Entry point |
|---|---|
| Load the current training datasets | [Standard H5](Dataset/standard.h5), [Expanded H5](Dataset/expanded.h5) |
| Understand the physics, inputs, targets, and later HONF experiments | [Dataset and training-readiness report](HETMF_DATASET_AND_TRAINING_READINESS_REPORT.md) |
| Find original simulations, evidence, and retention rules | [Dataset storage index](Dataset/README.md) |
| Run simulations, processing, verification, or visualization | [Generation guide](docs/HETMF_GENERATION_GUIDE.md) |
| Inspect historical qualification and generation measurements | [Pilot report](docs/HETMF_PILOT_REPORT.md), [Standard release report](docs/HETMF_STANDARD_RELEASE_REPORT.md) |

The current training archives are separate from historical pilot outputs. Browse pilot files only through `Dataset/pilot/`; the generic `Dataset/raw`, `processed`, `figures`, `catalogues`, and `links/active_manifest.json` aliases were retired during the approved 2026-10-24 cleanup. Original simulations, scientific evidence, and source worktrees remain retained.
