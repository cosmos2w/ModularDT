# HET-MF dataset browsing

All arrays live on the verified `/data2` filesystem. `standard.h5` and `expanded.h5` are ignored browsing links to the two self-contained processed training archives. Open them with `hetmf.training_data.HETMFDataset`, choosing an explicit split. `formal_standard` and `formal_expanded` expose the accepted raw references and release provenance; `readiness` exposes the audit receipts and report figures. The real Dataset directory retains these versioned contracts and schemas.

Run `python -m hetmf link-training --standard-h5 <data2-standard.h5> --expanded-h5 <data2-expanded.h5> --standard-release <release-root> --expanded-release <release-root> --readiness <audit-root> --case-dir <absolute-case-directory>` to create safe browsing links. The helper refuses regular destinations and foreign symlinks. It never copies or moves scientific payloads.

The earlier pilot remains separate. Its `raw`, `processed`, `figures`, `catalogues`, and `pilot` links refer to the pilot when present; they should not be mistaken for the formal training archive. `python -m hetmf link --root <pilot-root> --case-dir <case-directory>` creates those legacy links. See the case's detailed training-readiness report for membership, physical interpretation, and tested interfaces.
