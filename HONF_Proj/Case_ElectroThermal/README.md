# HET-MF: heterogeneous electro-thermal contact sheets

This case provides an independent conservative reference generator and a bounded pilot for the proposed `hetmf_contact_sheet_v1` dataset. It does not train HONF. The pilot report, numerical arrays, and selected figures describe observed behavior and the qualification limits. Formal generation requires a separately bound approval record and is not triggered by the pilot commands.

The approved Standard release completed on 2026-10-17; see [the release report](docs/HETMF_STANDARD_RELEASE_REPORT.md). In the home checkout, `Dataset/formal_standard` opens its verified `/data2` payload directly.

Use the verified ModularDT Python environment and set `PYTHONPATH=src` from this directory. See `docs/HETMF_GENERATION_GUIDE.md` for commands and `Dataset/PHYSICS_AND_DATA.md` for the input/target contract.
