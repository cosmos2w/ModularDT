# ThermalChannel hierarchical inverse dataset schema v1

The authoritative file is case-major HDF5. `D`, `c`, and `G` are stored once
per case; sixteen `R` variants are stored on a second axis. The public reader
flattens `(case, variant)` only at training time.

Required root attributes identify dataset/request/compact/full-plan schema
versions, fixed `M` and `K`, split/augmentation seeds, partial-debug status,
and case-ID hashes. Required groups are `design`, `context`, `plan`,
`functionals`, `requests`, `geometry`, `normalization`, `provenance`, and
`splits`. Numeric arrays are finite and compressed; strings are UTF-8; no
object pickles are permitted.

The exact array names and shape validation are executable in
`channelthermal.inverse.dataset_io.validate_inverse_hdf5`. The corresponding
flattened training contract is implemented by `InverseH5Dataset`.
