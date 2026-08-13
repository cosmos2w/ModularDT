# Support and compatibility

This 0.1.x release supports Python 3.10 or newer, PyTorch CPU execution, and
CUDA when provided by the installed PyTorch build. The validated environment
is recorded in `VALIDATION.md` and in every run manifest.

For a reproducible report, include the command, core/case/overlay configs,
`run_manifest.json`, `environment/software.json`, checkpoint selector, dataset
logical ID and fingerprint, and the full traceback. Do not attach private HDF5
files or checkpoints unless their distribution is authorized.

Compatibility policy:

- patch releases preserve schema-v1 configuration and checkpoint behavior;
- new required fields or changed tensor semantics require a new schema version;
- historical unversioned ThermalChannel checkpoints remain trusted-local inputs
  to the explicit compatibility loader;
- deprecated aliases are removed only after a documented migrator exists;
- `honf_inverse` is reserved and unsupported in 0.1.x.

The repository currently carries an all-rights-reserved notice. Contact the
project owner before external distribution or before selecting a public
software license.
