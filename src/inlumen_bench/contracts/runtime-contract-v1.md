# Runtime and submission contract (v1)

This contract is the same for every method and model. Use it together with the task prompt.

## Submission layout
- Submit a directory of Python source files. The entry module is `definitions.py` at the top level.
- `definitions.py` must define `defs = dagster.Definitions(assets=[...])`.
- You may add more `.py` modules next to `definitions.py` (or under `src/`). Only Python source is allowed; no Dockerfiles, install scripts or data files.

## Graph
- Implement every logical task in the prompt as exactly one Dagster asset (`@dagster.asset`).
- Express each dependency as an asset input (a function parameter named after the upstream asset), so the asset graph matches the required topology. Independent branches must not depend on each other.
- Do not name an asset `context`; Dagster reserves that name.
{asset_names}

## Files
- Read input files from the directory in the environment variable `SERVICECRAFT_INPUT_DIR`, using the file names listed in the prompt (the prompt writes them as `inputs/<name>`; at run time they are at `$SERVICECRAFT_INPUT_DIR/<name>`). Inputs are read-only.
- Write every mandatory artifact directly into the directory in `SERVICECRAFT_OUTPUT_DIR`, with exactly the file name given in the prompt.
- Each asset must write its own artifacts while it materializes, from data computed in the current run. Never read earlier outputs or hard-code results.
- Serialize models with `pickle`, as `.pkl`. A model artifact is either a fitted scikit-learn estimator/`Pipeline`, or a dict with `model`, optional `scaler`, `features` (list of raw input column names) and, for clustering, optional `mapping`/`canonical_mapping` from raw to canonical labels.
- JSON artifacts must contain only finite numbers.

## Environment
- Python {python_version}, run in-process by Dagster {dagster_version} (`execute_in_process` on the implicit asset job).
- Only the packages listed below are installed; nothing can be installed at run time. There is no network access.
- The filesystem is read-only except `SERVICECRAFT_OUTPUT_DIR` and `/tmp`.
- Limits: {timeout_seconds} s wall-clock, {memory} memory, {cpus} CPU, 128 processes.

## Installed packages
```
{packages}
```
