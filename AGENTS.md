# Repository completion and upload rule

- After a round of code revision, commit the durable changes and push the current non-default working branch. Fetch or inspect the remote branch before pushing, then verify the remote and local branch tips match. A local commit alone does not finish the round.
- Before every push, audit the **entire outgoing commit range**, including files that an earlier outgoing commit added and a later one deleted. Never upload highly one-time diagnostic code, generated figures, or generated data. Keep those artifacts locally in ignored paths. Do not force-add ignored outputs.
- If forbidden files are already in unpublished commits, rebuild the outgoing history before pushing. Preserve a local backup first, and leave the working checkout synchronized with the pushed branch.
- Run `git config core.hooksPath .githooks` in this clone so the repository pre-push artifact gate runs. Review its result and the outgoing file list; the gate supplements the required human audit.

# Scientific experiment reporting

- Begin final experiment reports with a plain-language account of what changed, then state gains, misses, measured evidence, and next steps separately for predictor, organizer, and inverse results. Distinguish completed measurements from startup checks, surrogate checks, and untested claims.
- Include readable, visually inspected figures from saved numerical evidence for physical fields and residuals, actual interaction graphs/support, fidelity versus work, responses, and inverse designs or sample trails. Label data source, partition, checkpoint, units, controls, failures, and physical-reference limits. Keep generated figures and one-time renderers in ignored local paths under the upload rule above.
- Retain only the most important result figures and remove superseded or redundant generated visual exports after validating the selected figures. Keep PDF as the default retained format; keep a small raster companion only when direct Markdown embedding is specifically required. Make the key figures easy to find in one short index.
- Embed the selected figures directly in the Markdown report when the display format supports it. Immediately after each figure, give its quantitative description and a concise summary of what the measurement supports or misses. Check every local figure link; do not add generated figures to Git.
- State whether sparse work and executor savings were actually measured; report full-access fallback separately. Do not interpret learned support as physical causality or sampler diversity as valid designs.
