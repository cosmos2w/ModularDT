# Repository completion and upload rule

- After a round of code revision, commit the durable changes and push the current non-default working branch. Fetch or inspect the remote branch before pushing, then verify the remote and local branch tips match. A local commit alone does not finish the round.
- Before every push, audit the **entire outgoing commit range**, including files that an earlier outgoing commit added and a later one deleted. Never upload highly one-time diagnostic code, generated figures, or generated data. Keep those artifacts locally in ignored paths. Do not force-add ignored outputs.
- If forbidden files are already in unpublished commits, rebuild the outgoing history before pushing. Preserve a local backup first, and leave the working checkout synchronized with the pushed branch.
- Run `git config core.hooksPath .githooks` in this clone so the repository pre-push artifact gate runs. Review its result and the outgoing file list; the gate supplements the required human audit.
