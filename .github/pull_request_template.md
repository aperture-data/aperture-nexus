## What this PR does

<!-- One sentence describing the feature or fix. -->

## Why

<!-- The motivation — what problem does this solve or what capability does it add? -->

## Changes

<!-- Bullet list of what changed and where. -->
-

## Test plan

- [ ] Unit tests added in `tests/test_<module>.py`
- [ ] All tests pass: `pytest tests/`
- [ ] Integration tests added in `tests/integration/` (if DB interaction involved)

## Security checklist

- [ ] No credentials hardcoded
- [ ] No system paths assumed (`/etc/`, `/var/`, `/root/`)
- [ ] All ports > 1024
- [ ] User input validated before any DB interaction
- [ ] Original exceptions chained (`raise ... from e`)
- [ ] `.env` not committed

## Notes for reviewer

<!-- Anything non-obvious the reviewer should know. -->
