## Summary
- What changed?
- Why was this needed?

Stack note: this repo is **Python + shell** (no app **JavaScript/TypeScript** / npm). Run **`make ci-fast`** before opening PRs (see [docs/CI_LOCAL.md](docs/CI_LOCAL.md)).

## Security Checklist
- [ ] Local quality gate: `make ci-fast` (or `make ci` for training-path / lockfile changes)
- [ ] No secrets or credentials added
- [ ] Dependency changes are intentional
- [ ] Risky behavior changes are documented
