Quick checks
- `pytest -q`

Canary sweep (synthetic)
- `python canary_sweep.py --use-test-config`

Canary sweep (real aligned CSV)
- `python canary_sweep.py --aligned-csv path\\to\\period_returns_aligned.csv --n-configs 10 --warmup-periods 20`

Notes
- Canary results are written under `canary_output/` by default.
- Add `--enable-plots` to generate plots during the canary sweep.
