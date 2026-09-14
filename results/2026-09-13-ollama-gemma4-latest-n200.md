# Model comparison: ollama/gemma4:latest

- Items sampled: 200 (200 scored, 0 errors/unparseable)
- Decision agreement with Claude's original keep/dismiss: **72%**
- Mean absolute score difference: 0.23
- Avg latency per item: 34.87s

## The two error types that actually matter

- **False dismiss** (Claude kept it, candidate would drop it — a real item silently lost): 30%
- **False keep** (Claude dismissed it, candidate would surface it — more noise to review): 27%

False dismiss is the expensive error: it's not a false alarm, it's an item that never gets a second chance.
