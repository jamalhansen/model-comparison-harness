# Model comparison: ollama/llama3.2:3b

- Items sampled: 200 (195 scored, 5 errors/unparseable)
- Decision agreement with Claude's original keep/dismiss: **57%**
- Mean absolute score difference: 0.33
- Avg latency per item: 1.41s

## The two error types that actually matter

- **False dismiss** (Claude kept it, candidate would drop it — a real item silently lost): 78%
- **False keep** (Claude dismissed it, candidate would surface it — more noise to review): 7%

False dismiss is the expensive error: it's not a false alarm, it's an item that never gets a second chance.
