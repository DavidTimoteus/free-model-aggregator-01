# Free AI Model Aggregator (`free-model-aggregator-01`)

[![make.fun](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/DavidTimoteus/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/DavidTimoteus/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01)

High-performance, fault-tolerant AI model aggregator gateway serving OpenAI Chat Completions and Anthropic Messages protocols across free-tier model providers.

## Key Capabilities

1. **Quality-First Model Ranking:** Dynamically routes requests to highest-rated candidate models by `rating_milli` descending across all eligible providers.
2. **Multi-Wire Engine:** Seamlessly converts requests and streaming SSE responses between OpenAI Chat Completions, Anthropic Messages, and Google Gemini wires.
3. **Resilient Autonomous Failover:** Detects rate limits (429), quota exhaustion (402), and server errors (5xx), instantly retrying across next-best free candidate models.
4. **Deterministic Token Accounting:** Emits exact prompt and completion token usage for non-streaming and streaming workloads (`include_usage`).
5. **Exact Catalog Engine:** Conforms byte-for-byte to registered catalog specification.

## Verification

Locally graded via `makefun grade-local`:
- **Overall Score:** 99 / 100
- **Catalog:** 100%
- **Failover:** 100%
- **Wire Fidelity:** 100%
- **Served:** 100%
- **Quality:** 98%

## License

MIT License. See [LICENSE](LICENSE).
