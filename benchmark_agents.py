import argparse
import json
import statistics
import time

from non_neuralx import AgentConfig, HybridAgentV2, HybridConfig, NonNeuralAgent, NonNeuralAgentV2


def benchmark_agents(runs: int = 15, cache_dir: str | None = None, enable_sympy_checks: bool = False) -> dict:
    corpus = (
        "Non-neural methods are interpretable and efficient. "
        "Spectral memory retrieves grounded evidence. "
        "Symbolic reasoning supports transparent inference. "
    ) * 40
    query = "Explain why symbolic reasoning helps transparent inference."

    v1 = NonNeuralAgent()
    v2 = NonNeuralAgentV2(config=AgentConfig(retrieval_top_k=4, reasoning_top_k=4, reasoning_max_steps=5))
    hybrid = HybridAgentV2(
        config=HybridConfig(
            cache_dir=cache_dir,
            use_disk_cache=bool(cache_dir),
            enable_sympy_checks=enable_sympy_checks,
        )
    )
    v1.learn(corpus)
    v2.learn(corpus)
    hybrid.learn(corpus)

    v1_times = []
    v2_times = []
    hybrid_times = []

    for _ in range(runs):
        t0 = time.perf_counter()
        out1 = v1.think(query, max_length=220)
        v1_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        out2 = v2.think(query)
        v2_times.append(time.perf_counter() - t1)

        t2 = time.perf_counter()
        out3 = hybrid.think(query)
        hybrid_times.append(time.perf_counter() - t2)

    results = {
        "v1_mean_ms": statistics.mean(v1_times) * 1000,
        "v1_p95_ms": statistics.quantiles(v1_times, n=20)[-1] * 1000,
        "v2_mean_ms": statistics.mean(v2_times) * 1000,
        "v2_p95_ms": statistics.quantiles(v2_times, n=20)[-1] * 1000,
        "hybrid_mean_ms": statistics.mean(hybrid_times) * 1000,
        "hybrid_p95_ms": statistics.quantiles(hybrid_times, n=20)[-1] * 1000,
        "v1_answer_len": len(out1.get("answer", "")),
        "v2_answer_len": len(out2.get("answer", "")),
        "hybrid_answer_len": len(out3.get("answer", "")),
        "v2_confidence": float(out2.get("confidence", 0.0)),
        "hybrid_verified": bool(out3.get("reasoning", {}).get("verified", False)),
    }

    print("=== Agent Benchmark (V1 vs V2 vs Hybrid) ===")
    print(f"V1 mean latency : {results['v1_mean_ms']:.2f} ms")
    print(f"V1 p95 latency  : {results['v1_p95_ms']:.2f} ms")
    print(f"V2 mean latency : {results['v2_mean_ms']:.2f} ms")
    print(f"V2 p95 latency  : {results['v2_p95_ms']:.2f} ms")
    print(f"HY mean latency : {results['hybrid_mean_ms']:.2f} ms")
    print(f"HY p95 latency  : {results['hybrid_p95_ms']:.2f} ms")
    print(f"V1 answer length: {results['v1_answer_len']}")
    print(f"V2 answer length: {results['v2_answer_len']}")
    print(f"HY answer length: {results['hybrid_answer_len']}")
    print(f"V2 confidence   : {results['v2_confidence']:.3f}")
    print(f"HY verified     : {results['hybrid_verified']}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark NonNeuralX agent variants.")
    parser.add_argument("--runs", type=int, default=15, help="Number of benchmark iterations.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional disk cache dir for hybrid embeddings.")
    parser.add_argument("--sympy", action="store_true", help="Enable optional SymPy consistency checks in hybrid reasoner.")
    parser.add_argument("--json", action="store_true", help="Print benchmark result payload as JSON.")
    args = parser.parse_args()

    output = benchmark_agents(runs=args.runs, cache_dir=args.cache_dir, enable_sympy_checks=args.sympy)
    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
