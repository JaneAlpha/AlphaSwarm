from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="MAS-FactorMiner command runner")
    parser.add_argument(
        "command",
        choices=["build-base-factors", "run-loop-once", "serve", "serve-dashboard"],
        help="Pipeline command to run.",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/base_factors.json",
        help="Output path for build-base-factors.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum loop iterations for run-loop-once.",
    )
    parser.add_argument(
        "--factors-per-round",
        type=int,
        default=3,
        help="Number of LLM-discovered factors requested per iteration.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the web server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the web server.",
    )
    args = parser.parse_args()

    if args.command == "build-base-factors":
        from src.base_factor_generator import BaseFactorGenerator

        result = BaseFactorGenerator().write(Path(args.output))
        quality = result["quality"]
        print(
            "base factors built: "
            f"{quality['factor_count']} factors, "
            f"{quality['symbol_count']} symbols, "
            f"latest_date={result['latest_date']}"
        )
    elif args.command == "run-loop-once":
        from src.llm_client import LLMConfigurationError
        from src.pipeline import SingleRunFactorMiningPipeline

        try:
            status = SingleRunFactorMiningPipeline(
                max_iterations=args.max_iterations,
                factors_per_round=args.factors_per_round,
            ).run()
            summary = status["summary"]
            print(
                "tool-call pipeline completed: "
                f"{status['max_iterations']} iterations, "
                f"{summary['total_discovered_factors']} total discovered, "
                f"{summary['calculation_success']} calculated, "
                f"{summary['calculation_failure']} failed"
            )
        except LLMConfigurationError as exc:
            raise SystemExit(
                f"{exc}\n"
                "Configure DeepSeek before running real tool-call factor mining:\n"
                "  DEEPSEEK_API_KEY=your_key\n"
                "  DEEPSEEK_MODEL=deepseek-chat\n"
                "  DEEPSEEK_BASE_URL=https://api.deepseek.com"
            ) from exc
    elif args.command == "serve":
        from src.web_server import FactorMinerWebServer

        FactorMinerWebServer(host=args.host, port=args.port).serve()
    elif args.command == "serve-dashboard":
        from src.dashboard_server import FactorMinerDashboardServer

        FactorMinerDashboardServer(host=args.host, port=args.port).serve()


if __name__ == "__main__":
    main()
