import argparse
import json
import time
from typing import Dict, Tuple

import numpy as np
from flask import Flask, request, jsonify


def simulate_buffon_needle(
    num_needles: int,
    needle_length: float = 1.0,
    line_spacing: float = 1.0,
    seed: int = None,
    batch_size: int = 1000000,
) -> Tuple[float, int, int]:
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if needle_length > line_spacing:
        raise ValueError("needle_length must be <= line_spacing for standard Buffon formula")

    l_over_d = needle_length / line_spacing

    total_hits = 0
    remaining = num_needles

    while remaining > 0:
        current_batch = min(remaining, batch_size)

        y = rng.uniform(0, line_spacing / 2, size=current_batch)
        theta = rng.uniform(0, np.pi / 2, size=current_batch)

        half_projection = (needle_length / 2) * np.sin(theta)
        hits = np.sum(y <= half_projection)

        total_hits += int(hits)
        remaining -= current_batch

    if total_hits == 0:
        return float("inf"), num_needles, 0

    pi_estimate = (2.0 * l_over_d * num_needles) / total_hits

    return pi_estimate, num_needles, total_hits


def compute_statistics(
    num_needles: int,
    needle_length: float,
    line_spacing: float,
    pi_estimate: float,
    total_hits: int,
) -> Dict:
    probability = total_hits / num_needles if num_needles > 0 else 0.0
    error_abs = abs(pi_estimate - np.pi)
    error_rel = error_abs / np.pi * 100
    theoretical_prob = (2.0 * needle_length) / (np.pi * line_spacing)

    return {
        "needle_length": needle_length,
        "line_spacing": line_spacing,
        "num_needles": num_needles,
        "total_hits": total_hits,
        "hit_probability": probability,
        "theoretical_hit_probability": theoretical_prob,
        "pi_estimate": pi_estimate,
        "pi_true": np.pi,
        "absolute_error": error_abs,
        "relative_error_percent": error_rel,
    }


app = Flask(__name__)


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    data = request.get_json(force=True, silent=True) or {}

    num_needles = int(data.get("num_needles", 1000000))
    needle_length = float(data.get("needle_length", 1.0))
    line_spacing = float(data.get("line_spacing", 1.0))
    seed = data.get("seed")
    if seed is not None:
        seed = int(seed)

    if num_needles <= 0:
        return jsonify({"error": "num_needles must be positive"}), 400

    if needle_length <= 0 or line_spacing <= 0:
        return jsonify({"error": "needle_length and line_spacing must be positive"}), 400

    if needle_length > line_spacing:
        return jsonify({"error": "needle_length must be <= line_spacing"}), 400

    try:
        start = time.perf_counter()
        pi_estimate, needles, hits = simulate_buffon_needle(
            num_needles, needle_length, line_spacing, seed
        )
        elapsed = time.perf_counter() - start

        stats = compute_statistics(needles, needle_length, line_spacing, pi_estimate, hits)
        stats["elapsed_seconds"] = elapsed
        stats["needles_per_second"] = needles / elapsed if elapsed > 0 else 0.0

        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok", "service": "buffon-needle-simulator"})


@app.route("/", methods=["GET"])
def index():
    return (
        "<h1>布丰投针 π 近似计算服务</h1>"
        "<p>使用 <code>POST /api/simulate</code> 进行模拟</p>"
        "<pre>"
        "请求示例：<br>"
        "curl -X POST http://localhost:5000/api/simulate \\<br>"
        "  -H 'Content-Type: application/json' \\<br>"
        "  -d '{\"num_needles\": 1000000, \"needle_length\": 1.0, \"line_spacing\": 1.0}'"
        "</pre>"
    )


def main():
    parser = argparse.ArgumentParser(description="布丰投针 π 近似计算器")
    subparsers = parser.add_subparsers(dest="command", help="命令")

    sim_parser = subparsers.add_parser("simulate", help="运行单次模拟")
    sim_parser.add_argument(
        "-n", "--num-needles", type=int, default=1000000, help="投针次数（默认：1000000）"
    )
    sim_parser.add_argument(
        "-l", "--needle-length", type=float, default=1.0, help="针长（默认：1.0）"
    )
    sim_parser.add_argument(
        "-d", "--line-spacing", type=float, default=1.0, help="平行线间距（默认：1.0）"
    )
    sim_parser.add_argument("-s", "--seed", type=int, default=None, help="随机种子")
    sim_parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1000000,
        help="批处理大小（默认：1000000）",
    )

    serve_parser = subparsers.add_parser("serve", help="启动 HTTP API 服务")
    serve_parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认：127.0.0.1）")
    serve_parser.add_argument("--port", type=int, default=5000, help="监听端口（默认：5000）")
    serve_parser.add_argument("--debug", action="store_true", help="调试模式")

    args = parser.parse_args()

    if args.command == "serve":
        app.run(host=args.host, port=args.port, debug=args.debug)

    elif args.command == "simulate":
        if args.num_needles <= 0:
            parser.error("num_needles 必须为正整数")

        if args.needle_length <= 0 or args.line_spacing <= 0:
            parser.error("needle_length 和 line_spacing 必须为正数")

        if args.needle_length > args.line_spacing:
            parser.error("needle_length 必须 <= line_spacing")

        start = time.perf_counter()
        pi_estimate, needles, hits = simulate_buffon_needle(
            args.num_needles,
            args.needle_length,
            args.line_spacing,
            args.seed,
            args.batch_size,
        )
        elapsed = time.perf_counter() - start

        stats = compute_statistics(
            needles, args.needle_length, args.line_spacing, pi_estimate, hits
        )

        print("=" * 50)
        print("布丰投针 π 近似计算结果")
        print("=" * 50)
        print(f"投针次数     : {stats['num_needles']:,}")
        print(f"针长         : {stats['needle_length']}")
        print(f"平行线间距   : {stats['line_spacing']}")
        print(f"命中次数     : {stats['total_hits']:,}")
        print(f"命中概率     : {stats['hit_probability']:.6f}")
        print(f"理论命中概率 : {stats['theoretical_hit_probability']:.6f}")
        print(f"π 估计值    : {stats['pi_estimate']:.10f}")
        print(f"π 真实值    : {stats['pi_true']:.10f}")
        print(f"绝对误差     : {stats['absolute_error']:.10f}")
        print(f"相对误差     : {stats['relative_error_percent']:.6f}%")
        print(f"耗时         : {elapsed:.4f} 秒")
        print(f"吞吐量       : {needles / elapsed:,.0f} 针/秒")
        print("=" * 50)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
