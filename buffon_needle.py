import argparse
import json
import math
import time
from typing import Dict, List, Tuple

import numpy as np
from flask import Flask, request, jsonify

MIN_RECOMMENDED_NEEDLES = 10000

RELIABILITY_TIERS = [
    (1000000, "excellent", "~0.1%"),
    (100000, "good", "~0.3%"),
    (10000, "fair", "~1%"),
    (0, "unreliable", ">1%"),
]


def assess_reliability(num_needles: int) -> Dict:
    warnings: List[str] = []
    recommended = MIN_RECOMMENDED_NEEDLES

    if num_needles < MIN_RECOMMENDED_NEEDLES:
        warnings.append(
            f"投针次数 ({num_needles:,}) 低于建议最低值 ({MIN_RECOMMENDED_NEEDLES:,})，"
            f"结果可能严重偏离 π，建议至少投针 {MIN_RECOMMENDED_NEEDLES:,} 次"
        )

    if num_needles < 100:
        warnings.append(
            "投针次数极少，统计波动极大，π 估计值几乎无参考价值"
        )

    tier_name = "unreliable"
    tier_accuracy = ">1%"
    for threshold, name, accuracy in RELIABILITY_TIERS:
        if num_needles >= threshold:
            tier_name = name
            tier_accuracy = accuracy
            break

    l_over_d = 1.0
    p = (2.0 * l_over_d) / np.pi
    if num_needles > 0 and p > 0:
        se_rel = math.sqrt((1 - p) / (p * num_needles))
        expected_error_pct = se_rel * 100
    else:
        expected_error_pct = float("inf")

    if num_needles < 1000:
        recommended = 10000
    elif num_needles < 100000:
        recommended = 100000
    elif num_needles < 1000000:
        recommended = 1000000
    else:
        recommended = num_needles

    return {
        "tier": tier_name,
        "tier_accuracy": tier_accuracy,
        "warnings": warnings,
        "expected_relative_error_percent": round(expected_error_pct, 4),
        "recommended_min_needles": recommended,
    }


def simulate_buffon_needle(
    num_needles: int,
    needle_length: float = 1.0,
    line_spacing: float = 1.0,
    seed: int = None,
    batch_size: int = 1000000,
    max_visual_needles: int = 0,
    num_visual_lines: int = 5,
) -> Tuple[float, int, int, Dict]:
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if needle_length > line_spacing:
        raise ValueError("needle_length must be <= line_spacing for standard Buffon formula")

    if max_visual_needles < 0:
        max_visual_needles = 0

    l_over_d = needle_length / line_spacing

    total_hits = 0
    remaining = num_needles

    vis_y_all = []
    vis_theta_all = []
    vis_x_all = []
    vis_hits_all = []
    vis_collected = 0

    while remaining > 0:
        current_batch = min(remaining, batch_size)

        y = rng.uniform(0, line_spacing / 2, size=current_batch)
        theta = rng.uniform(0, np.pi / 2, size=current_batch)

        half_projection = (needle_length / 2) * np.sin(theta)
        hit_mask = y <= half_projection
        hits = int(np.sum(hit_mask))

        total_hits += hits
        remaining -= current_batch

        if max_visual_needles > 0 and vis_collected < max_visual_needles:
            need_more = max_visual_needles - vis_collected
            take_count = min(current_batch, need_more)

            x = rng.uniform(
                -needle_length, num_visual_lines * line_spacing + needle_length,
                size=current_batch
            )

            vis_y_all.append(y[:take_count])
            vis_theta_all.append(theta[:take_count])
            vis_x_all.append(x[:take_count])
            vis_hits_all.append(hit_mask[:take_count])
            vis_collected += take_count

    if total_hits == 0:
        pi_estimate = float("inf")
    else:
        pi_estimate = (2.0 * l_over_d * num_needles) / total_hits

    visualization = {}
    if max_visual_needles > 0 and vis_collected > 0:
        y_arr = np.concatenate(vis_y_all)
        theta_arr = np.concatenate(vis_theta_all)
        x_arr = np.concatenate(vis_x_all)
        hits_arr = np.concatenate(vis_hits_all)

        half_l = needle_length / 2
        dx = half_l * np.cos(theta_arr)
        dy = half_l * np.sin(theta_arr)

        x1 = x_arr - dx
        y1 = y_arr - dy
        x2 = x_arr + dx
        y2 = y_arr + dy

        needles = []
        for i in range(len(x_arr)):
            needles.append({
                "center_x": float(x_arr[i]),
                "center_y": float(y_arr[i]),
                "angle_deg": float(np.degrees(theta_arr[i])),
                "x1": float(x1[i]),
                "y1": float(y1[i]),
                "x2": float(x2[i]),
                "y2": float(y2[i]),
                "is_hit": bool(hits_arr[i]),
            })

        line_y_positions = [i * line_spacing for i in range(num_visual_lines + 1)]

        visualization = {
            "needle_length": needle_length,
            "line_spacing": line_spacing,
            "num_lines": num_visual_lines + 1,
            "line_y_positions": line_y_positions,
            "viewport": {
                "x_min": float(-needle_length),
                "x_max": float(num_visual_lines * line_spacing + needle_length),
                "y_min": float(-needle_length / 2),
                "y_max": float(num_visual_lines * line_spacing + needle_length / 2),
            },
            "needles": needles,
            "num_visual_needles": len(needles),
            "num_hit": int(np.sum(hits_arr)),
            "num_miss": int(np.sum(~hits_arr)),
        }

    return pi_estimate, num_needles, total_hits, visualization


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

    reliability = assess_reliability(num_needles)

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
        "reliability": reliability,
    }


def generate_svg(visualization: Dict, width: int = 800, margin: int = 40) -> str:
    if not visualization or "needles" not in visualization:
        return ""

    vp = visualization["viewport"]
    line_spacing = visualization["line_spacing"]
    needle_length = visualization["needle_length"]
    line_y_positions = visualization["line_y_positions"]
    needles = visualization["needles"]

    world_w = vp["x_max"] - vp["x_min"]
    world_h = vp["y_max"] - vp["y_min"]
    height = int(width * world_h / world_w)

    scale_x = (width - 2 * margin) / world_w
    scale_y = (height - 2 * margin) / world_h

    def sx(x: float) -> float:
        return margin + (x - vp["x_min"]) * scale_x

    def sy(y: float) -> float:
        return height - margin - (y - vp["y_min"]) * scale_y

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<style>',
        '  .line { stroke: #333; stroke-width: 1.5; }',
        '  .needle-hit { stroke: #e74c3c; stroke-width: 1.2; opacity: 0.85; }',
        '  .needle-miss { stroke: #3498db; stroke-width: 1.2; opacity: 0.7; }',
        '  .hit-dot { fill: #e74c3c; r: 1.5; }',
        '  .miss-dot { fill: #3498db; r: 1.0; }',
        '</style>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fafafa"/>',
    ]

    for y_pos in line_y_positions:
        y_svg = sy(y_pos)
        svg_parts.append(
            f'<line class="line" x1="{margin}" y1="{y_svg:.1f}" '
            f'x2="{width - margin}" y2="{y_svg:.1f}"/>'
        )

    for n in needles:
        cls = "needle-hit" if n["is_hit"] else "needle-miss"
        x1s = sx(n["x1"])
        y1s = sy(n["y1"])
        x2s = sx(n["x2"])
        y2s = sy(n["y2"])
        svg_parts.append(
            f'<line class="{cls}" x1="{x1s:.1f}" y1="{y1s:.1f}" '
            f'x2="{x2s:.1f}" y2="{y2s:.1f}"/>'
        )

    num_hit = visualization.get("num_hit", 0)
    num_miss = visualization.get("num_miss", 0)
    total_vis = num_hit + num_miss
    hit_rate = num_hit / total_vis if total_vis > 0 else 0

    svg_parts.append(
        f'<text x="{margin}" y="{margin + 10}" font-family="sans-serif" '
        f'font-size="12" fill="#333">'
        f'Buffon Needle Simulation: {total_vis} needles, '
        f'<tspan fill="#e74c3c">{num_hit} hits</tspan> ('
        f'{hit_rate:.2%}), '
        f'<tspan fill="#3498db">{num_miss} misses</tspan>'
        f'</text>'
    )
    svg_parts.append(
        f'<text x="{margin}" y="{height - margin + 20}" font-family="sans-serif" '
        f'font-size="10" fill="#666">'
        f'Line spacing = {line_spacing}, Needle length = {needle_length}'
        f'</text>'
    )

    svg_parts.append('</svg>')

    return "\n".join(svg_parts)


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
    max_visual_needles = int(data.get("max_visual_needles", 0))
    num_visual_lines = int(data.get("num_visual_lines", 5))
    include_svg = bool(data.get("include_svg", False))

    if num_needles <= 0:
        return jsonify({"error": "num_needles must be positive"}), 400

    if needle_length <= 0 or line_spacing <= 0:
        return jsonify({"error": "needle_length and line_spacing must be positive"}), 400

    if needle_length > line_spacing:
        return jsonify({"error": "needle_length must be <= line_spacing"}), 400

    MAX_VISUAL = 5000
    if max_visual_needles > MAX_VISUAL:
        return jsonify({"error": f"max_visual_needles must not exceed {MAX_VISUAL}"}), 400

    try:
        start = time.perf_counter()
        pi_estimate, needles, hits, visualization = simulate_buffon_needle(
            num_needles, needle_length, line_spacing, seed,
            max_visual_needles=max_visual_needles,
            num_visual_lines=num_visual_lines,
        )
        elapsed = time.perf_counter() - start

        stats = compute_statistics(needles, needle_length, line_spacing, pi_estimate, hits)
        stats["elapsed_seconds"] = elapsed
        stats["needles_per_second"] = needles / elapsed if elapsed > 0 else 0.0

        if max_visual_needles > 0 and visualization:
            stats["visualization"] = visualization
            if include_svg:
                stats["visualization"]["svg"] = generate_svg(visualization)

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
        "<h2>模拟 API</h2>"
        "<p>使用 <code>POST /api/simulate</code> 进行模拟</p>"
        "<pre>"
        "请求示例（带可视化）：<br>"
        "curl -X POST http://localhost:5000/api/simulate \\<br>"
        "  -H 'Content-Type: application/json' \\<br>"
        "  -d '{\"num_needles\": 1000000, \"max_visual_needles\": 500, \"include_svg\": true}'"
        "</pre>"
        "<h3>请求参数</h3>"
        "<ul>"
        "<li><code>num_needles</code>: 投针总次数（默认 1000000）</li>"
        "<li><code>needle_length</code>: 针长（默认 1.0）</li>"
        "<li><code>line_spacing</code>: 平行线间距（默认 1.0）</li>"
        "<li><code>seed</code>: 随机种子（可选）</li>"
        "<li><code>max_visual_needles</code>: 返回可视化的针数，最大 5000（默认 0，不返回）</li>"
        "<li><code>num_visual_lines</code>: 可视化平行线数量（默认 5）</li>"
        "<li><code>include_svg</code>: 是否同时返回 SVG 图像（默认 false）</li>"
        "</ul>"
        "<h3>可视化响应字段（visualization）</h3>"
        "<ul>"
        "<li><code>needles</code>: 针列表，每根针包含 center_x, center_y, angle_deg, x1, y1, x2, y2, is_hit</li>"
        "<li><code>line_y_positions</code>: 平行线 y 坐标</li>"
        "<li><code>viewport</code>: 视口范围（x_min, x_max, y_min, y_max）</li>"
        "<li><code>svg</code>: SVG 图像字符串（当 include_svg=true 时）</li>"
        "</ul>"
        "<h2>健康检查</h2>"
        "<p><code>GET /api/health</code></p>"
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
    sim_parser.add_argument(
        "--visualize",
        type=int,
        default=0,
        help="导出可视化数据的针数（最大 5000，默认 0 不导出）",
    )
    sim_parser.add_argument(
        "--visual-lines",
        type=int,
        default=5,
        help="可视化平行线数量（默认：5）",
    )
    sim_parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="导出可视化数据到 JSON 文件路径",
    )
    sim_parser.add_argument(
        "--export-svg",
        type=str,
        default=None,
        help="导出可视化图像到 SVG 文件路径",
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

        max_vis = 0
        if args.visualize > 0 or args.export_json or args.export_svg:
            max_vis = args.visualize if args.visualize > 0 else 500
        if max_vis > 5000:
            parser.error("可视化针数不能超过 5000")

        start = time.perf_counter()
        pi_estimate, needles, hits, visualization = simulate_buffon_needle(
            args.num_needles,
            args.needle_length,
            args.line_spacing,
            args.seed,
            args.batch_size,
            max_visual_needles=max_vis,
            num_visual_lines=args.visual_lines,
        )
        elapsed = time.perf_counter() - start

        stats = compute_statistics(
            needles, args.needle_length, args.line_spacing, pi_estimate, hits
        )

        reliability = stats["reliability"]

        print("=" * 55)
        print("布丰投针 π 近似计算结果")
        print("=" * 55)
        print(f"投针次数         : {stats['num_needles']:,}")
        print(f"针长             : {stats['needle_length']}")
        print(f"平行线间距       : {stats['line_spacing']}")
        print(f"命中次数         : {stats['total_hits']:,}")
        print(f"命中概率         : {stats['hit_probability']:.6f}")
        print(f"理论命中概率     : {stats['theoretical_hit_probability']:.6f}")
        print(f"π 估计值        : {stats['pi_estimate']:.10f}")
        print(f"π 真实值        : {stats['pi_true']:.10f}")
        print(f"绝对误差         : {stats['absolute_error']:.10f}")
        print(f"相对误差         : {stats['relative_error_percent']:.6f}%")
        print(f"可靠性等级       : {reliability['tier']}（预期精度 {reliability['tier_accuracy']}）")
        print(f"预期相对误差     : {reliability['expected_relative_error_percent']:.4f}%")
        print(f"建议最低投针次数 : {reliability['recommended_min_needles']:,}")
        print(f"耗时             : {elapsed:.4f} 秒")
        print(f"吞吐量           : {needles / elapsed:,.0f} 针/秒")

        if reliability["warnings"]:
            print("-" * 55)
            print("⚠ 警告：")
            for w in reliability["warnings"]:
                print(f"  · {w}")

        print("=" * 55)

        if visualization and visualization.get("needles"):
            num_vis = visualization["num_visual_needles"]
            print(f"\n📊 可视化数据（{num_vis} 根针）：")
            print(f"   命中 {visualization['num_hit']} 根，未命中 {visualization['num_miss']} 根")
            print(f"   视口范围: x=[{visualization['viewport']['x_min']:.2f}, {visualization['viewport']['x_max']:.2f}]")
            print(f"             y=[{visualization['viewport']['y_min']:.2f}, {visualization['viewport']['y_max']:.2f}]")

            if args.export_json:
                export_data = {
                    "statistics": stats,
                    "visualization": visualization,
                }
                if "svg" in visualization:
                    del visualization["svg"]
                with open(args.export_json, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                print(f"   ✅ JSON 数据已导出: {args.export_json}")

            if args.export_svg:
                svg_content = generate_svg(visualization)
                with open(args.export_svg, "w", encoding="utf-8") as f:
                    f.write(svg_content)
                print(f"   ✅ SVG 图像已导出: {args.export_svg}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
