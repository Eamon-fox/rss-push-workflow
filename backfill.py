#!/usr/bin/env python3
"""
回填脚本：用较长时间窗口抓取历史数据，初始化候选池。

用法：
    python backfill.py --days 30  # 回填 30 天
    python backfill.py --days 90  # 回填 90 天
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def backfill(days: int = 30, skip_fetch: bool = False):
    """
    回填历史数据到候选池。

    Args:
        days: 回填天数
        skip_fetch: 是否跳过抓取（使用现有缓存）
    """
    logger.info(f"=== 回填模式：{days} 天 ===")

    # 1. 临时修改配置，使用更长的时间窗口
    import yaml
    sources_file = Path("src/S1_aggregate/sources.yaml")
    original_content = sources_file.read_text()

    if not skip_fetch:
        # 替换 days_back: 3 为 days_back: {days}
        modified_content = original_content.replace("days_back: 3", f"days_back: {days}")
        # 替换 PubMed 的时间窗口
        modified_content = modified_content.replace(
            '"last 3 days"[dp]', f'"last {days} days"[dp]'
        )
        sources_file.write_text(modified_content)
        logger.info(f"临时修改 sources.yaml: days_back → {days}")

    try:
        # 2. 清除旧缓存，强制重新抓取
        if not skip_fetch:
            raw_dir = Path("data/raw")
            if raw_dir.exists():
                import shutil
                for d in raw_dir.iterdir():
                    if d.is_dir():
                        shutil.rmtree(d)
                        logger.info(f"清除缓存: {d}")

        # 3. 运行 pipeline（只到 Layer1，保存候选池）
        logger.info("开始抓取...")

        from src.S1_aggregate import fetch_all
        from src.S2_clean import batch_clean
        from src.S4_filter import hybrid as filt
        from src.archive import archive_candidate_pool

        # Step 1: 聚合
        items = fetch_all()
        logger.info(f"Step 1 Aggregate: {len(items)} items")

        if not items:
            logger.warning("No items fetched")
            return

        # Step 2: 清洗
        items = batch_clean(items)
        logger.info(f"Step 2 Clean: {len(items)} items")

        # Step 3: 去重（不使用 seen，因为是回填）
        # 只做 DOI 去重，不标记 seen
        seen_dois = set()
        deduped = []
        for item in items:
            if item.doi:
                if item.doi not in seen_dois:
                    seen_dois.add(item.doi)
                    deduped.append(item)
            else:
                deduped.append(item)
        items = deduped
        logger.info(f"Step 3 Dedup (DOI only): {len(items)} items")

        # Step 4: Layer1 过滤
        layer1_passed, layer1_dropped = filt.filter_layer1(items)
        logger.info(f"Step 4 Layer1: {len(layer1_passed)} passed, {len(layer1_dropped)} dropped")

        if not layer1_passed:
            logger.warning("No items passed Layer1")
            return

        # Step 5: 计算语义分数（会缓存 embedding）
        logger.info("计算语义分数（embedding 会被缓存）...")
        scored_items = filt.score_with_anchors(layer1_passed)
        logger.info(f"Step 5 Scoring: {len(scored_items)} items scored")

        # Step 6: 保存到候选池（增量追加）
        candidates = [item for item, score in scored_items]
        archive_candidate_pool(candidates)
        logger.info(f"Step 6 Archive: {len(candidates)} candidates saved")

        # 统计
        logger.info("=== 回填完成 ===")
        logger.info(f"候选池新增: {len(candidates)} 篇")

    finally:
        # 4. 恢复原始配置
        if not skip_fetch:
            sources_file.write_text(original_content)
            logger.info("已恢复 sources.yaml")


def main():
    parser = argparse.ArgumentParser(description="回填历史数据到候选池")
    parser.add_argument(
        "--days", type=int, default=30,
        help="回填天数 (default: 30)"
    )
    parser.add_argument(
        "--skip-fetch", action="store_true",
        help="跳过抓取，使用现有缓存"
    )
    args = parser.parse_args()

    backfill(days=args.days, skip_fetch=args.skip_fetch)


if __name__ == "__main__":
    main()
