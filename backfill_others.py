#!/usr/bin/env python3
"""
回填脚本：跳过 bioRxiv，只跑其他源的 30 天数据。
"""

import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def backfill_others(days: int = 30):
    """回填除 bioRxiv 外的所有源。"""
    logger.info(f"=== 回填模式（跳过 bioRxiv）：{days} 天 ===")

    import yaml
    from pathlib import Path

    from src.S1_aggregate import _dispatch_source
    from src.S2_clean import batch_clean
    from src.S4_filter import hybrid as filt
    from src.archive import archive_candidate_pool

    # 读取 sources.yaml
    sources_file = Path("src/S1_aggregate/sources.yaml")
    sources = yaml.safe_load(sources_file.read_text())

    # 过滤掉 bioRxiv 相关源，临时修改 days_back
    filtered_sources = []
    for src in sources:
        name = src.get("name", "").lower()
        src_type = src.get("type", "").lower()

        # 跳过 bioRxiv
        if "biorxiv" in name or src_type == "biorxiv_api":
            logger.info(f"  跳过: {src.get('name')}")
            continue

        # 修改时间窗口
        src_copy = src.copy()
        if "days_back" in src_copy:
            src_copy["days_back"] = days
        if "term" in src_copy and '"last 3 days"' in src_copy.get("term", ""):
            src_copy["term"] = src_copy["term"].replace(
                '"last 3 days"[dp]', f'"last {days} days"[dp]'
            )
        filtered_sources.append(src_copy)

    logger.info(f"将抓取 {len(filtered_sources)} 个源")

    # Step 1: 聚合
    all_items = []
    for src in filtered_sources:
        try:
            items = _dispatch_source(src)
            all_items.extend(items)
            logger.info(f"  [{src.get('name')}] → {len(items)} items")
        except Exception as e:
            logger.error(f"  [{src.get('name')}] 错误: {e}")

    logger.info(f"Step 1 Aggregate: {len(all_items)} items")

    if not all_items:
        logger.warning("No items fetched")
        return

    # Step 2: 清洗
    all_items = batch_clean(all_items)
    logger.info(f"Step 2 Clean: {len(all_items)} items")

    # Step 3: DOI 去重
    seen_dois = set()
    deduped = []
    for item in all_items:
        if item.doi:
            if item.doi not in seen_dois:
                seen_dois.add(item.doi)
                deduped.append(item)
        else:
            deduped.append(item)
    all_items = deduped
    logger.info(f"Step 3 Dedup: {len(all_items)} items")

    # Step 4: Layer1 过滤
    layer1_passed, layer1_dropped = filt.filter_layer1(all_items)
    logger.info(f"Step 4 Layer1: {len(layer1_passed)} passed, {len(layer1_dropped)} dropped")

    if not layer1_passed:
        logger.warning("No items passed Layer1")
        return

    # Step 5: 语义打分
    logger.info("Step 5 Scoring...")
    scored_items = filt.score_with_anchors(layer1_passed)
    logger.info(f"  Scored {len(scored_items)} items")

    # Step 6: 存入候选池
    candidates = [item for item, score in scored_items]
    archive_candidate_pool(candidates)
    logger.info(f"Step 6 Archive: {len(candidates)} candidates saved")

    logger.info("=== 完成 ===")


if __name__ == "__main__":
    backfill_others(days=30)
