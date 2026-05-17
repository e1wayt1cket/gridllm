"""
Phase 1: 京津冀高速公路收费数据清洗
======================================
输入: export/ 下原始 CSV (GBK 编码)
输出: data/cleaned/ 下 Parquet 文件
筛选: 仅保留京津冀地区收费站
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 0. 配置
# ============================================================
RAW_DIR = Path("/mnt/c/Python/data/export/export")
OUT_DIR = Path("/mnt/c/Python/MakerB/highway_planning/data/cleaned")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 只处理前3个月小样
SAMPLE_MONTHS = ["202303"]  # 续跑，前2月已完成

# 京津冀城市关键词（收费站名匹配）
JJJ_KEYWORDS = [
    # 北京
    "北京", "京", "首都", "通州", "大兴", "房山", "密云", "延庆",
    "顺义", "昌平", "平谷", "怀柔", "朝阳", "海淀", "丰台", "石景山",
    "门头沟", "亦庄", "清河", "白鹿", "大羊坊", "京津", "京沪",
    "京哈", "京港澳", "京藏", "京承", "京平", "京开", "京通",
    # 天津
    "天津", "津", "滨海", "塘沽", "大港", "东丽", "北辰", "西青",
    "津南", "武清", "宝坻", "蓟州", "静海", "宁河", "汉沽",
    "临港", "空港", "小孙庄", "金钟路", "永定新河", "军粮城",
    "杨村", "杨柳青", "宜兴埠", "咸水沽", "新立", "华明",
    "前毕庄", "盛唐路", "金桥", "中心桥", "上仓", "芦台",
    "荣乌", "津滨", "津蓟", "津港", "津静", "津保",
    "津文", "津榆", "未来科技城",
    # 河北
    "河北", "石家庄", "唐山", "秦皇岛", "邯郸", "邢台",
    "保定", "张家口", "承德", "沧州", "廊坊", "衡水",
    "雄安", "定州", "辛集", "涿州", "三河", "霸州",
    "迁安", "遵化", "武安", "南宫", "沙河", "涞水",
    "正定", "井陉", "香河", "大厂", "永清", "固安",
    "文安", "大城",
]

# 编译正则（忽略大小写）
JJJ_PATTERN = re.compile("|".join(JJJ_KEYWORDS))


def is_jjj_station(name: str) -> bool:
    """判断收费站名是否属于京津冀"""
    if pd.isna(name) or not isinstance(name, str):
        return False
    return bool(JJJ_PATTERN.search(name))


def clean_dataframe(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """
    清洗单个DataFrame：
    - 编码修复
    - 字段校验
    - 异常值剔除
    - 派生字段
    """
    n_before = len(df)

    # --- Step 1: 列名标准化 ---
    df.columns = [c.strip().upper() for c in df.columns]

    # --- Step 2: 剔除空关键字段 ---
    required = ["VEHICLEID", "ENSTATIONID", "EXSTATIONID", "ENTIME", "EXTIME"]
    for col in required:
        if col in df.columns:
            df = df[df[col].notna() & (df[col] != "")]
    df = df[df["ENSTATIONID"] != df["EXSTATIONID"]]  # 进出站相同的无效行程

    # --- Step 3: 车型校验 ---
    if "VEHICLETYPE" in df.columns:
        df["VEHICLETYPE"] = pd.to_numeric(df["VEHICLETYPE"], errors="coerce")
        df = df[df["VEHICLETYPE"].between(1, 26)]

    # --- Step 4: 时间解析 ---
    df["ENTIME"] = pd.to_datetime(df["ENTIME"], errors="coerce")
    df["EXTIME"] = pd.to_datetime(df["EXTIME"], errors="coerce")
    df = df.dropna(subset=["ENTIME", "EXTIME"])

    # 行程时长合法性
    mask_valid_time = (df["EXTIME"] > df["ENTIME"]) & \
                      ((df["EXTIME"] - df["ENTIME"]).dt.total_seconds() < 86400)  # <24h
    df = df[mask_valid_time]

    # --- Step 5: 京津冀筛选 ---
    mask_jjj = df["ENSTATIONNAME"].apply(is_jjj_station) | \
               df["EXSTATIONNAME"].apply(is_jjj_station)
    df = df[mask_jjj]

    # --- Step 6: 派生字段 ---
    df["TRIP_DURATION_MIN"] = (df["EXTIME"] - df["ENTIME"]).dt.total_seconds() / 60
    df["DATE"] = df["ENTIME"].dt.date
    df["HOUR"] = df["ENTIME"].dt.hour
    df["MONTH"] = df["ENTIME"].dt.to_period("M").astype(str)
    df["DAY_OF_WEEK"] = df["ENTIME"].dt.dayofweek

    # 车型分类
    def categorize(vtype):
        if vtype <= 4:
            return "客车"
        elif vtype <= 16:
            return "货车"
        else:
            return "专项作业车"
    df["VEHICLE_CATEGORY"] = df["VEHICLETYPE"].apply(categorize)

    # 是否为小客车（客车1-2型，PDF4关注对象）
    df["IS_PASSENGER_CAR"] = df["VEHICLETYPE"].isin([1, 2])

    n_after = len(df)
    n_dropped = n_before - n_after

    return df, {
        "file": source_file,
        "before": n_before,
        "after": n_after,
        "dropped": n_dropped,
        "keep_pct": round(n_after / n_before * 100, 2) if n_before > 0 else 0,
    }


def process_month(month: str) -> dict:
    """处理单月数据"""
    month_dir = RAW_DIR / month
    csv_files = sorted([f for f in month_dir.glob("*.csv")
                        if f.name not in ["01 - 副本.csv"]])  # 跳过副本

    all_stats = []
    monthly_chunks = []

    print(f"\n{'='*60}")
    print(f"处理 {month}: {len(csv_files)} 个文件")
    print(f"{'='*60}")

    for i, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file, encoding="gbk", low_memory=False)
            df_clean, stats = clean_dataframe(df, csv_file.name)
            all_stats.append(stats)

            if len(df_clean) > 0:
                monthly_chunks.append(df_clean)

            if (i + 1) % 5 == 0 or i == len(csv_files) - 1:
                print(f"  [{i+1}/{len(csv_files)}] "
                      f"保留率 {stats['keep_pct']:.1f}% "
                      f"({stats['after']:,}/{stats['before']:,})")

        except Exception as e:
            print(f"  ✗ {csv_file.name}: {e}")

    # 合并并保存为 Parquet
    if monthly_chunks:
        df_month = pd.concat(monthly_chunks, ignore_index=True)
        out_path = OUT_DIR / f"{month}_jjj_cleaned.parquet"
        df_month.to_parquet(out_path, compression="snappy", index=False)

        file_mb = out_path.stat().st_size / 1024 / 1024
        print(f"\n  ✓ 保存: {out_path} ({file_mb:.1f} MB, {len(df_month):,} 行)")

    return all_stats


def main():
    print("=" * 60)
    print("Phase 1: 京津冀高速公路数据清洗")
    print(f"数据源: {RAW_DIR}")
    print(f"输出:   {OUT_DIR}")
    print(f"月份:   {SAMPLE_MONTHS}")
    print("=" * 60)

    all_stats = []
    for month in SAMPLE_MONTHS:
        stats = process_month(month)
        all_stats.extend(stats)

    # --- 汇总报告 ---
    print(f"\n{'='*60}")
    print("汇总报告")
    print(f"{'='*60}")

    df_stats = pd.DataFrame(all_stats)
    total_before = df_stats["before"].sum()
    total_after = df_stats["after"].sum()
    total_dropped = df_stats["dropped"].sum()

    print(f"  总记录数:     {total_before:>12,}")
    print(f"  清洗后保留:   {total_after:>12,}  ({total_after/total_before*100:.1f}%)")
    print(f"  剔除:         {total_dropped:>12,}")
    print(f"  日均记录:     {total_after/len(all_stats):>12,.0f}")

    # 按丢弃原因细分
    print(f"\n  数据文件数:   {len(all_stats)}")
    print(f"  输出目录:     {OUT_DIR}")

    df_stats.to_csv(OUT_DIR / "cleaning_stats.csv", index=False, encoding="utf-8-sig")
    print(f"  详细统计:     {OUT_DIR / 'cleaning_stats.csv'}")


if __name__ == "__main__":
    main()
