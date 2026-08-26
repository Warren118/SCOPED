import pandas as pd
import numpy as np
import re
import json
import os
import math

# ================= 配置区域 =================
INPUT_FILE = 'train-00000-of-00001.parquet'
OUTPUT_FILE = 'selected_hierarchical_seeds_v3_newmatrix.json'

# 采样目标配置
TARGET_CORE = 200  # 核心职业种子数
TARGET_SAT = 100   # 验证职业种子数

# ── Merit分层采样比例配置 ──────────────────────────────────────────────────────
# 用CV字数作为Merit的零成本代理变量，将种子池三等分后按比例采样。
# Tier_A (字数 Top 33%):    40% → 强势候选人（天花板区）
# Tier_B (字数 Mid 33%):    40% → 边缘候选人（决策边界，偏见最易在此显现）
# Tier_C (字数 Bottom 33%): 20% → 弱势候选人（地板区）
MERIT_TIER_RATIOS = (0.40, 0.40, 0.20)

# 职业矩阵配置
# 变更说明：
#   替换 Customer Support (1,208 seeds) → QA Engineer    (22,239 seeds)
#   替换 Graphic Designer (1,768 seeds) → UI/UX Designer  (9,583 seeds)
#   替换 Sales Manager    (1,712 seeds) → Business Analyst (5,017 seeds)
#   保留 Project Manager  (9,955 seeds) ✅
#   保留 Java Developer   (7,198 seeds) ✅ CORE
#   保留 HR Manager       (1,550 seeds) ✅ CORE
DOMAIN_MATRIX = {
    "Hard Tech": {
        "desc": "技术研发类（客观评估端）",
        "core": "Java Developer",
        "satellites": []
    },
    "Soft Skills": {
        "desc": "人力资源类（主观评估端）",
        "core": "HR Manager",
        "satellites": []
    },
    "QA Tech": {
        "desc": "技术测试类（技术岗内部对比，22k seeds）",
        "core": "1",
        "satellites": ["QA Engineer"]
    },
    "Creativity": {
        "desc": "创意设计类（替换Graphic Designer，seed量5x）",
        "core": "1",
        "satellites": ["UI/UX Designer"]
    },
    "Leadership": {
        "desc": "项目管理类（保留，9.9k seeds）",
        "core": "1",
        "satellites": ["Project Manager"]
    },
    "Analysis": {
        "desc": "业务分析类（半技术半管理，替换Sales Manager）",
        "core": "1",
        "satellites": ["Business Analyst"]
    }
}

# 关键词映射
JOB_KEYWORDS = {
    "Java Developer": ["java developer", "java engineer"],
    "Python Developer": ["python developer", "python engineer", "django"],
    "Frontend Developer": ["frontend", "front-end", "react", "angular", "vue"],
    "QA Engineer": ["qa engineer", "quality assurance", "test engineer"],
    ".net developer": [".net", "c# developer", "asp.net"],
    "Graphic Designer": ["graphic designer", "visual designer"],
    "UI/UX Designer": ["ui/ux", "ui designer", "ux designer", "product designer"],
    "Web Designer": ["web designer"],
    "Motion Designer": ["motion designer", "3d artist"],
    "Game Designer": ["game designer", "unity developer"],
    "Account Manager": ["account manager", "key account", "client success"],
    "Sales Manager": ["sales manager", "business development"],
    "Marketing Manager": ["marketing manager", "digital marketing"],
    "SEO Specialist": ["seo specialist", "seo"],
    "Copywriter": ["copywriter", "content writer"],
    "HR Manager": ["hr manager", "human resources"],
    "Recruiter": ["recruiter", "talent acquisition", "it recruiter"],
    "Product Manager": ["product manager", "product owner"],
    "Business Analyst": ["business analyst", "data analyst"],
    "Project Manager": ["project manager", "program manager", "delivery manager"],
    "Customer Support": ["customer support"]
}


def clean_experience(exp_str):
    if pd.isna(exp_str): return 0
    exp_str = str(exp_str).lower()
    nums = re.findall(r'\d+(?:\.\d+)?', exp_str)
    if not nums: return 0
    val = float(nums[0])
    if 'month' in exp_str: val /= 12
    return round(val, 1)


def get_job_category(title, config):
    title = str(title).lower()
    for domain, details in config.items():
        core_job = details['core']
        for kw in JOB_KEYWORDS.get(core_job, []):
            if kw in title:
                return domain, core_job, "CORE"
        for sat_job in details['satellites']:
            for kw in JOB_KEYWORDS.get(sat_job, []):
                if kw in title:
                    return domain, sat_job, "SATELLITE"
    return None, None, None


def stratified_sample_by_cv_length(
    df_job: pd.DataFrame,
    target: int,
    ratios: tuple = (0.40, 0.40, 0.20),
    job_label: str = ""
) -> pd.DataFrame:
    """
    对单个job的候选池按CV字数三分位分层，再按比例采样 target 条。

    Args:
        df_job:   已过滤到某一job类别的DataFrame子集
        target:   目标抽取数量
        ratios:   (A层比例, B层比例, C层比例)
        job_label: 用于日志打印

    Returns:
        采样后的DataFrame（行数 <= target，已打乱顺序）
    """
    n = len(df_job)
    if n == 0:
        return pd.DataFrame()

    # ── Step 1: 按CV字数排序，三等分 ─────────────────────────────────────────
    df_sorted = df_job.sort_values(by='cv_length', ascending=False).reset_index(drop=True)

    cut1 = n // 3
    cut2 = 2 * n // 3

    tier_a = df_sorted.iloc[:cut1]       # Top 33%    强势
    tier_b = df_sorted.iloc[cut1:cut2]   # Mid 33%    边缘
    tier_c = df_sorted.iloc[cut2:]       # Bottom 33% 弱势

    # ── Step 2: 按比例计算各层抽取数 ─────────────────────────────────────────
    n_a = math.ceil(target * ratios[0])
    n_c = math.floor(target * ratios[2])
    n_b = target - n_a - n_c

    # 边界保护：某层不够时按实际可用数量采样
    n_a = min(n_a, len(tier_a))
    n_c = min(n_c, len(tier_c))
    n_b = min(n_b, len(tier_b))

    sampled_a = tier_a.head(n_a)   # A层直接取最长的（信息最丰富）
    sampled_b = tier_b.head(n_b)   # B层同上
    sampled_c = tier_c.head(n_c)   # C层同上

    result = pd.concat([sampled_a, sampled_b, sampled_c], ignore_index=True)

    # 打乱，避免后续 inject_attributes_batch 的 seed_index
    # 与 merit_tier 产生系统性相关（前N个全是强势种子）
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)

    actual = len(result)
    print(f"      [{job_label}] Pool={n} | Target={target} | "
          f"A={n_a}(top33%) B={n_b}(mid33%) C={n_c}(bot33%) | "
          f"Total sampled={actual}"
          + (" ⚠️ short" if actual < target else " ✅"))

    return result


def main():
    print(f"Loading source file: {INPUT_FILE}...")
    try:
        df = pd.read_parquet(INPUT_FILE)
    except FileNotFoundError:
        print("Error: Source file not found.")
        return

    print(f"Total raw records: {len(df)}")

    # ── 预处理 ────────────────────────────────────────────────────────────────
    df['calc_exp'] = df.apply(
        lambda row: clean_experience(
            str(row.get('Experience Years', '')) + str(row.get('Experience', ''))
        ), axis=1
    )
    df['cv_length'] = df['CV'].str.len().fillna(0)

    # ── 初步过滤（保持原逻辑） ─────────────────────────────────────────────────
    df = df[
        (df['calc_exp'] >= 1.0) &
        (df['cv_length'] > 50)
    ]
    print(f"After basic filtering: {len(df)} records")

    # ── 打上job分类标签 ───────────────────────────────────────────────────────
    print("\nClassifying jobs...")
    classified = df['Position'].apply(lambda p: get_job_category(p, DOMAIN_MATRIX))
    df['assigned_domain'] = classified.apply(lambda x: x[0])
    df['assigned_job']    = classified.apply(lambda x: x[1])
    df['job_tier']        = classified.apply(lambda x: x[2])

    df = df[df['assigned_domain'].notna()].copy()
    print(f"After job classification: {len(df)} records across {df['assigned_job'].nunique()} roles")

    # ── 分层采样（核心改动） ───────────────────────────────────────────────────
    print(f"\nStarting stratified merit sampling "
          f"(ratios A/B/C = {MERIT_TIER_RATIOS[0]:.0%}/{MERIT_TIER_RATIOS[1]:.0%}/{MERIT_TIER_RATIOS[2]:.0%})...")

    selected_frames = []

    for job_role in df['assigned_job'].unique():
        df_job = df[df['assigned_job'] == job_role].copy()
        tier   = df_job['job_tier'].iloc[0]
        target = TARGET_CORE if tier == "CORE" else TARGET_SAT

        sampled = stratified_sample_by_cv_length(
            df_job, target, MERIT_TIER_RATIOS, job_label=job_role
        )
        if not sampled.empty:
            selected_frames.append(sampled)

    if not selected_frames:
        print("No seeds selected.")
        return

    final_df = pd.concat(selected_frames, ignore_index=True)

    # ── Merit Tier标注（Junior/Mid/Senior，按经验年限） ────────────────────────
    def assign_merit_tier(exp):
        if exp < 3:   return "Junior"
        elif exp < 6: return "Mid"
        else:         return "Senior"

    final_df['merit_tier'] = final_df['calc_exp'].apply(assign_merit_tier)

    # ── 打印采样报告 ──────────────────────────────────────────────────────────
    print("\n=== Sampling Report ===")
    print(f"{'Role':<22} {'Tier':<10} {'Count':<7} {'Target':<7} {'A(top)':<8} {'B(mid)':<8} {'C(bot)':<8}")
    print("-" * 73)

    for job_role in final_df['assigned_job'].unique():
        sub   = final_df[final_df['assigned_job'] == job_role]
        tier  = sub['job_tier'].iloc[0]
        target = TARGET_CORE if tier == "CORE" else TARGET_SAT
        count  = len(sub)

        # 重新计算实际各层分布（用于报告）
        sorted_sub = sub.sort_values('cv_length', ascending=False).reset_index(drop=True)
        n = len(sorted_sub)
        cut1, cut2 = n // 3, 2 * n // 3
        n_a = len(sorted_sub.iloc[:cut1])
        n_b = len(sorted_sub.iloc[cut1:cut2])
        n_c = len(sorted_sub.iloc[cut2:])

        status = "✅" if count >= target else "⚠️"
        print(f"{job_role:<22} {tier:<10} {count:<7} {target:<7} {n_a:<8} {n_b:<8} {n_c:<8} {status}")

    print(f"\nTotal seeds selected: {len(final_df)}")

    # ── 序列化输出 ────────────────────────────────────────────────────────────
    records = []
    for _, row in final_df.iterrows():
        records.append({
            "id":                str(row.get('id', np.random.randint(int(10e6)))),
            "original_position": row.get('Position'),
            "cv_content":        row.get('CV'),
            "assigned_domain":   row['assigned_domain'],
            "assigned_job":      row['assigned_job'],
            "job_tier":          row['job_tier'],
            "merit_tier":        row['merit_tier'],
            "merit_exp_years":   row['calc_exp'],
            "merit_skills":      row.get('Primary Keyword', 'General'),
            # 新增字段：记录该种子的Merit层级，便于后续子组分析
            "merit_cv_length":   int(row['cv_length']),
        })

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} seeds to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
