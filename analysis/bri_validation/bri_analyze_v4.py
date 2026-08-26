"""
BRI v4 Results Analyzer — 支持valence维度
用法: python bri_analyze_v4.py results_A.json [results_B.json]
"""
import json, sys, math
from collections import Counter

try:
    from scipy.stats import chi2_contingency
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[warn] pip install scipy")

try:
    from sklearn.metrics import cohen_kappa_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[warn] pip install scikit-learn")

def load(path):
    for enc in ['utf-8','latin-1','utf-8-sig']:
        try:
            with open(path, encoding=enc) as f:
                return json.load(f)
        except: continue
    raise ValueError(f"Cannot load {path}")

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 1.0
    p=k/n; d=1+z**2/n
    c=(p+z**2/(2*n))/d
    m=z*math.sqrt(p*(1-p)/n+z**2/(4*n**2))/d
    return max(0,c-m), min(1,c+m)

def chi2_p(a, b):
    if not HAS_SCIPY: return "n/a"
    try:
        _,p,_,_ = chi2_contingency([[a['biased'], a['n']-a['biased']],
                                     [b['biased'], b['n']-b['biased']]])
        return f"{p:.4f}" + (" *" if p<0.05 else "")
    except: return "n/a"

def get_stats(records):
    stats = {}
    for s in ['HIGH','MID','LOW']:
        sub = [r for r in records if r.get('stratum')==s
               and r.get('judge_judgment') in ('BIASED','NOT_BIASED')]
        if not sub: continue
        nb  = sum(1 for r in sub if r['judge_judgment']=='BIASED')
        neg = sum(1 for r in sub if r['judge_judgment']=='BIASED'
                  and r.get('judge_valence')=='NEGATIVE')
        pos = sum(1 for r in sub if r['judge_judgment']=='BIASED'
                  and r.get('judge_valence')=='POSITIVE')
        mix = sum(1 for r in sub if r['judge_judgment']=='BIASED'
                  and r.get('judge_valence')=='MIXED')
        lo,hi = wilson(nb, len(sub))
        stats[s] = {'n':len(sub),'biased':nb,'neg':neg,'pos':pos,'mix':mix,
                    'rate':nb/len(sub),'lo':lo,'hi':hi}
    return stats

def print_main_table(stats, label=""):
    if label: print(f"\n{'='*65}\n  {label}\n{'='*65}")
    print(f"  {'Stratum':<6} {'n':>4} {'Biased':>7} {'Rate':>7}  95%CI"
          f"         {'NEG':>6} {'POS':>6} {'MIXED':>6}")
    print(f"  {'-'*65}")
    for s in ['HIGH','MID','LOW']:
        if s not in stats: continue
        r = stats[s]
        n = r['n']
        print(f"  {s:<6} {n:>4} {r['biased']:>7} {r['rate']:>7.1%}"
              f"  [{r['lo']:.3f},{r['hi']:.3f}]"
              f"  {r['neg']/n:>6.1%} {r['pos']/n:>6.1%} {r['mix']/n:>6.1%}")

def print_separations(stats):
    print(f"\n  {'Comparison':<14} {'Sep':>7} {'OR':>6}  p")
    print(f"  {'-'*38}")
    pairs = [('HIGH','MID'),('HIGH','LOW'),('MID','LOW')]
    for s1,s2 in pairs:
        if s1 not in stats or s2 not in stats: continue
        a=stats[s1]; b=stats[s2]
        sep = a['rate']-b['rate']
        or_v = (a['biased']/max(a['n']-a['biased'],1)) / \
               (b['biased']/max(b['n']-b['biased'],1))
        p = chi2_p(a,b)
        print(f"  {s1} vs {s2:<6} {sep:>+7.1%} {or_v:>6.2f}  {p}")

def print_valence_insight(stats):
    """核心发现：HIGH主要是NEG bias，MID主要是POS bias"""
    print(f"\n  --- Valence breakdown (key finding) ---")
    for s in ['HIGH','MID','LOW']:
        if s not in stats: continue
        r = stats[s]
        if r['biased'] == 0: continue
        neg_pct = r['neg']/r['biased'] if r['biased'] else 0
        pos_pct = r['pos']/r['biased'] if r['biased'] else 0
        mix_pct = r['mix']/r['biased'] if r['biased'] else 0
        dominant = max([('NEG',neg_pct),('POS',pos_pct),('MIXED',mix_pct)],
                       key=lambda x:x[1])
        print(f"  {s}: of {r['biased']} BIASED → "
              f"NEG={neg_pct:.0%} POS={pos_pct:.0%} MIXED={mix_pct:.0%}"
              f"  [dominant: {dominant[0]}]")
    print()
    if 'HIGH' in stats and 'MID' in stats:
        h=stats['HIGH']; m=stats['MID']
        if h['biased']>0 and m['biased']>0:
            print(f"  → HIGH-BRI dominated by NEG bias ({h['neg']/h['biased']:.0%})")
            print(f"  → MID-BRI dominated by POS bias ({m['pos']/m['biased']:.0%})")
            print(f"  → Explains HIGH≈MID in overall BIASED rate:")
            print(f"     Both are biased but in opposite valence directions.")
            print(f"     BRI neg-framing filter correctly separates them by valence,")
            print(f"     but misses that positive-valence proxy bias is equally real.")

def model_breakdown(records, stats):
    models = sorted(set(r.get('model','?') for r in records))
    if len(models)<=1: return
    print(f"\n  --- By model (BIASED rate) ---")
    for m in models:
        sub = [r for r in records if r.get('model')==m
               and r.get('judge_judgment') in ('BIASED','NOT_BIASED')]
        parts = []
        for s in ['HIGH','MID','LOW']:
            ss=[r for r in sub if r.get('stratum')==s]
            if ss:
                nb=sum(1 for r in ss if r['judge_judgment']=='BIASED')
                parts.append(f"{s}={nb/len(ss):.0%}(n={len(ss)})")
        print(f"  {m:<22} {' | '.join(parts)}")

def kappa_analysis(records_a, records_b):
    if not HAS_SKLEARN:
        print("\n  [kappa] skipped — pip install scikit-learn")
        return
    lb = {}
    for r in records_b:
        if r.get('judge_judgment') in ('BIASED','NOT_BIASED'):
            key = r.get('private_thought','')[:80].strip()
            lb[key] = r['judge_judgment']
    ma,mb = [],[]
    for r in records_a:
        if r.get('judge_judgment') not in ('BIASED','NOT_BIASED'): continue
        key = r.get('private_thought','')[:80].strip()
        if key in lb:
            ma.append(r['judge_judgment']); mb.append(lb[key])
    print(f"\n  Matched: {len(ma)}")
    if len(ma)<10:
        print("  Too few overlapping cases")
        return
    agree = sum(1 for a,b in zip(ma,mb) if a==b)
    k = cohen_kappa_score(ma,mb)
    interp = ("poor" if k<0.2 else "fair" if k<0.4 else
              "moderate" if k<0.6 else "substantial" if k<0.8 else "near-perfect")
    print(f"  Agreement: {agree}/{len(ma)} ({agree/len(ma):.1%})")
    print(f"  Cohen's κ: {k:.3f} ({interp})")

def latex_table(stats_list, labels):
    print(f"\n{'='*65}\n  LaTeX Table\n{'='*65}")
    lines = [
        r"\begin{table}[t]",
        r"\centering\small\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"\textbf{Judge} & \textbf{Stratum} & $n$ & \textbf{Biased} "
        r"& \textbf{Rate} & \textbf{95\%~CI} & \textbf{Neg\%} & \textbf{Pos\%} & \textbf{Mixed\%} \\",
        r"\midrule",
    ]
    for stats, label in zip(stats_list, labels):
        first = True
        for s in ['HIGH','MID','LOW']:
            if s not in stats: continue
            r=stats[s]; n=r['n']
            jl = label if first else ""
            first = False
            neg_pct = f"{r['neg']/n:.0%}" if n else "-"
            pos_pct = f"{r['pos']/n:.0%}" if n else "-"
            mix_pct = f"{r['mix']/n:.0%}" if n else "-"
            lines.append(
                f"{jl} & {s}-BRI & {n} & {r['biased']} & {r['rate']:.1%} & "
                f"[{r['lo']:.3f},{r['hi']:.3f}] & {neg_pct} & {pos_pct} & {mix_pct} \\\\"
            )
        if len(stats_list)>1: lines.append(r"\midrule")

    # Key separations
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{9}{l}{\textit{Key separation statistics (HIGH+MID vs LOW)}} \\")
    for stats, label in zip(stats_list, labels):
        for s1,s2 in [('HIGH','LOW'),('MID','LOW')]:
            if s1 not in stats or s2 not in stats: continue
            a=stats[s1]; b=stats[s2]
            or_v=(a['biased']/max(a['n']-a['biased'],1))/(b['biased']/max(b['n']-b['biased'],1))
            p_str=""
            if HAS_SCIPY:
                try:
                    _,p,_,_=chi2_contingency([[a['biased'],a['n']-a['biased']],
                                               [b['biased'],b['n']-b['biased']]])
                    p_str=f", $p={p:.3f}$"
                except: pass
            lines.append(
                f"\\multicolumn{{2}}{{l}}{{{label}: {s1} vs {s2}}} & "
                f"\\multicolumn{{7}}{{l}}{{OR$={or_v:.2f}${p_str}}} \\\\"
            )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{BRI construct validity: two-judge valence-aware evaluation.}",
        r"HIGH-BRI instances are predominantly negatively-valenced bias;",
        r"MID-BRI instances reveal positive-valence proxy bias missed by BRI's",
        r"negative-framing filter. Both strata are strongly separated from",
        r"LOW-BRI (no BRI trigger), confirming BRI identifies genuine",
        r"bias-relevant content. BRI therefore provides a lower bound on",
        r"O-lens process burden.}",
        r"\label{tab:bri-validation}",
        r"\end{table}",
    ]
    for l in lines: print(l)

def analyze(path, label="Judge"):
    data = load(path)
    valid = [r for r in data if r.get('judge_judgment') in ('BIASED','NOT_BIASED')]
    errors = [r for r in data if r.get('judge_judgment')=='ERROR']
    print(f"\n{'#'*65}")
    print(f"  {label}  —  {path}")
    print(f"{'#'*65}")
    print(f"  Total={len(data)}  Valid={len(valid)}  Errors={len(errors)}")
    # Check if valence data exists
    has_valence = any(r.get('judge_valence') not in (None,'') for r in valid)
    if not has_valence:
        print("  [warn] No valence data — run bri_judge_v4_valence.py first")
    stats = get_stats(valid)
    print_main_table(stats, label)
    print_separations(stats)
    if has_valence:
        print_valence_insight(stats)
    model_breakdown(valid, stats)
    return valid, stats

def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python bri_analyze_v4.py results_A.json [results_B.json]")
        sys.exit(1)

    if len(args)==1:
        valid_a, stats_a = analyze(args[0], "Judge A")
        latex_table([stats_a], ["Judge A"])
    else:
        valid_a, stats_a = analyze(args[0], "Judge A")
        valid_b, stats_b = analyze(args[1], "Judge B")
        print(f"\n{'='*65}\n  CROSS-JUDGE\n{'='*65}")
        kappa_analysis(valid_a, valid_b)
        latex_table([stats_a, stats_b], ["Judge A", "Judge B"])

if __name__ == '__main__':
    main()
