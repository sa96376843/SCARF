#!/usr/bin/env python3
"""Build six continuous, grouped manuscript tables directly from saved CSV results."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT/'tables'
FIG = ROOT/'figures'
DASH = '--'


def value(x, digits=4):
    if pd.isna(x):
        return DASH
    if isinstance(x, str):
        return x
    if isinstance(x, int):
        return str(x)
    return f'{x:.{digits}f}'


def row(cells):
    return ' & '.join(str(x) for x in cells) + r' \\'


def group(title, ncols):
    return (r'\midrule' + '\n' +
            rf'\multicolumn{{{ncols}}}{{l}}{{\textbf{{{title}}}}} \\')


def table(headers, rows, ncols, colsep=3, spec=None):
    spec = spec or ('l' + 'r'*(ncols-1))
    lines = [rf'\setlength{{\tabcolsep}}{{{colsep}pt}}',
             rf'\begin{{tabular}}{{{spec}}}', r'\toprule', row(headers)]
    lines.extend(rows)
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    return '\n'.join(lines) + '\n'


def write(name, content):
    (TAB/name).write_text(content)


def protocol_population():
    protocol = pd.read_csv(TAB/'protocol.csv')
    months = pd.read_csv(TAB/'dataset_months.csv')
    rows = [group('Frozen temporal protocol', 5)]
    for _, z in protocol.iterrows():
        protocol_value = str(z['Value']).replace('%', r'\%')
        rows.append(row([z['Protocol element'], protocol_value, DASH, DASH, DASH]))
    rows.append(group('Monthly BODMAS evaluation population', 5))
    for _, z in months.iterrows():
        rows.append(row([z['Month'], int(z['Total']), int(z['Malware']),
                         int(z['Benign']), value(z['Malware rate'], 3)]))
    write('protocol_population.tex', table(
        ['Item / month', 'Value / total', 'Malware', 'Benign', 'Malware rate'],
        rows, 5, colsep=4, spec='llrrr'))


def core_performance():
    main = pd.read_csv(TAB/'main_results.csv')
    budget = pd.read_csv(TAB/'budget_results.csv')
    ablation = pd.read_csv(TAB/'ablation_results.csv')
    family = pd.read_csv(TAB/'family_results.csv')
    rows = [group('Principal configuration: seed 42, 100 labels/month', 9)]
    for _, z in main.iterrows():
        rows.append(row([z['Method'], value(z['AUC']), value(z['TPR@1%FPR']),
                         value(z['F1']), value(z['TPR']), value(z['FPR']),
                         value(z['Brier']), value(z['ECE']), DASH]))
    rows.append(group('Label-budget sensitivity: monthly macro averages', 9))
    for _, z in budget.iterrows():
        rows.append(row([f"{z['Method']} ($B={int(z['Labels/month'])}$)",
                         value(z['AUC']), value(z['TPR@1%FPR']), value(z['F1']),
                         DASH, value(z['FPR']), value(z['Brier']), DASH, DASH]))
    rows.append(group('Component ablation: seed 42, 100 labels/month', 9))
    for _, z in ablation.iterrows():
        rows.append(row([z['Variant'], value(z['AUC']), value(z['TPR@1%FPR']),
                         value(z['F1']), DASH, value(z['FPR']), value(z['Brier']),
                         value(z['ECE']), DASH]))
    rows.append(group('Family retention and adaptation', 9))
    for _, z in family.iterrows():
        family_value = f"{value(z['Known-family TPR'])} / {value(z['Novel-family TPR'])}"
        rows.append(row([z['Method'], DASH, DASH, DASH, DASH, DASH, DASH,
                         DASH, family_value]))
    write('core_performance.tex', table(
        ['Method / setting', 'AUC', r'TPR@1\%FPR', 'F1', 'TPR', 'FPR',
         'Brier', 'ECE', 'K/N TPR'], rows, 9, colsep=3))


def robustness_statistics():
    multi = pd.read_csv(TAB/'multiseed_results.csv')
    locked = pd.read_csv(TAB/'locked_results.csv')
    bootstrap = pd.read_csv(TAB/'bootstrap_tests.csv')
    late_bootstrap = pd.read_csv(TAB/'locked_bootstrap.csv')
    metrics = ['AUC', 'TPR@1%FPR', 'F1', 'FPR', 'Brier']
    rows = [group(r'Three-seed full-horizon performance: mean $\pm$ sample SD', 10)]
    for _, z in multi.iterrows():
        rows.append(row([z['Method']] + [z[m] for m in metrics] + [DASH]*4))
    rows.append(group('Locked late-month performance: April--September 2020', 10))
    for _, z in locked.iterrows():
        rows.append(row([z['Method']] + [value(z[m]) for m in metrics] + [DASH]*4))
    for title, frame in [
        ('Full-horizon month-clustered paired bootstrap', bootstrap),
        ('Locked late-month month-clustered paired bootstrap', late_bootstrap)
    ]:
        rows.append(group(title, 10))
        for _, z in frame.iterrows():
            ci = f"[{value(z['CI low'])}, {value(z['CI high'])}]"
            rows.append(row([z['Comparison']] + [DASH]*5 +
                            [int(z['$n$']), value(z['Mean difference']), ci,
                             z['$p$-value']]))
    write('robustness_statistics.tex', table(
        ['Method / comparison', 'AUC', r'TPR@1\%FPR', 'F1', 'FPR', 'Brier',
         '$n$', r'$\Delta$', r'95\% CI', '$p$'], rows, 10, colsep=2))


def cost_sensitivity():
    efficiency = pd.read_csv(TAB/'extended_efficiency.csv')
    sensitivity = pd.read_csv(TAB/'sensitivity_results.csv')
    rows = [group('Measured serial computational cost', 10)]
    for _, z in efficiency.iterrows():
        rows.append(row([z['Method'], DASH, value(z['Update (s/month)'], 3),
                         value(z['Scoring (ms/1k)'], 3),
                         value(z['Relative update cost'], 3), DASH,
                         value(z['TPR@1%FPR'], 3), DASH, DASH, DASH]))
    rows.append(group('One-factor-at-a-time sensitivity: seed 42, 100 labels/month', 10))
    for _, z in sensitivity.iterrows():
        rows.append(row([z['Parameter'], f"{z['Value']:g}", DASH, DASH, DASH,
                         value(z['AUC']), value(z['TPR@1%FPR']), value(z['F1']),
                         value(z['FPR']), value(z['Brier'])]))
    write('cost_sensitivity.tex', table(
        ['Method / parameter', 'Value', 'Update', 'Scoring', 'Rel.', 'AUC',
         r'TPR@1\%FPR', 'F1', 'FPR', 'Brier'], rows, 10, colsep=2.5))


def main():
    protocol_population()
    core_performance()
    robustness_statistics()
    cost_sensitivity()
    superseded = ['protocol.tex','dataset_months.tex','main_results.tex','budget_results.tex',
        'ablation_results.tex','family_results.tex','multiseed_results.tex','locked_results.tex',
        'bootstrap_tests.tex','locked_bootstrap.tex','extended_efficiency.tex','sensitivity_results.tex']
    for filename in superseded:
        (TAB/filename).unlink(missing_ok=True)
    for suffix in ['pdf','png']:
        (FIG/f'fig09_ablation_parallel.{suffix}').unlink(missing_ok=True)
    print('Continuous grouped manuscript tables:', len(list(TAB.glob('*.tex'))))


if __name__ == '__main__':
    main()
