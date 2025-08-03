# length_robustness_grid.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# === Configuration ===
INPUT_BASE  = "../N-gram Scoring"
OUTPUT_DIR  = "../results/aggregated/length_grid"
DATASETS    = ["pubmed", "writing", "xsum"]
VARIANTS    = ["gpt-4", "gemini", "turbo_3.5"]
TEMPS       = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
DELTA_FEATS = [
    "delta_score_2gram","delta_entropy_2gram","delta_variance_2gram",
    "delta_score_3gram","delta_entropy_3gram","delta_variance_3gram",
    "delta_score_4gram","delta_entropy_4gram","delta_variance_4gram",
    "delta_score_5gram","delta_entropy_5gram","delta_variance_5gram",
]

# styling
sns.set(style="whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "lines.linewidth": 1.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
})

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Custom y-limits per dataset to optimize vertical space
y_limits = {
    'pubmed': (0.6, 1.05),
    'writing': (0.70, 1.02),
    'xsum': (0.89, 1.01)
}

for ds in DATASETS:
    # determine bins per dataset
    sample_path = os.path.join(
        INPUT_BASE,
        f"kenlm_filtered_features_{ds}_{VARIANTS[0]}_temp{TEMPS[0]}.csv"
    )
    df0 = pd.read_csv(sample_path)
    lengths_all = df0['source_text'].str.split().apply(len)
    bins = [0] + np.percentile(lengths_all, [10,30,50,70,90]).tolist() + [lengths_all.max()+1]

    fig, axes = plt.subplots(1, len(TEMPS), figsize=(18, 4), sharey=False)
    # adjust subplot top margin to fit suptitle
    plt.subplots_adjust(top=0.80)
    # add single x-axis label at bottom center
    fig.text(0.5, 0.02, 'Average Passage Length (words)', ha='center', va='center', fontsize=12)
    for idx, temp in enumerate(TEMPS):
        ax = axes[idx]
        records = []
        for var in VARIANTS:
            path = os.path.join(
                INPUT_BASE,
                f"kenlm_filtered_features_{ds}_{var}_temp{temp}.csv"
            )
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df['length'] = df['source_text'].str.split().apply(len)
            df['label_num'] = df['label'].map({'human':0,'ai':1})
            X = df[DELTA_FEATS]
            y = df['label_num']
            X_tr, X_te, y_tr, y_te, len_tr, len_te = train_test_split(
                X, y, df['length'], test_size=0.2,
                stratify=y, random_state=42
            )
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_te)[:,1]
            df_te = pd.DataFrame({'length':len_te.values,'prob':probs,'label':y_te.values})
            df_te['bin'] = pd.cut(df_te['length'], bins=bins, right=False)

            # compute avg auc per bin
            bin_data = df_te.groupby('bin', observed=True).apply(
                lambda sub: pd.Series({
                    'avg_length': sub['length'].mean(),
                    'auc': roc_auc_score(sub['label'], sub['prob'])
                }) if len(sub)>=20 and sub['label'].nunique()>1 else pd.Series({'avg_length':np.nan,'auc':np.nan})
            ).dropna()
            bin_data['variant'] = var
            records.append(bin_data.reset_index())

        if len(records)==0:
            continue
        plot_df = pd.concat(records, ignore_index=True)
        sns.lineplot(
            data=plot_df,
            x='avg_length', y='auc',
            hue='variant', marker='o', ax=ax
        )
        ax.set_title(f"T={temp}")
        ax.set_xlabel(None)
        if idx==0:
            ax.set_ylabel("ROC-AUC")
        else:
            ax.set_ylabel(None)
        # omit individual x-axis labels; using common label
            # ax.set_xlabel("Avg Length")
        ax.set_xlim(plot_df['avg_length'].min()-5, plot_df['avg_length'].max()+5)
                # apply fixed y-limits per dataset
        y_min, y_max = y_limits[ds]
        ax.set_ylim(y_min, y_max)
        if idx==len(TEMPS)-1:
            ax.legend(title='Variant', loc='lower right')
        else:
            ax.get_legend().remove()
    plt.suptitle(f"{ds.upper()} Length Robustness Across Temperatures", y=1.02)
    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, f"{ds}_length_grid.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved grid for {ds}: {out_file}")
