import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

CSV_FILE = "llm_scores.csv"
PNG_FILE = "llm_eval_chart.png"
COLOR_PALETTE = "tab20"

df = pd.read_csv(CSV_FILE)

model_names = df["Model"].values
categories = df.columns[1:]
scores = df.iloc[:, 1:].values

x = np.arange(len(categories))
num_models = len(model_names)
bar_width = 0.8 / num_models
offsets = np.linspace(-bar_width * (num_models - 1) / 2, bar_width * (num_models - 1) / 2, num_models)

cmap = plt.get_cmap(COLOR_PALETTE)
colors = [cmap(i % cmap.N) for i in range(len(model_names))]

plt.style.use('seaborn-darkgrid')

plt.figure(figsize=(14, 8))
for i, (model, color) in enumerate(zip(model_names, colors)):
    plt.bar(x + offsets[i], scores[i], width=bar_width, label=model, color=color)

plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=5,
    fontsize="small",
    frameon=False
)

plt.xticks(x, categories)
plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.title("LLM Evaluation on Japanese Tasks")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig(PNG_FILE, dpi=300, bbox_inches="tight")

plt.show()
