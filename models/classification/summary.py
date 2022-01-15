import collections
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


dicts = []
for cat in "coarse fine pol asp reb_coarse reb_fine".split():
    try:
      results = pd.read_csv("results/test_" + cat + ".csv")
    except FileNotFoundError:
      continue

    y_true = results["target"]
    y_pred = results["output"]
    macro = f1_score(y_true, y_pred, average='macro')
    micro = f1_score(y_true, y_pred, average='micro')
    acc = accuracy_score(y_true, y_pred)
    (maj_val, _), = collections.Counter(y_true).most_common(1)
    maj = len([x for x in y_true if x == maj_val])/float(len(y_true))
    dicts.append({
        "category":cat,
        "acc": acc,
        "micro":micro,
        "macro":macro,
        "maj":maj
    })
    
df = pd.DataFrame.from_dict(dicts)
print(df)
