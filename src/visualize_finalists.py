from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

root = Path(__file__).resolve().parent
run = root / 'results/hyperparameter_optimization/runs/standard_20_trials_08542f399edf426395d330ec55b2e74a'
final = run / 'finalists'
out = final / 'figures'
if out.exists():
    raise FileExistsError(f'Output already exists: {out}')
data = pd.read_csv(root / 'data/emails.csv')
X = data.drop(columns=[c for c in ('Email No.', 'Prediction') if c in data])
y = data['Prediction'].astype(int)
_, xt, _, yt = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
summary = json.loads((final / 'held_out_comparison.txt').read_text(encoding='utf-8'))
models = {}
for row in summary['models']:
    name = row['model']
    folder = final / name.lower().replace(' ', '_')
    stored = pd.read_csv(folder / 'predictions.txt', sep=r'\s+')
    model = joblib.load(folder / 'model.joblib')
    predicted = model.predict(xt)
    if not (np.array_equal(stored.y_true.to_numpy(), np.asarray(yt)) and np.array_equal(stored.y_pred.to_numpy(), predicted)):
        raise ValueError(f'Holdout or predictions do not match saved results: {name}')
    scores = model.predict_proba(xt)[:, list(model.classes_).index(1)]
    models[name] = (row, predicted, scores)
out.mkdir(parents=True)
results = {}
for name, (row, pred, score) in models.items():
    cm = confusion_matrix(yt, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    im = ax.imshow(cm, cmap='Blues')
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha='center', va='center', color='white' if value > cm.max()/2 else 'black')
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=['Normal','Spam'], yticklabels=['Normal','Spam'], xlabel='Predicted class', ylabel='True class', title=f'Confusion matrix — {name}')
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out / (name.lower().replace(' ','_') + '_confusion_matrix.pdf')); plt.close(fig)
    results[name] = {'tn': int(cm[0,0]), 'fp': int(cm[0,1]), 'fn': int(cm[1,0]), 'tp': int(cm[1,1]), 'roc_auc': float(roc_auc_score(yt, score)), 'average_precision': float(average_precision_score(yt, score))}
fig, ax = plt.subplots(figsize=(9, 4.8))
metrics = ['accuracy','precision','recall','f1_score']; x = np.arange(len(metrics)); width = 0.23
for i, (name, (row, _, _)) in enumerate(models.items()):
    ax.bar(x+(i-1)*width, [row[m] for m in metrics], width, label=name)
ax.set(xticks=x, xticklabels=['Accuracy','Precision','Recall','F1'], ylim=(0,1.05), ylabel='Score', title='Held-out metric comparison'); ax.legend(); fig.tight_layout(); fig.savefig(out/'metric_comparison.pdf'); plt.close(fig)
fig, ax = plt.subplots(figsize=(6,5))
for name, (_, _, score) in models.items():
    fpr, tpr, _ = roc_curve(yt, score); ax.plot(fpr,tpr,label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
ax.plot([0,1],[0,1], '--', color='gray', label='Random ranking'); ax.set(xlabel='False positive rate',ylabel='True positive rate',title='ROC curves — held-out set'); ax.legend(); fig.tight_layout(); fig.savefig(out/'roc_curves.pdf'); plt.close(fig)
fig, ax = plt.subplots(figsize=(6,5))
for name, (_, _, score) in models.items():
    precision, recall, _ = precision_recall_curve(yt, score); ax.plot(recall,precision,label=f"{name} (AP={results[name]['average_precision']:.3f})")
ax.axhline(float(np.mean(yt)), linestyle='--', color='gray', label=f'Spam prevalence ({np.mean(yt):.3f})'); ax.set(xlabel='Recall',ylabel='Precision',title='Precision-recall curves — held-out set',ylim=(0,1.05)); ax.legend(); fig.tight_layout(); fig.savefig(out/'precision_recall_curves.pdf'); plt.close(fig)
(final / 'visualization_metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))
