"""Metrics and the original confusion-matrix, learning-curve and t-SNE plots."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay,
)

CLASS_LABELS = ['Left hand', 'Right hand', 'Foot', 'Tongue']


def plot_cm_counts(cm, labels, save_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap='Blues', ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_cm_percent(cm, labels, save_path):
    cm_f = cm.astype(float)
    row_sums = np.where(cm_f.sum(1, keepdims=True) == 0, 1, cm_f.sum(1, keepdims=True))
    cm_pct   = cm_f / row_sums * 100
    text     = np.array([[f"{v:.1f}%" for v in row] for row in cm_pct])
    thresh   = cm_pct.max() / 2.0

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm_pct, cmap='Greens')
    fig.colorbar(im, ax=ax)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
           xticklabels=labels, yticklabels=labels,
           xlabel='Predicted', ylabel='True')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, text[i, j], ha='center', va='center',
                    color='white' if cm_pct[i, j] > thresh else 'black')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_tsne(model, inputs, y_labels, split_name, save_path):
    features  = model.predict(inputs)
    X_flat    = features.reshape(features.shape[0], -1)
    # The unaugmented smoke run has only 29 validation trials (default perplexity is 30).
    embeddings = TSNE(n_components=2, random_state=42,
                      perplexity=min(30, len(y_labels) - 1)).fit_transform(X_flat)

    unique_cls = np.unique(y_labels)
    colors     = plt.cm.tab10(np.linspace(0, 1, len(unique_cls)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, cls in enumerate(unique_cls):
        idx = y_labels == cls
        ax.scatter(embeddings[idx, 0], embeddings[idx, 1],
                   color=colors[i], label=CLASS_LABELS[cls], s=20)
    ax.set_xlabel('t-SNE 1');  ax.set_ylabel('t-SNE 2')
    ax.set_title(f't-SNE — {split_name}')
    ax.legend(title='Class', loc='upper left')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"t-SNE ({split_name}) saved.")


def plot_learning_curves(history, save_path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(history['accuracy'],     label='Train')
    axs[0].plot(history['val_accuracy'], label='Val')
    axs[0].set_title('Accuracy');  axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy'); axs[0].legend()

    axs[1].plot(history['loss'],     label='Train')
    axs[1].plot(history['val_loss'], label='Val')
    axs[1].set_title('Loss');  axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss'); axs[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)



def evaluate_split(model, inputs, y_true, output_dir, split_name="Test", tsne=True):
    """Save notebook metrics and predictions for one split; return scalar metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_prob = model.predict(inputs, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    cm = confusion_matrix(y_true, y_pred, labels=range(4))
    np.savez(output_dir / f"{split_name}_predictions.npz", y_true=y_true, y_pred=y_pred, y_prob=y_prob, confusion_matrix=cm)
    (output_dir / f"{split_name}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    plot_cm_counts(cm, CLASS_LABELS, output_dir / f"{split_name}_cm_counts.png")
    plot_cm_percent(cm, CLASS_LABELS, output_dir / f"{split_name}_cm_percent.png")
    if tsne:
        plot_name = "Validation" if split_name == "Val" else split_name
        file_name = "val" if split_name in ("Val", "Validation") else split_name.lower()
        plot_tsne(model, inputs, y_true, plot_name, output_dir / f"tsne_{file_name}.png")
    print(f"{split_name}: {metrics}")
    return metrics
