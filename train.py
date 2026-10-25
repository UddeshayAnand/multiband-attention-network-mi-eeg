"""Train the notebook model independently for each requested subject."""
import argparse
import csv
import json
from pathlib import Path
import time


def train_subject(data_path, subject, output_dir, epochs=1320, batch_size=64,
                  expansion_factor=25, seed=42, tsne=True):
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.callbacks import Callback, ModelCheckpoint
    from get_data import get_data
    from preprocessing import prepare_training_data, make_inputs
    from model import build_model
    from evaluate import evaluate_split, plot_learning_curves

    if epochs < 1 or batch_size < 1 or expansion_factor < 0:
        raise ValueError("epochs/batch_size must be positive; expansion_factor must be nonnegative")
    # A separate model and seed for each subject, as in the individual notebooks.
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    run_dir = Path(output_dir) / f"subject_{subject:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config = dict(subject=subject, epochs=epochs, batch_size=batch_size,
                  expansion_factor=expansion_factor, seed=seed, val_fraction=0.1,
                  learning_rate=1e-4, stopping_threshold=0.025,
                  evaluation_weights="final", tsne=tsne, tensorflow=tf.__version__,
                  keras=tf.keras.__version__, numpy=np.__version__)
    (run_dir / "run.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    X_train, y_train, X_test, y_test = get_data(data_path, subject)
    train_inputs, val_inputs, y_tr, y_val = prepare_training_data(
        X_train, y_train, expansion_factor=expansion_factor, seed=seed,
    )
    test_inputs = make_inputs(X_test.transpose(0, 2, 1))
    model = build_model()
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=["accuracy"])

    class ThresholdStopping(Callback):
        """The notebook's absolute validation-loss threshold (no patience)."""
        def on_epoch_end(self, epoch, logs=None):
            val = (logs or {}).get("val_loss")
            if val is not None and val <= 0.025:
                print(f"Epoch {epoch + 1}: val_loss={val:.4f}; stopping.")
                self.model.stop_training = True

    callbacks = [
        ThresholdStopping(),
        ModelCheckpoint(str(run_dir / "best.weights.h5"), monitor="val_loss",
                        save_best_only=True, save_weights_only=True, mode="min", verbose=1),
    ]
    start = time.time()
    history = model.fit(
        train_inputs, tf.keras.utils.to_categorical(y_tr, 4),
        validation_data=(val_inputs, tf.keras.utils.to_categorical(y_val, 4)),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks, shuffle=True, verbose=2,
    )
    # Notebook evaluation uses the final in-memory model, not the best checkpoint.
    model.save_weights(run_dir / "final.weights.h5")
    np.savez(run_dir / "training_history.npz", **history.history)
    plot_learning_curves(history.history, run_dir / "learning_curves.png")
    config.update(completed_epochs=len(history.history["loss"]), training_seconds=time.time() - start)
    (run_dir / "run.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for name, inputs, labels in [("Train", train_inputs, y_tr), ("Val", val_inputs, y_val),
                                 ("Test", test_inputs, y_test)]:
        metrics = evaluate_split(model, inputs, labels, run_dir, name, tsne=tsne)
    return {"subject": subject, **metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--subjects", nargs="+", type=int, choices=range(1, 10), default=list(range(1, 10)))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=1320)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--expansion-factor", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tsne", action=argparse.BooleanOptionalAction, default=True,
                        help="Generate notebook t-SNE plots (default: enabled; --no-tsne skips them)")
    args = parser.parse_args()
    if len(set(args.subjects)) != len(args.subjects):
        parser.error("subjects must not contain duplicates")
    # Check all destinations before starting a multi-subject run.
    if any((args.output_dir / f"subject_{s:02d}").exists() for s in args.subjects):
        parser.error("a subject output folder already exists; choose a new --output-dir")
    rows = []
    for subject in args.subjects:
        rows.append(train_subject(args.data_path, subject, args.output_dir, args.epochs,
                                  args.batch_size, args.expansion_factor, args.seed, args.tsne))
        # Persist progress after each completed subject.
        with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
