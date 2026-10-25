"""Evaluate a saved subject checkpoint on its untouched Session E data."""
import argparse
import json
from pathlib import Path


def test_subject(data_path, run_dir, checkpoint="final", tsne=True):
    from get_data import load_BCI2a_data
    from preprocessing import make_inputs
    from model import build_model
    from evaluate import evaluate_split

    run_dir = Path(run_dir)
    config = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    X, y = load_BCI2a_data(data_path, config["subject"], training=False)
    model = build_model()
    model.load_weights(run_dir / f"{checkpoint}.weights.h5")
    return evaluate_split(model, make_inputs(X.transpose(0, 2, 1)), y,
                          run_dir / f"evaluation_{checkpoint}", "Test", tsne=tsne)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", choices=["final", "best"], default="final")
    parser.add_argument("--tsne", action=argparse.BooleanOptionalAction, default=True,
                        help="Generate the test t-SNE plot (default: enabled; --no-tsne skips it)")
    args = parser.parse_args()
    test_subject(args.data_path, args.run_dir, args.checkpoint, args.tsne)


if __name__ == "__main__":
    main()
