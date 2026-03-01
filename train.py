import argparse
import os

import pandas as pd

from src.model import cross_validate, train_full, save_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with columns: text,label")
    ap.add_argument("--out", required=True, help="Output model directory")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV folds (auto-reduced if dataset too small).",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise SystemExit(f"CSV must contain columns '{args.text_col}' and '{args.label_col}'")

    print(f"== {args.n_splits}-fold cross validation ==")
    cv = cross_validate(df, text_col=args.text_col, label_col=args.label_col, n_splits=args.n_splits)
    for k, v in cv.items():
        print(f"{k}: {v:.4f}")

    print("\n== Training full model ==")
    model = train_full(df, text_col=args.text_col, label_col=args.label_col)

    os.makedirs(args.out, exist_ok=True)
    save_model(model, args.out)
    print(f"Saved model to: {args.out}")


if __name__ == "__main__":
    main()
