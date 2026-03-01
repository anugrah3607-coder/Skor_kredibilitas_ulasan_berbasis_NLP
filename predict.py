import argparse

import pandas as pd

from src.model import load_model, score_reviews


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--input", required=True, help="CSV with column: text")
    ap.add_argument("--output", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--reference-csv", default=None, help="Optional CSV of reference reviews (for similarity)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise SystemExit(f"Input CSV must contain column '{args.text_col}'")

    model = load_model(args.model_dir)

    reference_texts = None
    if args.reference_csv:
        ref = pd.read_csv(args.reference_csv)
        if args.text_col not in ref.columns:
            raise SystemExit(f"Reference CSV must contain column '{args.text_col}'")
        reference_texts = ref[args.text_col].astype(str).tolist()

    scores = score_reviews(model, df[args.text_col].astype(str).tolist(), reference_texts=reference_texts)
    out = pd.concat([df, scores], axis=1)
    out.to_csv(args.output, index=False)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
