from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_yelp_dataset(raw_dir: Path, out_dir: Path, top_cities: set[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    biz_in = raw_dir / "yelp_academic_dataset_business.json"
    usr_in = raw_dir / "yelp_academic_dataset_user.json"
    rev_in = raw_dir / "yelp_academic_dataset_review.json"

    item_out = out_dir / "item.json"
    review_out = out_dir / "review.json"
    user_out = out_dir / "user.json"

    for file_path in (biz_in, usr_in, rev_in):
        if not file_path.exists():
            raise FileNotFoundError(f"Missing Yelp file: {file_path}")

    # 1) Filter businesses -> item.json and collect allowed business IDs
    allowed_biz: set[str] = set()
    with biz_in.open("r", encoding="utf-8") as f, item_out.open(
        "w", encoding="utf-8"
    ) as w:
        for line in f:
            x = json.loads(line)
            if x.get("city") in top_cities:
                x["item_id"] = x.pop("business_id")
                x["source"] = "yelp"
                x["type"] = "business"
                allowed_biz.add(x["item_id"])
                w.write(json.dumps(x, ensure_ascii=False) + "\n")

    # 2) Filter reviews by allowed business -> review.json and collect user IDs
    allowed_users: set[str | None] = set()
    with rev_in.open("r", encoding="utf-8") as f, review_out.open(
        "w", encoding="utf-8"
    ) as w:
        for line in f:
            x = json.loads(line)
            bid = x.get("business_id")
            if bid in allowed_biz:
                x["item_id"] = x.pop("business_id")
                x["source"] = "yelp"
                x["type"] = "business"
                allowed_users.add(x.get("user_id"))
                w.write(json.dumps(x, ensure_ascii=False) + "\n")

    # 3) Filter users by users present in filtered reviews -> user.json
    with usr_in.open("r", encoding="utf-8") as f, user_out.open(
        "w", encoding="utf-8"
    ) as w:
        for line in f:
            x = json.loads(line)
            if x.get("user_id") in allowed_users:
                x["source"] = "yelp"
                w.write(json.dumps(x, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build yelp_dataset/{item,review,user}.json from raw Yelp data."
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw_yelp",
        help="Directory containing Yelp raw JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        default="yelp_dataset",
        help="Output directory for item.json, review.json, user.json.",
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=["Philadelphia", "Tampa", "Tucson"],
        help="City names to keep.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    top_cities = {city.strip() for city in args.cities if city.strip()}

    if not top_cities:
        raise ValueError("At least one city is required via --cities.")

    build_yelp_dataset(raw_dir=raw_dir, out_dir=out_dir, top_cities=top_cities)
    print(f"Done: {out_dir}")
    print(f"Cities: {sorted(top_cities)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
