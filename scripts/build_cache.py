"""Reads WOMD tf_example shards and writes a filtered scenario bank.

Example:
    python scripts/build_cache.py --tfrecords data/womd/val/*.tfrecord* \
        --num 200 --out data/cache/val.pkl
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from safedrive import data  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tfrecords", nargs="+", required=True)
    p.add_argument("--num", type=int, default=300)
    p.add_argument("--out", required=True)
    p.add_argument("--rg_points", type=int, default=3000)
    p.add_argument("--max_objects", type=int, default=32)
    p.add_argument("--min_path_length", type=float, default=20.0)
    args = p.parse_args()

    t0 = time.time()
    bank = data.build_bank(args.tfrecords, args.num, rg_points=args.rg_points,
                           max_num_objects=args.max_objects,
                           min_path_length=args.min_path_length)
    data.save_bank(bank, args.out)
    size = os.path.getsize(args.out) / 1e6
    print(f"saved {len(bank['ids'])} scenarios to {args.out} ({size:.0f} MB) "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
