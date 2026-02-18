#!/usr/bin/env python

import argparse
from src.preprocessing import create_autopet_decathlon_datalist




def main():
    parser = argparse.ArgumentParser(
        description="Create a datalist for the AutoPET Decathlon dataset.",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="",
        help="root directory of the  AutoPET dataset",
    )

    args = parser.parse_args()

    create_autopet_decathlon_datalist(
        root_dir=args.root_dir,
    )

if __name__ == "__main__":
    main()