"""Write train / half_a / half_b / test_inputs / test_labels parquet files (80/20 split from the team's split.py)."""
from _common import parser, setup_logging

from sentineldesk.triage.config import artifacts_dir, load_config
from sentineldesk.triage.data import build_parts, write_parts


def main() -> None:
    args = parser(__doc__).parse_args()
    setup_logging()
    cfg = load_config(args.config)
    parts = build_parts(cfg["data"].get("csv_path"), sample=args.sample)
    write_parts(parts, artifacts_dir(cfg, args.artifacts, args.sample) / "splits")


if __name__ == "__main__":
    main()
