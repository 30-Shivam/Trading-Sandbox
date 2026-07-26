"""
Champion/challenger promotion CLI (Phase 5).

Optuna (optimize.py) only ever writes `candidate` System_Config documents --
it never touches `active`. This script is the deliberate, human-gated step
that actually puts a candidate live, so a noisy optimization run can't
silently degrade real trading. With no arguments, lists every System_Config
document (version, status, key metrics, notes) so you can review before
deciding.

Usage:
    python promote_config.py                  # list all versions
    python promote_config.py --promote 3       # promote version 3 to active
"""

import argparse
import sys

import storage


def print_configs() -> None:
    configs = storage.list_configs()
    if not configs:
        print("No System_Config documents yet -- run optimize.py first.")
        return

    headers = ["Version", "Status", "Sharpe", "Trades", "Win%", "Created", "Notes"]
    widths = [9, 10, 9, 8, 8, 22, 60]
    print("".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * sum(widths))
    for c in configs:
        m = c.get("metrics", {}) or {}
        notes = (c.get("notes") or "")
        if len(notes) > widths[-1] - 2:
            notes = notes[: widths[-1] - 5] + "..."
        row = [
            str(c["version"]), c["status"],
            _fmt(m.get("sharpe_like")), str(m.get("trade_count", "-")), _fmt(m.get("win_rate")),
            str(c["created_at"])[:19], notes,
        ]
        print("".join(v.ljust(w) for v, w in zip(row, widths)))


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--promote", type=int, default=None, metavar="VERSION",
                         help="Promote this version to active, retiring whatever was active before.")
    args = parser.parse_args()

    try:
        storage.ensure_indexes()
    except storage.MongoNotConfigured as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    if args.promote is None:
        print_configs()
        return

    doc = storage.get_config_by_version(args.promote)
    if doc is None:
        print(f"[ERROR] No System_Config document with version={args.promote}", file=sys.stderr)
        sys.exit(1)

    previous_active = storage.get_active_config()
    storage.promote_candidate(args.promote)
    print(f"Promoted version {args.promote} to active.")
    print(f"  params: {doc['params']}")
    if previous_active is not None:
        print("Previous active config retired.")
    else:
        print("(No config was active before -- this is the first promotion.)")


if __name__ == "__main__":
    main()
