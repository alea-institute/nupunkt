#!/usr/bin/env python3
"""
Compact the bundled default model for faster loading.

Removes orthographic-context entries that cannot influence tokenization (see
``PunktParameters.compact_ortho_context``) and rewrites the model with maximum
gzip compression. Tokenization output is unchanged; only load time, file size
and memory use improve. Run this after retraining the default model.

With ``--drop-ortho`` the orthographic context is removed entirely (see
``PunktParameters.drop_ortho_context``). This is how the bundled default model
is exported: it shrinks the file from ~4 MB to ~25 KB and load time from
~400 ms to ~2 ms, and changes fewer than 0.1% of boundaries on the legal and
general-English gold sets (F1 within +/-0.0003).
"""

import argparse
import gzip
import json
import time
from pathlib import Path

from nupunkt.core.parameters import PunktParameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default = (
        Path(__file__).resolve().parent.parent / "nupunkt" / "models" / "default_model.json.gz"
    )
    parser.add_argument("--input", type=Path, default=default)
    parser.add_argument("--output", type=Path, default=None, help="defaults to --input")
    parser.add_argument(
        "--remove",
        default="",
        help="comma-separated abbreviations (without trailing period) to remove from the model",
    )
    parser.add_argument(
        "--drop-malformed",
        action="store_true",
        help="remove abbreviation entries that start with a period or contain an underscore",
    )
    parser.add_argument(
        "--drop-ortho",
        action="store_true",
        help="remove the orthographic context entirely (inference-only model; used for the "
        "bundled default model)",
    )
    args = parser.parse_args()
    output = args.output or args.input

    t0 = time.perf_counter()
    with gzip.open(args.input, "rt", encoding="utf-8") as f:
        data = json.load(f)
    print(
        f"loaded {args.input} ({args.input.stat().st_size / 1e6:.1f} MB) in {time.perf_counter() - t0:.2f}s"
    )

    params_data = data.get("parameters", data)
    params = PunktParameters.from_json(params_data)

    remove = {a.strip().lower().rstrip(".") for a in args.remove.split(",") if a.strip()}
    if args.drop_malformed:
        remove |= {a for a in params.abbrev_types if a.startswith(".") or "_" in a}
    remove &= params.abbrev_types
    remove.discard("...")
    if remove:
        params.abbrev_types -= remove
        print(
            f"removed {len(remove)} abbreviations: {sorted(remove)[:20]}{' ...' if len(remove) > 20 else ''}"
        )

    before = len(params.ortho_context)
    removed = params.drop_ortho_context() if args.drop_ortho else params.compact_ortho_context()
    print(f"ortho_context: {before} -> {len(params.ortho_context)} entries ({removed} removed)")

    if "parameters" in data:
        data["parameters"] = params.to_json()
    else:
        data = params.to_json()

    with gzip.open(output, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
