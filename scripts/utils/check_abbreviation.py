#!/usr/bin/env python3
"""
Script to check if a token is in the model's abbreviation list.

This utility allows users to verify if a specific token is recognized as an abbreviation
in the default model or a custom model.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Set, Tuple

# Add the parent directory to the path so we can import nupunkt
script_dir = Path(__file__).parent
root_dir = script_dir.parent.parent
sys.path.append(str(root_dir))

# Import nupunkt
from nupunkt.core.parameters import PunktParameters
from nupunkt.models import load_default_model


def check_abbreviation(token: str, model_path: str | None = None) -> Tuple[bool, Set[str]]:
    """
    Check if a token is in the model's abbreviation list.

    Args:
        token: The token to check
        model_path: Optional path to a custom model file

    Returns:
        A tuple containing:
            - True if the token is in the abbreviation list, False otherwise
            - The set of abbreviation types from the model
    """
    # Load the model
    if model_path:
        model_path_obj = Path(model_path)
        # Load parameters directly from file
        params = PunktParameters.load(model_path_obj)
        abbrev_types = params.abbrev_types
    else:
        # Load the default model
        tokenizer = load_default_model()
        # Access to protected member _params is necessary as there's no public API
        # to get the abbreviation types directly from the tokenizer
        abbrev_types = tokenizer._params.abbrev_types

    # Clean the token for checking (remove trailing period if present)
    clean_token = token.lower()
    if clean_token.endswith("."):
        clean_token = clean_token[:-1]

    # Check if the token is in the abbreviation list
    is_abbrev = clean_token in abbrev_types

    return is_abbrev, abbrev_types


def main() -> None:
    """Check if a token is in the model's abbreviation list."""
    parser = argparse.ArgumentParser(
        description="Check if a token is recognized as an abbreviation in the nupunkt model"
    )
    parser.add_argument("token", type=str, nargs="?", help="The token to check")
    parser.add_argument("--model", "-m", type=str, help="Path to a custom model file")
    parser.add_argument(
        "--list", "-l", action="store_true", help="List all abbreviations in the model"
    )
    parser.add_argument(
        "--startswith", "-s", type=str, help="List abbreviations starting with the given prefix"
    )
    parser.add_argument(
        "--count",
        "-c",
        action="store_true",
        help="Show the total count of abbreviations in the model",
    )

    args = parser.parse_args()

    # For operations that don't require a specific token
    if args.list or args.startswith is not None or args.count:
        dummy_token = "a"  # Just use a dummy token to load the model
        _, abbrev_types = check_abbreviation(dummy_token, args.model)

        if args.list:
            # Sort and list all abbreviations
            sorted_abbrevs = sorted(abbrev_types)
            print(f"\nAll abbreviations in the model ({len(sorted_abbrevs)}):")
            for abbrev in sorted_abbrevs:
                print(f"  {abbrev}")
            print()

        if args.startswith is not None:
            # List abbreviations starting with the given prefix
            prefix = args.startswith.lower()
            matching_abbrevs = [abbr for abbr in abbrev_types if abbr.startswith(prefix)]
            sorted_matches = sorted(matching_abbrevs)
            print(f"\nAbbreviations starting with '{prefix}' ({len(sorted_matches)}):")
            for abbrev in sorted_matches:
                print(f"  {abbrev}")
            print()

        if args.count:
            # Show total count
            print(f"\nTotal abbreviations in the model: {len(abbrev_types)}\n")

        return

    # For checking a specific token
    if args.token is None:
        parser.print_help()
        print("\nError: Please provide a token to check or use --list, --startswith, or --count.")
        sys.exit(1)

    # Load the model and get abbreviations
    is_abbrev, abbrev_types = check_abbreviation(args.token, args.model)

    # Check the specific token
    token = args.token
    clean_token = token.lower()
    if clean_token.endswith("."):
        clean_token = clean_token[:-1]

    if is_abbrev:
        print(f"\nYes, '{clean_token}' is recognized as an abbreviation in the model.\n")
    else:
        print(f"\nNo, '{clean_token}' is NOT recognized as an abbreviation in the model.\n")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # This prevents the "Broken pipe" error message when piping output to tools like 'head'
        # Python flushes standard streams on exit; redirect remaining output
        # to /dev/null to avoid another BrokenPipeError at shutdown
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(0)
