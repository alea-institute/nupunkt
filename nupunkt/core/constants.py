"""
Constants module for nupunkt.

This module provides constants used in the Punkt algorithm,
particularly for orthographic context.
"""

from typing import Dict, Tuple

# -------------------------------------------------------------------
# Orthographic Context Constants
# -------------------------------------------------------------------

# Bit flags for orthographic contexts
ORTHO_BEG_UC = 1 << 1  # Beginning of sentence, uppercase
ORTHO_MID_UC = 1 << 2  # Middle of sentence, uppercase
ORTHO_UNK_UC = 1 << 3  # Unknown position, uppercase
ORTHO_BEG_LC = 1 << 4  # Beginning of sentence, lowercase
ORTHO_MID_LC = 1 << 5  # Middle of sentence, lowercase
ORTHO_UNK_LC = 1 << 6  # Unknown position, lowercase

# Combined flags
ORTHO_UC = ORTHO_BEG_UC | ORTHO_MID_UC | ORTHO_UNK_UC  # Any uppercase
ORTHO_LC = ORTHO_BEG_LC | ORTHO_MID_LC | ORTHO_UNK_LC  # Any lowercase

# Mapping from (position, case) to flag
ORTHO_MAP: Dict[Tuple[str, str], int] = {
    ("initial", "upper"): ORTHO_BEG_UC,
    ("internal", "upper"): ORTHO_MID_UC,
    ("unknown", "upper"): ORTHO_UNK_UC,
    ("initial", "lower"): ORTHO_BEG_LC,
    ("internal", "lower"): ORTHO_MID_LC,
    ("unknown", "lower"): ORTHO_UNK_LC,
}
