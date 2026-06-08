"""
Shared sequence, region, and amino-acid validation utilities.

Centralizes pure helpers that are needed by several layers (CLI,
mutagenesis, analysis) so the logic lives in exactly one place:

- Amino-acid validation (``validate_amino_acids`` / ``validate_sequence``)
- Region string parsing (``parse_region``)
- Region normalization including the special ``"all"`` token
  (``normalize_regions``)

Keeping these here removes the previous duplication of ``parse_region``
between the CLI and the mutagenesis module.
"""
from typing import List, Optional, Tuple, Union

from .config import VALID_AAS


# =============================================================================
# AMINO ACID VALIDATION
# =============================================================================

def validate_amino_acids(
        input_data: Union[str, List[str]],
        name: str = "amino acids",
        strict: bool = False
) -> bool:
    """
    Validate amino acid codes against the canonical 20-letter set.

    Args:
        input_data: A sequence string, or a list of single-letter codes
        name: Description used in error messages
        strict: If True, raise ValueError on invalid input instead of
                returning False

    Returns:
        True if all codes are valid, False otherwise (when strict=False)

    Raises:
        ValueError: If strict=True and invalid amino acids are found
        TypeError: If input_data is neither str nor list
    """
    if isinstance(input_data, str):
        invalid_chars = [
            char for char in set(input_data.upper())
            if char not in VALID_AAS
        ]
        if invalid_chars:
            if strict:
                raise ValueError(
                    f"Invalid {name}: contains invalid characters "
                    f"{invalid_chars}. "
                    f"Valid amino acids: {', '.join(sorted(VALID_AAS))}"
                )
            return False
        return True

    elif isinstance(input_data, list):
        invalid = [
            aa for aa in input_data
            if len(aa) != 1 or aa.upper() not in VALID_AAS
        ]
        if invalid:
            if strict:
                raise ValueError(
                    f"Invalid {name}: {invalid}. "
                    f"Valid amino acids: {', '.join(sorted(VALID_AAS))}"
                )
            return False
        return True

    else:
        raise TypeError(
            f"input_data must be str or list, got {type(input_data).__name__}"
        )


def validate_sequence(sequence: str) -> bool:
    """Validate that the sequence contains only valid amino acid codes."""
    return validate_amino_acids(sequence, "sequence", strict=False)


# =============================================================================
# REGION PARSING
# =============================================================================

def parse_region(region_str: str, seq_length: int) -> Tuple[int, int]:
    """
    Parse a region string in ``start:stop`` format (1-indexed, inclusive).

    Args:
        region_str: Region specification (e.g., "10:50")
        seq_length: Length of the full sequence (for bounds checking)

    Returns:
        Tuple of (start, stop) positions

    Raises:
        ValueError: If the format is invalid or the region is out of bounds
    """
    try:
        if ':' not in region_str:
            raise ValueError("Region must be in format start:stop")

        start_str, stop_str = region_str.split(':')
        start = int(start_str.strip())
        stop = int(stop_str.strip())

        if start < 1 or stop > seq_length:
            raise ValueError(
                f"Region {start}:{stop} out of bounds (1-{seq_length})"
            )
        if start > stop:
            raise ValueError(
                f"Start position {start} cannot be greater than "
                f"stop position {stop}"
            )

        return start, stop
    except ValueError as e:
        raise ValueError(f"Invalid region format '{region_str}': {e}")


def normalize_regions(
        regions: Optional[List[str]],
        seq_length: int
) -> Optional[List[str]]:
    """
    Normalize ``--regions`` values, expanding the special ``"all"`` token.

    If any token equals "all" (case-insensitive), the whole sequence
    (``1:<seq_length>``) is analyzed and takes precedence over other tokens.

    Args:
        regions: List of region strings, or None
        seq_length: Length of the full sequence

    Returns:
        Normalized list of region strings, or None if nothing was provided
    """
    if not regions:
        return None

    tokens = [
        r.strip() for r in regions
        if r is not None and str(r).strip()
    ]
    if not tokens:
        return None

    if any(t.lower() == "all" for t in tokens):
        return [f"1:{seq_length}"]

    return tokens
