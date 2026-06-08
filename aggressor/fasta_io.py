"""
FASTA input/output and multi-mutation file organization.

Groups all on-disk I/O concerns in one place:

- Reading a single-sequence FASTA (``read_fasta``)
- Writing mutations to FASTA (``write_fasta`` / ``_write_fasta_file``)
- Creating the organized multi-mutation output directory structure
  (``create_output_directory`` / ``write_multi_mutations_by_category``)

Separating I/O from argument parsing and reporting keeps the CLI module
focused on orchestration.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import FASTA_LINE_LENGTH, logger


# =============================================================================
# FASTA READING
# =============================================================================

def read_fasta(filepath: str) -> Tuple[str, str]:
    """
    Read a single-sequence FASTA file.

    Args:
        filepath: Path to FASTA file

    Returns:
        Tuple of (header_line, sequence)

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid or empty
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        header = ""
        sequence = ""

        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                if header:
                    # Already have a header — single sequence mode
                    break
                header = line
            elif line and not header:
                raise ValueError(
                    "FASTA file must start with header line (>)"
                )
            elif header:
                sequence += line.upper()

        if not header or not sequence:
            raise ValueError("Invalid FASTA file or empty sequence")

        return header, sequence

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Error reading FASTA file: {e}")


# =============================================================================
# FASTA WRITING
# =============================================================================

def write_fasta(
        output_file: str,
        original_header: str,
        original_seq: str,
        mutations: List[Tuple[str, str]],
        include_original: bool = True
) -> None:
    """
    Write mutations to a FASTA file.

    Args:
        output_file: Path to output FASTA file
        original_header: Original FASTA header line
        original_seq: Original protein sequence
        mutations: List of (description, sequence) tuples
        include_original: Whether to write the original sequence first
    """
    with open(output_file, 'w') as f:
        if include_original:
            f.write(f"{original_header}\n")
            for i in range(0, len(original_seq), FASTA_LINE_LENGTH):
                f.write(f"{original_seq[i:i + FASTA_LINE_LENGTH]}\n")

        protein_name = original_header[1:].strip()
        for description, mutated_seq in mutations:
            f.write(f">{protein_name}_{description}\n")
            for j in range(0, len(mutated_seq), FASTA_LINE_LENGTH):
                f.write(f"{mutated_seq[j:j + FASTA_LINE_LENGTH]}\n")


def _write_fasta_file(
        output_file: str,
        original_header: str,
        original_seq: str,
        mutations: List[Tuple[str, str, int]],
        include_original: bool = True
) -> None:
    """
    Write scored mutations (description, sequence, score) to a FASTA file.

    Args:
        output_file: Path to output FASTA file
        original_header: Original FASTA header line
        original_seq: Original protein sequence
        mutations: List of (description, sequence, score) tuples
        include_original: Whether to write the original sequence first
    """
    with open(output_file, 'w') as f:
        if include_original:
            f.write(f"{original_header}\n")
            for i in range(0, len(original_seq), FASTA_LINE_LENGTH):
                f.write(f"{original_seq[i:i + FASTA_LINE_LENGTH]}\n")

        protein_name = original_header[1:].strip()
        for item in mutations:
            description = item[0]
            mutated_seq = item[1]
            f.write(f">{protein_name}_{description}\n")
            for j in range(0, len(mutated_seq), FASTA_LINE_LENGTH):
                f.write(f"{mutated_seq[j:j + FASTA_LINE_LENGTH]}\n")


# =============================================================================
# MULTI-MUTATION OUTPUT STRUCTURE
# =============================================================================

def _level_to_text(level: int) -> str:
    """Convert a mutation level number to text (2 -> "double", etc.)."""
    level_names = {
        2: 'double',
        3: 'triple',
        4: 'quadruple',
        5: 'quintuple',
        6: 'sextuple'
    }
    return level_names.get(level, f'{level}x')


def create_output_directory(
        base_path: str,
        multi_mutation_levels: Optional[List[int]] = None
) -> Path:
    """
    Create the organized multi-mutation output directory structure.

    Args:
        base_path: Base directory path
        multi_mutation_levels: Levels to create subdirectories for

    Returns:
        Path object for the base output directory
    """
    output_dir = Path(base_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if multi_mutation_levels:
        for level in sorted(multi_mutation_levels):
            level_dir = output_dir / f"{_level_to_text(level)}_mutations"
            level_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {level_dir}")

    return output_dir


def write_multi_mutations_by_category(
        output_dir: str,
        original_header: str,
        original_seq: str,
        categorized_mutations: Dict[int, Dict[str, List[Dict]]],
        include_original: bool = True
) -> None:
    """
    Write categorized multi-mutations to separate FASTA files per level.

    Args:
        output_dir: Base output directory
        original_header: Original FASTA header
        original_seq: Original sequence
        categorized_mutations: Categorized mutations (level -> category -> items)
        include_original: Whether to write the original sequence in each file
    """
    output_base = Path(output_dir)

    for level in sorted(categorized_mutations.keys()):
        level_dir = output_base / f"{_level_to_text(level)}_mutations"
        level_dir.mkdir(parents=True, exist_ok=True)

        categories = categorized_mutations[level]

        category_files = [
            ('single_region', 'single_region.fasta'),
            ('multi_region', 'multi_region.fasta'),
            ('all_gatekeeper', 'all_gatekeeper.fasta'),
            ('all_core', 'all_core.fasta'),
            ('mixed', 'mixed.fasta'),
        ]

        for category_name, filename in category_files:
            if categories.get(category_name):
                file_path = level_dir / filename
                mutations_list = [
                    (
                        item['description'],
                        item['sequence'],
                        item.get('agg_score', 0)
                    )
                    for item in categories[category_name]
                ]
                _write_fasta_file(
                    str(file_path), original_header, original_seq,
                    mutations_list, include_original
                )
                logger.info(
                    f"Wrote {len(mutations_list)} {category_name} "
                    f"{_level_to_text(level)} mutations to {filename}"
                )
