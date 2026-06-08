"""
Command-line interface for the AGGRESSOR mutagenesis pipeline.

This module is the orchestration layer only. It:

- Builds the argument parser (``setup_argument_parser``)
- Validates parsed arguments (``validate_arguments``)
- Wires together the pipeline in ``main``

Supporting concerns live in dedicated modules:

- ``sequtils``   : region parsing/normalization and amino-acid validation
- ``fasta_io``   : reading/writing FASTA and the multi-mutation directory tree
- ``reporting``  : human-readable help and result summaries
- ``analysis``   : aggregation-region analysis
- ``mutagenesis``: single- and multi-point mutation generation

Usage:
    python cli.py <input_file> [options]
    python main.py <input_file> [options]
"""
import argparse
import sys
from pathlib import Path

from .config import (
    GATEKEEPING_AAS,
    GATEKEEPER_DISTANCE,
    DEFAULT_MUTATIONS,
    MAX_MUTATION_LEVEL,
    setup_logging,
    logger
)
from .sequtils import parse_region, normalize_regions, validate_amino_acids
from .fasta_io import (
    read_fasta,
    write_fasta,
    create_output_directory,
    write_multi_mutations_by_category
)
from .reporting import (
    print_help_info,
    print_mutation_summary,
    print_aggregation_summary
)
from .analysis import analyze_region
from .mutagenesis import (
    mutate_sequence,
    generate_multi_point_mutations,
    categorize_multi_mutations
)


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Configure and return the argument parser.

    Returns:
        Configured ArgumentParser with all option groups
    """
    parser = argparse.ArgumentParser(
        description=(
            'AGGRESSOR: Aggregation-Guided Generation of REgion-Specific '
            'Substitution ORiented mutations.\n'
            'Performs rule-based in silico mutagenesis on protein sequences '
            'with multi-point mutation support.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        usage='aggressor <input_file> [options]  (or: python -m aggressor ...)'
    )

    # Required arguments
    required = parser.add_argument_group('REQUIRED ARGUMENTS')
    required.add_argument(
        'input_file', nargs='?',
        help='Input FASTA file containing protein sequence'
    )

    # Region and rule arguments
    region_args = parser.add_argument_group('REGION-BASED MUTAGENESIS')
    region_args.add_argument(
        '-r', '--regions', type=str, nargs='+',
        help='Regions to analyze (format: start:stop) or "all"'
    )
    region_args.add_argument(
        '--rules', type=str, nargs='+',
        choices=[
            'hydrophobic_aliphatic', 'aromatic',
            'amide', 'hydrophobic_and_aromatic'
        ],
        help='Specific rules to apply'
    )

    # Direct mutation arguments
    mutation_args = parser.add_argument_group('DIRECT MUTATIONS')
    mutation_args.add_argument(
        '-p', '--positions', type=int, nargs='+', default=[],
        help='Specific positions to mutate (1-indexed)'
    )
    mutation_args.add_argument(
        '-m', '--mutations', type=str, nargs='+',
        default=DEFAULT_MUTATIONS,
        help=f'Amino acids to mutate to (default: {DEFAULT_MUTATIONS})'
    )

    # Gatekeeping amino acids
    gatekeeping_args = parser.add_argument_group('GATEKEEPING AMINO ACIDS')
    gatekeeping_args.add_argument(
        '-g', '--gatekeeping', type=str, nargs='+',
        default=GATEKEEPING_AAS,
        help=(
            f'Amino acids for edge positions '
            f'(default: {GATEKEEPING_AAS})'
        )
    )
    gatekeeping_args.add_argument(
        '--gatekeeper-distance', type=int, default=GATEKEEPER_DISTANCE,
        help=(
            'Length of the residue stretch flanking each aggregation-prone '
            'region (on each side) that is also mutated (gatekeeping AAs only) '
            f'as part of the gatekeeper zone (default: {GATEKEEPER_DISTANCE}; '
            'use 0 to mutate cluster core positions only)'
        )
    )
    gatekeeping_args.add_argument(
        '--internal-gatekeepers', action='store_true',
        help=(
            'Also mutate the internal gap residues inside merged clusters '
            '(the short zones between merged sub-clusters) as gatekeepers '
            '(default: off)'
        )
    )

    # Insertion arguments
    insertion_args = parser.add_argument_group('INSERTIONS')
    insertion_args.add_argument(
        '--insert-positions', type=int, nargs='+',
        help='Positions for insertions (before this position)'
    )
    insertion_args.add_argument(
        '--insert-aas', type=str, nargs='+',
        help='Amino acids to insert'
    )

    # Multi-point mutation arguments
    multi_args = parser.add_argument_group('MULTI-POINT MUTATIONS')
    multi_args.add_argument(
        '--multi-mutations', type=int, nargs='+',
        help='Levels to generate (e.g., 2 3 for double and triple)'
    )
    multi_args.add_argument(
        '--multi-top-per-position', type=int, default=3,
        help='Limit variants per position (default: 3)'
    )
    multi_args.add_argument(
        '--multi-output', default='mutated_sequences',
        help='Output directory for mutations (default: mutated_sequences)'
    )
    multi_args.add_argument(
        '--threads', '-t', type=int, default=None,
        help='Number of parallel workers (default: CPU count - 1)'
    )

    # Aggregation analysis arguments
    agg_args = parser.add_argument_group('AGGREGATION ANALYSIS')
    agg_args.add_argument(
        '--agg-only', action='store_true',
        help='Only identify aggregation hotspots without generating mutations'
    )
    agg_args.add_argument(
        '--min-agg-score', type=int, default=4,
        help='Minimum aggregation score for hotspot (default: 4)'
    )

    # Output arguments
    output_args = parser.add_argument_group('OUTPUT')
    output_args.add_argument(
        '-o', '--output', default='mutated_sequences.fasta',
        help='Output FASTA file for single mutations'
    )
    output_args.add_argument(
        '--no-original', action='store_true',
        help='Do not include original sequence in output'
    )

    # Other options
    other_args = parser.add_argument_group('OTHER OPTIONS')
    other_args.add_argument(
        '-v', '--verbose', action='store_true',
        help='Show detailed analysis'
    )
    other_args.add_argument(
        '-h', '--help', action='store_true',
        help='Show this help message'
    )

    return parser


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.

    Args:
        args: Parsed command line arguments

    Raises:
        SystemExit: If validation fails
    """
    if (
            not args.agg_only
            and not args.positions
            and not args.regions
            and not args.insert_positions
    ):
        print("\nERROR: You must specify at least one of:")
        print("  • --positions for direct mutations")
        print("  • --regions for rule-based mutations")
        print("  • --insert-positions for insertions")
        print("  • --agg-only for aggregation analysis only")
        print("\nUse --help for more information.")
        sys.exit(1)

    # Validate mutation list
    try:
        validate_amino_acids(args.mutations, "mutation amino acids", strict=True)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate gatekeeping amino acids
    try:
        validate_amino_acids(args.gatekeeping, "gatekeeping amino acids", strict=True)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate gatekeeper distance
    if args.gatekeeper_distance < 0:
        print(
            "\nERROR: --gatekeeper-distance must be >= 0",
            file=sys.stderr
        )
        sys.exit(1)

    # Validate insertion arguments come in pairs
    if bool(args.insert_positions) != bool(args.insert_aas):
        print(
            "\nERROR: Both --insert-positions and --insert-aas "
            "must be provided together",
            file=sys.stderr
        )
        sys.exit(1)

    # Validate multi-mutation levels
    if args.multi_mutations:
        for level in args.multi_mutations:
            if level < 2:
                print(
                    f"\nERROR: Multi-mutation levels must be >= 2 (got {level})",
                    file=sys.stderr
                )
                sys.exit(1)
            if level > MAX_MUTATION_LEVEL:
                print(
                    f"\nERROR: Maximum mutation level is "
                    f"{MAX_MUTATION_LEVEL} (got {level})",
                    file=sys.stderr
                )
                sys.exit(1)

        if args.multi_top_per_position is not None and args.multi_top_per_position < 1:
            print(
                "\nERROR: --multi-top-per-position must be >= 1",
                file=sys.stderr
            )
            sys.exit(1)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """Main entry point for AGGRESSOR pipeline."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Setup logging before any operations
    setup_logging(verbose=args.verbose)

    if args.help or not args.input_file:
        print_help_info(parser)
        if not args.input_file:
            logger.error("Input FASTA file is required")
            sys.exit(1)
        sys.exit(0)

    try:
        validate_arguments(args)
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Argument validation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    try:
        logger.info("=" * 70)
        logger.info("AGGRESSOR: RULE-BASED MUTAGENESIS WITH OPTIMIZED CLUSTERING")
        logger.info("=" * 70)

        # Read input
        try:
            header, sequence = read_fasta(args.input_file)
        except FileNotFoundError:
            logger.error(f"Input file not found: {args.input_file}")
            sys.exit(1)
        except ValueError as e:
            logger.error(f"Invalid FASTA format: {e}")
            sys.exit(1)

        logger.info(f"Input: {args.input_file}")
        logger.info(f"Sequence length: {len(sequence)} residues")

        # Normalize regions
        try:
            args.regions = normalize_regions(args.regions, len(sequence))
        except Exception as e:
            logger.error(f"Region normalization failed: {e}")
            sys.exit(1)

        if args.regions:
            logger.info(f"Regions to analyze: {args.regions}")
        if args.positions:
            logger.info(f"Direct mutation positions: {args.positions}")
        logger.info(f"Mutations: {args.mutations}")
        logger.info(f"Gatekeeping amino acids: {args.gatekeeping}")

        if not args.agg_only:
            # Generate single mutations
            mutations, region_analyses = mutate_sequence(
                sequence,
                args.positions,
                [m.upper() for m in args.mutations],
                args.regions,
                args.rules,
                args.insert_positions,
                (
                    [aa.upper() for aa in args.insert_aas]
                    if args.insert_aas else None
                ),
                [aa.upper() for aa in args.gatekeeping],
                args.verbose,
                gatekeeper_distance=args.gatekeeper_distance,
                internal_gatekeepers=args.internal_gatekeepers
            )

            logger.info(f"Generated {len(mutations)} mutations")

            # Handle multi-point mutations
            if args.multi_mutations:
                output_base = Path(args.multi_output)
                output_base.mkdir(parents=True, exist_ok=True)

                # Write single mutations
                single_output = output_base / 'single_mutations.fasta'
                write_fasta(
                    str(single_output), header, sequence,
                    mutations, not args.no_original
                )
                logger.info(f"Single mutations written to {single_output}")

                # Generate multi-point mutations
                logger.info(
                    f"Generating multi-point mutations "
                    f"(levels: {args.multi_mutations})"
                )
                try:
                    multi_mutations = generate_multi_point_mutations(
                        mutations,
                        sequence,
                        args.multi_mutations,
                        regions=args.regions,
                        top_variants_per_position=args.multi_top_per_position,
                        n_workers=args.threads
                    )

                    # Categorize
                    categorized = categorize_multi_mutations(
                        multi_mutations, regions=args.regions
                    )

                    # Create output structure
                    create_output_directory(
                        str(output_base), args.multi_mutations
                    )

                    # Write categorized mutations
                    write_multi_mutations_by_category(
                        str(output_base), header, sequence,
                        categorized, include_original=not args.no_original
                    )

                    logger.info(
                        f"Multi-mutation results written to {output_base}"
                    )

                except MemoryError:
                    logger.error(
                        "Out of memory during multi-mutation generation. "
                        "Try reducing --multi-mutations levels or "
                        "--multi-top-per-position"
                    )
                    sys.exit(1)

            else:
                # Single mutations only
                write_fasta(
                    args.output, header, sequence,
                    mutations, not args.no_original
                )
                logger.info(f"Results written to {args.output}")

            print_mutation_summary(mutations)

        else:
            # Aggregation analysis only
            region_analyses = []
            if args.regions:
                for region_str in args.regions:
                    try:
                        start, stop = parse_region(
                            region_str, len(sequence)
                        )
                        analysis = analyze_region(
                            sequence, start, stop,
                            selected_rules=args.rules
                        )
                        region_analyses.append(analysis)
                        logger.info(
                            f"Region {start}:{stop}: "
                            f"{len(analysis['merged_clusters'])} clusters found"
                        )
                    except ValueError as e:
                        logger.error(
                            f"Error analyzing region {region_str}: {e}"
                        )

            logger.info("Aggregation analysis completed")
            print_aggregation_summary(region_analyses, sequence)

        logger.info("=" * 70)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 70)

    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user")
        sys.exit(130)
    except MemoryError:
        logger.error(
            "Out of memory - try reducing --multi-mutations levels "
            "or --multi-top-per-position"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
