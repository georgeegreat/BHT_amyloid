"""
Human-readable reporting for the AGGRESSOR pipeline.

Contains all console output formatting: usage examples, detailed help,
the single-mutation summary, and the aggregation-analysis summary.
Keeping presentation here separates "what to print" from the CLI
orchestration and the on-disk I/O.
"""
import argparse
from typing import Dict, List, Tuple

from .config import MAX_MUTATION_LEVEL
from .models import MutationType
from .rules import RULES


# =============================================================================
# HELP AND USAGE DISPLAY
# =============================================================================

def print_usage_example() -> None:
    """Print detailed usage examples."""
    example = """
INVOCATION:
=========================================================
After installation (./install.sh or pip install .):
   aggressor protein.fasta [options]
Equivalent alternatives without installing:
   python -m aggressor protein.fasta [options]
   python aggressor/main.py protein.fasta [options]

USAGE EXAMPLES:
=========================================================

1. Rule-based mutagenesis in specific regions:
   aggressor protein.fasta --regions 10:20 30:40 50:60

2. Rule-based with custom mutations:
   aggressor protein.fasta --regions 5:15 -m A D E

3. Rule-based with specific rules only:
   aggressor protein.fasta --regions 10:30 --rules hydrophobic_aliphatic aromatic

4. Combined approach (rules + specific positions):
   aggressor protein.fasta --regions 10:20 --positions 15 25 --mutations P G

5. With insertions and rule-based:
   aggressor protein.fasta --regions 5:15 --insert-positions 10 --insert-aas K

6. With gatekeeping amino acids (only for edge positions):
   aggressor protein.fasta --regions 10:20 --gatekeeping Y K

7. Tune the gatekeeper zone and include internal merge-zone gatekeepers:
   aggressor protein.fasta --regions 10:40 --gatekeeper-distance 3 --internal-gatekeepers

8. Detailed verbose output:
   aggressor protein.fasta --regions 10:20 -v

9. Analyze the entire sequence (all residues):
   aggressor protein.fasta --regions all

10. Generate double and triple mutations:
    aggressor protein.fasta --regions 10:30 --multi-mutations 2 3

11. Multi-mutations with parallel processing:
    aggressor protein.fasta --regions 10:30 --multi-mutations 2 3 --threads 4

AVAILABLE RULES:
=========================================================
• hydrophobic_aliphatic    : Triggers if ≥3 V, I, L, A, M residues within 4 positions
• aromatic                 : Triggers if ≥2 F, Y, W residues within 3 positions
• amide                    : Triggers if ≥2 Q, N residues within 3 positions
• hydrophobic_and_aromatic : Triggers if ≥2 hydrophobic-aromatic adjacent pairs 
                             OR 1 pair + at least 1 hydrophobic within 3 positions

MUTATION TYPE CLASSIFICATION:
=========================================================
• BETA_CORE  : Within identified aggregation cluster (highest risk)
• GATEKEEPER : At cluster boundary with gatekeeper AA (P, K, R, D, E)
• BOUNDARY   : At cluster boundary but not gatekeeper AA
• FLANKING   : Adjacent to APR but outside gatekeeper zone
• DIRECT     : User-specified position (no rule context)
• INSERTION  : Amino acid insertion

AGGREGATION SCORE RANKING (highest to lowest):
1. hydrophobic_aliphatic: 3
2. hydrophobic_and_aromatic: 2
3. aromatic: 2
4. amide: 1

REQUIRED PARAMETERS:
=========================================================
• input_file    : Input FASTA file (multifasta not supported)
• Either --positions OR --regions must be specified
"""
    print(example)


def print_help_info(parser: argparse.ArgumentParser) -> None:
    """
    Print detailed help information including examples and rule descriptions.

    Args:
        parser: Configured ArgumentParser
    """
    print("=" * 70)
    print("AGGRESSOR: AGGREGATION-GUIDED IN SILICO MUTAGENESIS")
    print("WITH MULTI-POINT MUTATION SUPPORT")
    print("=" * 70)
    print("\nDESCRIPTION:")
    print("Performs rule-based mutagenesis on protein sequences.")
    print("Rules apply ONLY when amino acids are clustered together.")
    print("Multiple rules can apply simultaneously to the same motif.")
    print("Supports generation of multi-point mutations (double, triple, etc.)")
    print("\n" + "=" * 70)

    print_usage_example()
    parser.print_help()

    print("\n" + "=" * 70)
    print("MULTI-POINT MUTATION FEATURES:")
    print("=" * 70)
    print("\nWhen --multi-mutations is specified, the script generates mutations at multiple")
    print("levels and organizes them in a directory structure.")
    print("\nCombinatorics control:")
    print("  • --multi-top-per-position limits variants per position (ranked by agg_score)")
    print("  • --threads enables parallel processing for large combination spaces")
    print(f"  • Maximum mutation level is {MAX_MUTATION_LEVEL}")
    print("\nOutput structure:")
    print("  mutated_sequences/")
    print("  ├── single_mutations.fasta")
    print("  ├── double_mutations/")
    print("  │   ├── single_region.fasta")
    print("  │   ├── multi_region.fasta")
    print("  │   ├── all_gatekeeper.fasta")
    print("  │   ├── all_core.fasta")
    print("  │   └── mixed.fasta")
    print("  └── triple_mutations/")
    print("      └── ...")

    print("\n" + "=" * 70)
    print("RULE DETAILS:")
    print("=" * 70)
    for rule_name, rule in RULES.items():
        print(f"\n{rule_name}:")
        print(f"  {rule['description']}")
        if rule_name == 'hydrophobic_and_aromatic':
            print(f"  Conditions:")
            print(f"    1. At least 2 hydrophobic-aromatic adjacent pairs")
            print(f"    2. OR 1 pair + at least 1 hydrophobic within 3 positions")
        else:
            print(f"  Residues: {', '.join(sorted(rule['residues']))}")
            print(f"  Min cluster size: {rule['min_cluster_size']}")
            print(f"  Max gap: {rule['max_gap']} positions")
        print(f"  Aggregation score: {rule['aggregation_score']}")

    print("\n" + "=" * 70)
    print("BIOLOGICAL REFERENCES:")
    print("=" * 70)
    print("• Rousseau et al., J Mol Biol 2006 - Gatekeeper hypothesis")
    print("• Beerten et al., FEBS Lett 2012 - APR boundary effects")
    print("• Tartaglia et al., J Mol Biol 2008 - Aggregation propensity scale")
    print("=" * 70)


# =============================================================================
# RESULT SUMMARIES
# =============================================================================

def print_mutation_summary(mutations: List[Tuple[str, str]]) -> None:
    """
    Print summary of generated mutations with statistics.

    Args:
        mutations: List of (description, sequence) tuples
    """
    if not mutations:
        print("\nNo mutations generated. Check your criteria.")
        return

    print(f"\n{'=' * 70}")
    print("MUTATION SUMMARY (Sorted by aggregation score)")
    print(f"{'=' * 70}")

    # Extract aggregation scores
    mutation_data = []
    for desc, seq in mutations:
        agg_score = 0
        if "(agg_score=" in desc:
            try:
                agg_str = desc.split("(agg_score=")[1].split(")")[0]
                agg_score = int(agg_str)
            except (IndexError, ValueError):
                agg_score = 0
        mutation_data.append((desc, seq, agg_score))

    # Count by mutation type
    type_counts = {mt.name: 0 for mt in MutationType}
    for desc, _, _ in mutation_data:
        for mt in MutationType:
            if mt.name in desc:
                type_counts[mt.name] += 1
                break

    print(f"\nMutation Type Breakdown:")
    for mt_name, count in type_counts.items():
        if count > 0:
            print(f"  • {mt_name}: {count}")

    print(f"\nTOTAL: {len(mutations)}")

    # Show top mutations
    if mutation_data:
        print(f"\nTOP 5 MUTATIONS BY AGGREGATION SCORE:")
        for i, (desc, _, agg_score) in enumerate(mutation_data[:5], 1):
            if len(desc) > 80:
                desc = desc[:77] + "..."
            print(f"{i}. {desc}")


def print_aggregation_summary(
        region_analyses: List[Dict],
        sequence: str
) -> None:
    """
    Print summary of aggregation analysis results.

    Args:
        region_analyses: List of region analysis dictionaries
        sequence: Full protein sequence
    """
    if not region_analyses:
        print("\n" + "=" * 70)
        print("AGGREGATION ANALYSIS RESULTS")
        print("=" * 70)
        print("\nNo regions analyzed. Specify regions with --regions")
        return

    print("\n" + "=" * 70)
    print("AGGREGATION ANALYSIS RESULTS")
    print("=" * 70)

    total_hotspots = 0
    total_multi_rule = 0
    all_hotspot_positions = set()

    for analysis in region_analyses:
        region_start, region_end = analysis['region']
        clusters = analysis['merged_clusters']
        multi_rule = analysis['multi_rule_clusters']
        hotspots = analysis['aggregation_hotspots']

        print(
            f"\nREGION {region_start}:{region_end} "
            f"({region_end - region_start + 1} residues)"
        )
        print("-" * 70)
        print(f"Sequence: {analysis['sequence']}")
        print(f"Total clusters found: {len(clusters)}")
        print(
            f"Aggregation hotspot positions: "
            f"{', '.join(map(str, hotspots)) if hotspots else 'None'}"
        )

        # Rule-by-rule breakdown
        print(f"\nRule Breakdown:")
        for rule_name, rule_data in analysis['rules'].items():
            if rule_data['condition_met']:
                num_clusters = len(rule_data['qualifying_clusters'])
                positions = rule_data['matching_positions']
                print(
                    f"  • {rule_name}: {num_clusters} cluster(s) "
                    f"at positions {positions}"
                )

        # Highlight multi-rule clusters
        if multi_rule:
            print(f"\n⚠️  HIGH AGGREGATION RISK (Multi-Rule Clusters):")
            for i, cluster in enumerate(multi_rule, 1):
                positions = cluster['positions']
                residues = ''.join(cluster['residues'])
                rules = cluster['rules']
                score = cluster['combined_aggregation_score']
                print(f"  Multi-Rule Cluster {i}:")
                print(f"    Positions: {positions}")
                print(f"    Residues:  {residues}")
                print(f"    Converging Rules: {', '.join(rules)}")
                print(f"    Combined Aggregation Score: {score}/8")

        total_hotspots += len(hotspots)
        total_multi_rule += len(multi_rule)
        all_hotspot_positions.update(hotspots)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total regions analyzed: {len(region_analyses)}")
    print(f"Total aggregation hotspot positions: {total_hotspots}")
    print(f"Total multi-rule clusters (highest risk): {total_multi_rule}")

    if total_multi_rule > 0:
        print(f"\n⚠️  ATTENTION: {total_multi_rule} high-risk region(s) detected!")

    if total_hotspots > 0:
        print(
            f"\nRecommended mutation targets: "
            f"{', '.join(map(str, sorted(all_hotspot_positions)))}"
        )
    else:
        print("\nNo aggregation-prone hotspots detected in specified regions.")

    print("=" * 70)
