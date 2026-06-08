"""
Mutation generation and application for AGGRESSOR pipeline.

Handles:
- Single point mutations (direct, rule-based, insertions)
- Multi-point mutation generation with parallelization
- Mutation type classification
- Sequence manipulation utilities

Architecture:
    mutate_sequence() — main single-mutation generator
    apply_rule_mutations() — rule-based mutation logic
    generate_multi_point_mutations() — combinatorial multi-mutations
    categorize_multi_mutations() — classification of combinations
"""
import re
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations, product
from math import comb, prod
from multiprocessing import cpu_count
from typing import (
    List, Tuple, Dict, Set, Any, Optional
)

from .config import (
    MAX_GAP_FOR_MERGING,
    GATEKEEPER_DISTANCE,
    GATEKEEPING_AAS,
    CANONICAL_GATEKEEPER_AAS,
    MAX_MUTATION_LEVEL,
    PARALLEL_RESULT_THRESHOLD,
    logger
)
from .models import (
    Cluster,
    MultiRuleCluster,
    MutationType,
    MutationInfo,
    MultiMutationResult,
)
from .analysis import (
    analyze_region,
    classify_mutation_type,
)
from .clustering import (
    create_cluster,
    merge_overlapping_clusters_unionfind,
)
from .sequtils import parse_region


# =============================================================================
# SEQUENCE MANIPULATION
# =============================================================================

def create_mutated_sequence(
        sequence: str,
        position: int,
        new_aa: str
) -> str:
    """
    Create a mutated sequence by replacing one amino acid.

    Optimized implementation:
    - For small sequences (<1000 aa): uses list conversion (faster)
    - For large sequences (≥1000 aa): uses string slicing (memory efficient)

    Args:
        sequence: Original protein sequence
        position: 1-indexed position to mutate
        new_aa: Single-letter code of new amino acid

    Returns:
        Mutated sequence string

    Raises:
        ValueError: If position is out of range

    Example:
        >>> create_mutated_sequence("ALVIF", 2, 'P')
        'APVIF'
    """
    if position < 1 or position > len(sequence):
        raise ValueError(
            f"Position {position} out of range (1-{len(sequence)})"
        )

    if len(sequence) < 1000:
        seq_list = list(sequence)
        seq_list[position - 1] = new_aa
        return ''.join(seq_list)
    else:
        return sequence[:position - 1] + new_aa + sequence[position:]


# =============================================================================
# POSITION CLASSIFICATION HELPERS
# =============================================================================

def is_edge_position(
        pos: int,
        cluster_positions: List[int],
        region_start: int,
        region_end: int
) -> bool:
    """
    Check if a position is at the edge of a cluster or region boundary.

    Edge positions receive gatekeeping amino acids in addition to
    standard mutations, as they can disrupt β-sheet extension.

    Args:
        pos: Position to check
        cluster_positions: All positions in the cluster
        region_start: Start of the region (1-indexed)
        region_end: End of the region (1-indexed)

    Returns:
        True if position is at edge of cluster or adjacent to region boundary

    Example:
        >>> is_edge_position(5, [5,6,7,8], 1, 10)
        True
        >>> is_edge_position(6, [5,6,7,8], 1, 10)
        False
    """
    if not cluster_positions:
        return False

    min_pos = min(cluster_positions)
    max_pos = max(cluster_positions)

    # At cluster boundaries
    if pos == min_pos or pos == max_pos:
        return True

    # At region boundaries
    if pos == region_start or pos == region_end:
        return True

    # Adjacent to region boundaries
    if pos == region_start + 1 or pos == region_end - 1:
        return True

    return False


def get_mutations_for_position(
        pos: int,
        cluster_positions: List[int],
        region_start: int,
        region_end: int,
        mutations: List[str],
        gatekeeping_aas: List[str]
) -> List[str]:
    """
    Get list of mutations to apply based on position type.

    Edge positions get both standard mutations AND gatekeeping amino acids.
    Internal positions get only standard mutations.

    Args:
        pos: Position to check
        cluster_positions: All positions in the cluster
        region_start: Region start
        region_end: Region end
        mutations: Standard mutation amino acids
        gatekeeping_aas: Gatekeeping amino acids (for edges only)

    Returns:
        List of amino acid codes to try
    """
    if is_edge_position(pos, cluster_positions, region_start, region_end):
        return list(dict.fromkeys(mutations + gatekeeping_aas))
    return mutations


# =============================================================================
# RULE-BASED MUTATIONS
# =============================================================================

def apply_rule_mutations(
        sequence: str,
        region_analysis: Dict,
        mutations: List[str],
        selected_rules: List[str] = None,
        gatekeeping_aas: List[str] = None,
        gatekeeper_distance: int = GATEKEEPER_DISTANCE,
        internal_gatekeepers: bool = False
) -> List[Tuple[str, str, int, MutationType]]:
    """
    Apply mutations based on aggregation rules with biological classification.

    Algorithm:
    1. Filter rules if specific ones selected
    2. Collect all qualifying clusters
    3. Merge overlapping clusters
    4. For each merged cluster, build the set of target positions:
       - the cluster core positions, plus
       - up to ``gatekeeper_distance`` flanking residues on each side
         (the gatekeeper zone adjacent to the APR), plus
       - optionally the internal gap residues inside a merged cluster
         (the small zones between merged sub-clusters), if
         ``internal_gatekeepers`` is enabled
    5. For each target position:
       - Core interior positions receive standard substitutions
       - Core boundary positions receive standard + gatekeeping substitutions
       - Flanking and internal (gatekeeper-zone) positions receive ONLY
         gatekeeping substitutions
       - Classify mutation type (GATEKEEPER, BETA_CORE, BOUNDARY, FLANKING)

    Args:
        sequence: Full protein sequence
        region_analysis: Analysis results from analyze_region()
        mutations: Standard mutation amino acids
        selected_rules: Optional subset of rules to apply
        gatekeeping_aas: Gatekeeping amino acids for edge/flanking positions
        gatekeeper_distance: Number of residues flanking each APR (on each
            side) that are also mutated as part of the gatekeeper zone.
            Set to 0 to restrict mutations to cluster core positions only.
        internal_gatekeepers: If True, also mutate the internal gap residues
            inside merged clusters (the merge zones between sub-clusters)
            as gatekeeper-zone positions.

    Returns:
        List of (description, sequence, agg_score, mutation_type) tuples
    """
    results = []

    if gatekeeping_aas is None:
        gatekeeping_aas = GATEKEEPING_AAS

    seq_len = len(sequence)

    # Gatekeeper-zone substitutions (flanking + internal): only gatekeeping AAs,
    # order-preserving and deterministic.
    gatekeeper_substitutions = list(dict.fromkeys(gatekeeping_aas))

    # Combine user-specified and canonical gatekeepers
    gatekeeper_aa_set = (
            set(aa.upper() for aa in gatekeeping_aas) |
            CANONICAL_GATEKEEPER_AAS
    )

    # Filter rules if specific ones are selected
    if selected_rules:
        rules_to_apply = {
            k: v for k, v in region_analysis['rules'].items()
            if k in selected_rules
        }
    else:
        rules_to_apply = region_analysis['rules']

    region_start, region_end = region_analysis['region']

    # Collect all clusters from all rules
    cluster_objects = []
    for rule_name, rule_data in rules_to_apply.items():
        if not rule_data['condition_met']:
            continue

        for cluster_dict in rule_data['qualifying_clusters']:
            cluster_objects.append(create_cluster(
                positions=cluster_dict['positions'],
                sequence=sequence,
                rule_name=cluster_dict.get('rule_name', rule_name),
                aggregation_score=cluster_dict.get(
                    'aggregation_score',
                    rule_data['aggregation_score']
                )
            ))

    # Merge overlapping clusters
    merged_clusters = merge_overlapping_clusters_unionfind(
        cluster_objects, sequence, gap_tolerance=MAX_GAP_FOR_MERGING
    )

    # Track mutated positions to avoid duplicates
    mutated_positions: Set[int] = set()

    # Union of all core positions: flanking (gatekeeper-zone) residues of one
    # cluster must never override the core classification of another cluster.
    all_core_positions: Set[int] = set()
    for cluster in merged_clusters:
        all_core_positions.update(cluster.positions)

    # Apply mutations to each merged cluster
    for cluster in merged_clusters:
        core_positions = list(cluster.positions)
        core_set = set(core_positions)

        # Get cluster boundaries
        cluster_min = min(core_positions)
        cluster_max = max(core_positions)

        # Build the flanking (gatekeeper) zone adjacent to the APR
        flanking_positions: Set[int] = set()
        for d in range(1, gatekeeper_distance + 1):
            left = cluster_min - d
            right = cluster_max + d
            if 1 <= left <= seq_len and left not in all_core_positions:
                flanking_positions.add(left)
            if 1 <= right <= seq_len and right not in all_core_positions:
                flanking_positions.add(right)

        # Build the internal gatekeeper zone: gap residues inside the merged
        # cluster span (the small zones between merged sub-clusters).
        internal_positions: Set[int] = set()
        if internal_gatekeepers:
            for p in range(cluster_min + 1, cluster_max):
                if p not in core_set:
                    internal_positions.add(p)

        positions_to_mutate = (
            core_positions
            + sorted(flanking_positions)
            + sorted(internal_positions)
        )

        # Get aggregation score and rule description
        if isinstance(cluster, MultiRuleCluster):
            agg_score = cluster.combined_aggregation_score
            rule_desc = f"MERGED RULES {'+'.join(cluster.rules)}"
        else:
            agg_score = cluster.aggregation_score
            rule_desc = f"Rule '{cluster.rule_name}'"

        for pos in positions_to_mutate:
            if pos in mutated_positions:
                continue

            mutated_positions.add(pos)
            original_aa = sequence[pos - 1]

            in_core = pos in core_set
            is_internal = pos in internal_positions
            is_boundary = (pos == cluster_min or pos == cluster_max)

            # Core boundary positions get standard + gatekeeping; core interior
            # gets standard only; flanking and internal gatekeeper-zone positions
            # get ONLY gatekeeping substitutions.
            if not in_core:
                all_mutations = gatekeeper_substitutions
            elif is_boundary:
                all_mutations = list(dict.fromkeys(mutations + gatekeeping_aas))
            else:
                all_mutations = mutations

            for new_aa in all_mutations:
                if new_aa == original_aa:
                    continue

                mutated_seq = create_mutated_sequence(sequence, pos, new_aa)

                # Classify mutation type. Internal merge-zone gatekeepers may
                # sit far from the outer cluster boundaries, so classify them
                # directly by whether the substitution is a gatekeeper residue.
                if is_internal:
                    mutation_type = (
                        MutationType.GATEKEEPER
                        if new_aa.upper() in gatekeeper_aa_set
                        else MutationType.FLANKING
                    )
                else:
                    mutation_type = classify_mutation_type(
                        position=pos,
                        new_aa=new_aa,
                        cluster_positions=core_positions,
                        cluster_min=cluster_min,
                        cluster_max=cluster_max,
                        gatekeeper_aas=gatekeeper_aa_set,
                        gatekeeper_distance=gatekeeper_distance
                    )

                # Build description with type label
                type_label = mutation_type.name
                description = (
                    f"{original_aa}{pos}{new_aa} | {rule_desc} | "
                    f"{type_label} (agg_score={agg_score})"
                )

                results.append(
                    (description, mutated_seq, agg_score, mutation_type)
                )

    return results


# =============================================================================
# MAIN MUTATION GENERATOR
# =============================================================================

def mutate_sequence(
        sequence: str,
        positions: List[int],
        mutations: List[str],
        regions: List[str] = None,
        selected_rules: List[str] = None,
        insertion_positions: List[int] = None,
        insertion_aas: List[str] = None,
        gatekeeping_aas: List[str] = None,
        verbose: bool = False,
        gatekeeper_distance: int = GATEKEEPER_DISTANCE,
        internal_gatekeepers: bool = False
) -> Tuple[List[Tuple[str, str]], List[Dict]]:
    """
    Generate mutated sequences with point mutations, rule-based mutations, and insertions.

    Mutation types generated:
    - Direct point mutations at specified positions
    - Rule-based mutations in aggregation-prone regions
    - Insertions at specified positions
    - Edge positions receive additional gatekeeping mutations

    All mutations are classified by structural context:
    BETA_CORE, GATEKEEPER, BOUNDARY, FLANKING, DIRECT, INSERTION

    Args:
        sequence: Original protein sequence
        positions: List of 1-indexed positions for direct mutations
        mutations: Amino acid codes to mutate to
        regions: List of region strings (e.g., ["10:50", "60:100"])
        selected_rules: Optional subset of rules to apply
        insertion_positions: Positions for insertions (before this position)
        insertion_aas: Amino acids to insert
        gatekeeping_aas: Gatekeeping amino acids for edge positions
        verbose: Enable debug logging
        gatekeeper_distance: Number of flanking residues adjacent to each APR
            to also mutate as part of the gatekeeper zone (0 = core only)
        internal_gatekeepers: If True, also mutate internal merge-zone residues
            (gaps between merged sub-clusters) as gatekeepers

    Returns:
        Tuple of:
        - List of (description, sequence) tuples sorted by aggregation score
        - List of region analysis dictionaries

    Raises:
        ValueError: If positions are out of range or insertion args mismatch
    """
    all_results = []
    region_analyses = []
    seq_len = len(sequence)

    if gatekeeping_aas is None:
        gatekeeping_aas = GATEKEEPING_AAS

    # Validate positions
    for pos in positions:
        if pos < 1 or pos > seq_len:
            raise ValueError(
                f"Position {pos} is out of range "
                f"(sequence length: {seq_len})"
            )

    # ============ RULE-BASED MUTATIONS ============

    if regions:
        for region_str in regions:
            start, stop = parse_region(region_str, seq_len)
            analysis = analyze_region(
                sequence, start, stop, selected_rules=selected_rules
            )
            region_analyses.append(analysis)

            if verbose:
                logger.debug(f"Analyzing region {start}:{stop}")
                logger.debug(f"Sequence: {analysis['sequence']}")
                logger.debug(f"Length: {analysis['length']} residues")

                for rule_name, rule_data in analysis['rules'].items():
                    if rule_data['condition_met']:
                        logger.debug(
                            f"  {rule_name}: "
                            f"{len(rule_data['qualifying_clusters'])} clusters"
                        )

            # Generate rule-based mutations
            rule_mutations = apply_rule_mutations(
                sequence, analysis, mutations,
                selected_rules, gatekeeping_aas,
                gatekeeper_distance=gatekeeper_distance,
                internal_gatekeepers=internal_gatekeepers
            )
            all_results.extend(rule_mutations)

            if verbose:
                logger.debug(
                    f"Generated {len(rule_mutations)} rule-based mutations "
                    f"for region {start}:{stop}"
                )

    # ============ DIRECT POINT MUTATIONS ============

    for pos in positions:
        original_aa = sequence[pos - 1]

        for new_aa in mutations:
            if new_aa == original_aa:
                continue

            mutated_seq = create_mutated_sequence(sequence, pos, new_aa)
            description = (
                f"{original_aa}{pos}{new_aa} | "
                f"Direct mutation | DIRECT (agg_score=0)"
            )
            all_results.append(
                (description, mutated_seq, 0, MutationType.DIRECT)
            )

    # ============ INSERTIONS ============

    if insertion_positions and insertion_aas:
        if len(insertion_positions) != len(insertion_aas):
            raise ValueError(
                "insertion_positions and insertion_aas "
                "must have the same length"
            )

        for ins_pos, ins_aa in zip(insertion_positions, insertion_aas):
            if ins_pos < 1 or ins_pos > seq_len + 1:
                raise ValueError(
                    f"Insertion position {ins_pos} is out of range "
                    f"(1-{seq_len + 1})"
                )

            mutated_seq = (
                    sequence[:ins_pos - 1] + ins_aa + sequence[ins_pos - 1:]
            )
            description = (
                f"Insertion: {ins_aa} inserted before position {ins_pos} | "
                f"INSERTION (agg_score=0)"
            )
            all_results.append(
                (description, mutated_seq, 0, MutationType.INSERTION)
            )

    # ============ SORT AND FORMAT ============

    # Sort by aggregation score (descending)
    all_results.sort(key=lambda x: x[2], reverse=True)

    # Convert to expected format (description, sequence)
    sorted_results = [(desc, seq) for desc, seq, _, _ in all_results]

    return sorted_results, region_analyses


# =============================================================================
# MULTI-POINT MUTATION GENERATION
# =============================================================================

def extract_mutation_info(
        description: str,
        sequence: str = "",
        regions: Optional[List[str]] = None
) -> MutationInfo:
    """
    Extract structured mutation information from a description string.

    Uses explicit, anchored patterns so aggregation scores are never
    mistaken for positions, and returns a typed :class:`MutationInfo`
    capturing position, residues, score, rule, type, and region.

    Args:
        description: FASTA-style mutation description
        sequence: Full mutated sequence (stored for downstream use)
        regions: Optional region strings used to assign a region

    Returns:
        A populated MutationInfo (with safe defaults if unparseable)
    """
    # OriginalAA + Position + NewAA at start of the description
    mutation_pattern = re.compile(r'^([A-Z])(\d+)([A-Z])\s*\|')
    insertion_pattern = re.compile(
        r'Insertion:\s*([A-Z])\s+inserted\s+before\s+position\s+(\d+)'
    )
    score_pattern = re.compile(r'agg_score=(\d+)')
    rule_pattern = re.compile(r"Rule\s+'([^']+)'")
    merged_pattern = re.compile(r'MERGED RULES\s+([\w+]+)')

    position = None
    original_aa = None
    new_aa = None
    agg_score = 0
    rule_name = None
    mutation_type = MutationType.UNKNOWN

    match = mutation_pattern.match(description)
    if match:
        original_aa = match.group(1)
        position = int(match.group(2))
        new_aa = match.group(3)
    else:
        ins_match = insertion_pattern.search(description)
        if ins_match:
            new_aa = ins_match.group(1)
            position = int(ins_match.group(2))
            original_aa = '-'
            mutation_type = MutationType.INSERTION

    score_match = score_pattern.search(description)
    if score_match:
        agg_score = int(score_match.group(1))

    rule_match = rule_pattern.search(description)
    if rule_match:
        rule_name = rule_match.group(1)
    else:
        merged_match = merged_pattern.search(description)
        if merged_match:
            rule_name = merged_match.group(1)

    if mutation_type == MutationType.UNKNOWN:
        mutation_type = MutationType.from_description(description)

    region = None
    if regions and position:
        for region_str in regions:
            try:
                start, stop = map(int, region_str.split(':'))
                if start <= position <= stop:
                    region = region_str
                    break
            except ValueError:
                continue

    return MutationInfo(
        position=position or 0,
        original_aa=original_aa or '?',
        new_aa=new_aa or '?',
        mutation_type=mutation_type,
        aggregation_score=agg_score,
        region=region,
        rule_name=rule_name,
        description=description,
        sequence=sequence,
    )


def _parse_mutation_infos(
        mutations: List[Tuple[str, str]],
        sequence: str,
        regions: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Parse mutation descriptions into structured info dictionaries
    for multi-point combination.

    Only point mutations (OriginalAA-Position-NewAA) are retained, since
    insertions cannot be combined positionally.

    Args:
        mutations: List of (description, sequence) tuples
        sequence: Original protein sequence
        regions: Optional list of region strings

    Returns:
        List of structured mutation info dicts
    """
    mutation_infos = []

    for desc, seq in mutations:
        info = extract_mutation_info(desc, seq, regions)
        if not info.position or not seq:
            continue

        # Only point mutations can be combined
        match = re.search(r'([A-Z])(\d+)([A-Z])', desc)
        if not match:
            continue

        mutation_infos.append({
            'position': info.position,
            'mutation_type': info.mutation_type,
            'is_gatekeeper': info.is_gatekeeper,
            'region': info.region,
            'agg_score': info.aggregation_score,
            'description': desc,
            'sequence': seq,
            'original_aa': (
                sequence[info.position - 1]
                if info.position <= len(sequence) else '?'
            ),
            'new_aa': match.group(3),
        })

    return mutation_infos


def _filter_by_position(
        mutation_infos: List[Dict[str, Any]],
        top_variants_per_position: int = 3
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Group and filter mutations by position, keeping top variants.

    Args:
        mutation_infos: List of structured mutation info dicts
        top_variants_per_position: Max variants to keep per position

    Returns:
        Dict mapping position to list of best variant dicts
    """
    by_position: Dict[int, List[Dict[str, Any]]] = {}

    for mi in mutation_infos:
        by_position.setdefault(mi['position'], []).append(mi)

    # Deduplicate and keep best per new_aa
    filtered: Dict[int, List[Dict[str, Any]]] = {}

    for pos, variants in by_position.items():
        best_by_newaa: Dict[str, Dict[str, Any]] = {}

        for v in variants:
            new_aa = v.get('new_aa')
            if not new_aa:
                continue
            existing = best_by_newaa.get(new_aa)
            if (
                    existing is None or
                    v.get('agg_score', 0) > existing.get('agg_score', 0)
            ):
                best_by_newaa[new_aa] = v

        kept = list(best_by_newaa.values())
        kept.sort(key=lambda x: x.get('agg_score', 0), reverse=True)

        if top_variants_per_position and top_variants_per_position > 0:
            kept = kept[:top_variants_per_position]

        filtered[pos] = kept

    return filtered


def _create_multi_mutation_result(
        variant_combo: Tuple[Dict[str, Any], ...],
        sequence: str,
        level: int,
        regions: Optional[List[str]]
) -> MultiMutationResult:
    """
    Create a MultiMutationResult from a variant combination.

    Args:
        variant_combo: Tuple of mutation variant dicts
        sequence: Original sequence
        level: Number of simultaneous mutations
        regions: Optional region list

    Returns:
        MultiMutationResult object
    """
    positions = [v['position'] for v in variant_combo]
    new_aas = tuple(v['new_aa'] for v in variant_combo)
    original_aas = tuple(v['original_aa'] for v in variant_combo)
    mutation_types = tuple(
        v.get('mutation_type', MutationType.UNKNOWN)
        for v in variant_combo
    )

    # Apply mutations (reverse order to maintain position indices)
    mutations_to_apply = sorted(
        zip(positions, new_aas), reverse=True, key=lambda x: x[0]
    )
    combined_seq = sequence
    for pos, aa in mutations_to_apply:
        combined_seq = create_mutated_sequence(combined_seq, pos, aa)

    combined_agg_score = sum(v.get('agg_score', 0) for v in variant_combo)

    # Build description
    mutations_str = ' + '.join(
        f"{v['original_aa']}{v['position']}{v['new_aa']}"
        for v in sorted(variant_combo, key=lambda x: x['position'])
    )

    regions_involved = sorted({
        v['region'] for v in variant_combo if v.get('region')
    })
    regions_str = ', '.join(regions_involved) if regions_involved else "Direct"

    type_composition = tuple(
        mt.name if isinstance(mt, MutationType) else str(mt)
        for mt in mutation_types
    )

    result = MultiMutationResult(
        description="",  # Will be set below
        sequence=combined_seq,
        aggregation_score=combined_agg_score,
        positions=positions,
        regions=regions_involved,
        level=level,
        mutation_types=mutation_types,
        type_composition=type_composition,
        original_aas=original_aas,
        new_aas=new_aas,
    )

    # Build full description
    description = (
        f"{mutations_str} | {regions_str} "
        f"(level={level}, agg_score={combined_agg_score})"
    )
    if result.is_all_gatekeeper:
        description += " | ALL_GATEKEEPER"
    elif result.is_all_core:
        description += " | ALL_CORE"
    elif result.is_mixed:
        description += " | MIXED"

    # MultiMutationResult is a mutable dataclass, so assign directly.
    result.description = description

    return result


def _process_position_chunk(
        position_combos: List[Tuple[int, ...]],
        filtered_by_position: Dict[int, List[Dict]],
        sequence: str,
        level: int,
        regions: Optional[List[str]],
        max_per_chunk: int
) -> List[Dict[str, Any]]:
    """
    Process a chunk of position combinations (worker for parallel processing).

    Args:
        position_combos: Position combinations to process
        filtered_by_position: Filtered variants per position
        sequence: Original sequence
        level: Mutation level
        regions: Optional region list
        max_per_chunk: Maximum results per chunk

    Returns:
        List of mutation result dicts
    """
    results = []
    count = 0

    for pos_combo in position_combos:
        if count >= max_per_chunk:
            break

        variant_lists = [filtered_by_position[p] for p in pos_combo]

        for variant_combo in product(*variant_lists):
            if count >= max_per_chunk:
                break

            result = _create_multi_mutation_result(
                variant_combo, sequence, level, regions
            )
            results.append(result.to_dict())
            count += 1

    return results


def generate_multi_point_mutations(
        mutations: List[Tuple[str, str]],
        sequence: str,
        multi_mutation_levels: Optional[List[int]] = None,
        regions: Optional[List[str]] = None,
        max_combinations: int = 10000,
        top_variants_per_position: int = 3,
        n_workers: Optional[int] = None
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Generate multi-point mutations from single mutations.

    Supports parallel processing for large combination spaces.

    Args:
        mutations: List of (description, sequence) tuples
        sequence: Original protein sequence
        multi_mutation_levels: Levels to generate (e.g., [2, 3])
        regions: List of region strings for categorization
        max_combinations: Maximum combinations per level
        top_variants_per_position: Limit variants per position
        n_workers: Number of parallel workers (None = auto)

    Returns:
        Dict mapping level to list of mutation result dicts
    """
    if not multi_mutation_levels:
        logger.warning(
            "No multi-mutation levels specified; returning empty results"
        )
        return {}

    # Validate levels
    valid_levels = []
    for level in multi_mutation_levels:
        if not isinstance(level, int):
            logger.warning(f"Skipping non-integer level: {level}")
            continue
        if level < 2:
            logger.warning(
                f"Skipping invalid level {level} (must be ≥2)"
            )
            continue
        if level > MAX_MUTATION_LEVEL:
            logger.warning(
                f"Skipping level {level} "
                f"(exceeds maximum {MAX_MUTATION_LEVEL})"
            )
            continue
        valid_levels.append(level)

    if not valid_levels:
        logger.warning("No valid multi-mutation levels after filtering")
        return {}

    # Parse mutation info
    mutation_infos = _parse_mutation_infos(mutations, sequence, regions)
    filtered_by_position = _filter_by_position(
        mutation_infos, top_variants_per_position
    )
    viable_positions = sorted([
        p for p, vs in filtered_by_position.items() if vs
    ])

    if not viable_positions:
        logger.warning("No viable positions for multi-mutations")
        return {}

    # Determine worker count
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    multi_results: Dict[int, List[Dict[str, Any]]] = {}

    for level in valid_levels:
        if level > len(viable_positions):
            logger.warning(
                f"Skipping level {level} "
                f"(only {len(viable_positions)} positions available)"
            )
            continue

        total_combos = comb(len(viable_positions), level)
        logger.info(
            f"Level {level}: up to {total_combos} position combinations"
        )

        # Deterministically select the prefix of position combinations whose
        # cumulative variant expansion reaches max_combinations. Choosing whole
        # combinations up front makes the result set fully reproducible and
        # independent of the worker count: parallelism only speeds up the
        # expansion, it never changes *which* combinations are kept.
        selected_combos: List[Tuple[int, ...]] = []
        cumulative = 0
        for combo in combinations(viable_positions, level):
            selected_combos.append(combo)
            cumulative += prod(len(filtered_by_position[p]) for p in combo)
            if cumulative >= max_combinations:
                break

        if not selected_combos:
            multi_results[level] = []
            continue

        # Parallelize only when the workload is large enough to outweigh the
        # process/pickling overhead. The output is identical either way.
        use_parallel = (
            n_workers > 1
            and len(selected_combos) > 1
            and cumulative > PARALLEL_RESULT_THRESHOLD
        )

        all_results: List[Dict[str, Any]] = []
        if use_parallel:
            n_chunks = min(n_workers, len(selected_combos))
            chunk_size = (len(selected_combos) + n_chunks - 1) // n_chunks
            chunks = [
                selected_combos[i:i + chunk_size]
                for i in range(0, len(selected_combos), chunk_size)
            ]

            try:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    # Submit chunks and collect results in submission order
                    # (NOT completion order) so concatenation is deterministic.
                    futures = [
                        executor.submit(
                            _process_position_chunk,
                            chunk, filtered_by_position,
                            sequence, level, regions, max_combinations
                        )
                        for chunk in chunks
                    ]
                    for future in futures:
                        all_results.extend(future.result())
            except Exception as e:
                logger.warning(
                    f"Parallel processing failed, "
                    f"falling back to sequential: {e}"
                )
                all_results = _process_position_chunk(
                    selected_combos, filtered_by_position,
                    sequence, level, regions, max_combinations
                )
        else:
            all_results = _process_position_chunk(
                selected_combos, filtered_by_position,
                sequence, level, regions, max_combinations
            )

        # Keep the deterministic canonical prefix, then order by a total key:
        # highest aggregation score first, description as a stable tie-break.
        all_results = all_results[:max_combinations]
        all_results.sort(
            key=lambda x: (-x.get('agg_score', 0), x.get('description', ''))
        )
        multi_results[level] = all_results

        logger.info(
            f"Level {level}: Generated "
            f"{len(multi_results[level])} mutations"
        )

    return multi_results


def categorize_multi_mutations(
        multi_mutations: Dict[int, List[Dict[str, Any]]],
        regions: Optional[List[str]] = None
) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
    """
    Categorize multi-mutations by region and mutation type.

    Categories:
    - single_region: All positions in same region
    - multi_region: Positions span different regions
    - all_gatekeeper: Only gatekeeper mutations
    - all_core: Only β-core mutations
    - mixed: Combination of different types

    Args:
        multi_mutations: Results from generate_multi_point_mutations()
        regions: Optional region list

    Returns:
        Nested dict: level → category → list of mutation dicts
    """
    categorized = {}

    for level, mutations in multi_mutations.items():
        categorized[level] = {
            'single_region': [],
            'multi_region': [],
            'all_gatekeeper': [],
            'all_core': [],
            'mixed': [],
        }

        for item in mutations:
            is_all_gatekeeper = item.get('is_all_gatekeeper', False)
            is_all_core = item.get('is_all_core', False)
            regions_involved = item.get('regions', [])

            # Categorize by mutation type
            if is_all_gatekeeper:
                categorized[level]['all_gatekeeper'].append(item)
            elif is_all_core:
                categorized[level]['all_core'].append(item)
            else:
                categorized[level]['mixed'].append(item)

            # Categorize by region
            if not regions_involved or len(set(regions_involved)) == 1:
                categorized[level]['single_region'].append(item)
            else:
                item['region_types'] = tuple(regions_involved)
                categorized[level]['multi_region'].append(item)

    return categorized

