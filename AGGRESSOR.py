#!/usr/bin/env python3
"""
In Silico Mutagenesis Script with Rule-Based Mutations
Performs point mutations and insertions at specified positions in protein sequences
with rule-based mutagenesis in specified regions

Rules apply only when amino acids from the rule are clustered together
Multiple rules can apply simultaneously to the same motif

Usage: python AGGRESSOR.py <input_file> [options]
"""
import argparse
import re
import sys
import logging
from dataclasses import dataclass, field
from typing import (
    List, Tuple, Dict, Set, Any, Union, Optional,
    FrozenSet, Protocol, runtime_checkable
)
from functools import cached_property
from pathlib import Path
from itertools import combinations, product

# Defining constants
VALID_AAS = set('ACDEFGHIKLMNPQRSTVWY')
FASTA_LINE_LENGTH = 60
MAX_GAP_FOR_MERGING = 2

# Default mutation list - customizable here
DEFAULT_MUTATIONS = ['P', 'G', 'D', 'K']

# Gatekeeping amino acids (only applied to edge positions)
GATEKEEPING_AAS = ['Y']

# Empirically-derived aggregation propensity values
# Source: Tartaglia et al., J Mol Biol 2008
AGGREGATION_PROPENSITY: Dict[str, float] = {
    'I': 1.822,  'V': 1.594,  'L': 1.380,  'F': 1.376,
    'Y': 0.888,  'W': 0.893,  'M': 0.739,  'A': 0.411,
    'C': 0.382,  'T': 0.039,  'S': -0.228, 'G': -0.535,
    'N': -0.547, 'Q': -0.691, 'H': -0.731, 'P': -1.402,
    'D': -1.836, 'E': -1.892, 'K': -2.030, 'R': -1.814,
}

# Define Module-level logger
logger = logging.getLogger(__name__)

def setup_logging(
    verbose: bool = False,
    log_file: Optional[str] = None
) -> None:
    """
    Configure logging for the mutagenesis pipeline.
    
    Args:
        verbose: If True, show DEBUG level messages
        log_file: Optional path to write detailed logs
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)s | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


# Data classes for cluster structures
@dataclass(frozen=True, slots=True)
class Cluster:
    """
    Immutable representation of an aggregation-prone cluster.
    
    Using frozen=True enables:
    - Hashability (can be used in sets, as dict keys)
    - Thread safety (no mutation possible)
    - Semantic clarity (clusters are identified, not modified)
    
    Using slots=True reduces memory footprint by ~40% per instance.
    
    Attributes:
        positions: 1-indexed positions of residues in the cluster (immutable tuple)
        residues: Amino acids at each position (immutable tuple)
        rule_name: Name of the rule that identified this cluster
        aggregation_score: Integer score indicating aggregation propensity
    """
    positions: Tuple[int, ...]
    residues: Tuple[str, ...]
    rule_name: str
    aggregation_score: int
    
    def __post_init__(self):
        """Validate cluster invariants after initialization."""
        if len(self.positions) != len(self.residues):
            raise ValueError(
                f"Position count ({len(self.positions)}) must match "
                f"residue count ({len(self.residues)})"
            )
        if not self.positions:
            raise ValueError("Cluster must contain at least one position")
        
        # Validate positions are sorted and 1-indexed
        if any(p < 1 for p in self.positions):
            raise ValueError("Positions must be 1-indexed (≥1)")
        if list(self.positions) != sorted(self.positions):
            # For frozen dataclass, we can't fix this—must raise error
            raise ValueError("Positions must be in sorted order")
    
    @property
    def size(self) -> int:
        """Number of residues in the cluster."""
        return len(self.positions)
    
    @property
    def span(self) -> int:
        """Sequence span from first to last position (inclusive)."""
        return max(self.positions) - min(self.positions) + 1
    
    @property
    def density(self) -> float:
        """
        Cluster density: occupied positions / total span.
        
        Higher density correlates with stronger aggregation propensity
        as residues are more tightly packed in the primary sequence.
        """
        return self.size / self.span if self.span > 0 else 0.0
    
    @property
    def motif(self) -> str:
        """The cluster as a contiguous sequence string."""
        return ''.join(self.residues)
    
    @property
    def mean_propensity(self) -> float:
        """Average intrinsic aggregation propensity of cluster residues."""
        propensities = [AGGREGATION_PROPENSITY.get(aa, 0.0) for aa in self.residues]
        return sum(propensities) / len(propensities) if propensities else 0.0
    
    def overlaps_with(self, other: 'Cluster', gap_tolerance: int = 0) -> bool:
        """
        Check if this cluster overlaps with another.
        
        Args:
            other: Another cluster to check against
            gap_tolerance: Maximum gap between clusters to consider as "overlapping"
        
        Returns:
            True if clusters overlap or are within gap_tolerance of each other
        """
        self_min, self_max = min(self.positions), max(self.positions)
        other_min, other_max = min(other.positions), max(other.positions)
        
        # No overlap if one ends before the other starts (accounting for gap)
        return not (
            self_max + gap_tolerance < other_min or
            other_max + gap_tolerance < self_min
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            'positions': list(self.positions),
            'residues': list(self.residues),
            'size': self.size,
            'span': self.span,
            'rule_name': self.rule_name,
            'aggregation_score': self.aggregation_score,
            'density': self.density,
            'mean_propensity': self.mean_propensity,
        }


@dataclass(frozen=True, slots=True)
class HydrophobicAromaticCluster:
    """
    Specialized cluster for hydrophobic-aromatic interaction patterns.
    """
    positions: Tuple[int, ...]
    residues: Tuple[str, ...]
    rule_name: str
    aggregation_score: int
    pair_count: int = 0
    condition: str = ""  # 'at_least_2_pairs' or '1_pair_plus_hydrophobic'
    hydrophobic_count: int = 0
    aromatic_count: int = 0
    pair_interactions: Tuple[str, ...] = ()  # e.g., ("V10-F11", "L13-Y14")
    nearby_hydrophobic_positions: Tuple[int, ...] = ()
    
    @property
    def size(self) -> int:
        return len(self.positions)
    
    @property
    def span(self) -> int:
        return max(self.positions) - min(self.positions) + 1 if self.positions else 0
    
    def overlaps_with(self, other: 'Cluster', gap_tolerance: int = 0) -> bool:
        """Check overlap with another cluster."""
        self_min, self_max = min(self.positions), max(self.positions)
        other_min, other_max = min(other.positions), max(other.positions)
        return not (
            self_max + gap_tolerance < other_min or
            other_max + gap_tolerance < self_min
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'positions': list(self.positions),
            'residues': list(self.residues),
            'size': self.size,
            'span': self.span,
            'rule_name': self.rule_name,
            'aggregation_score': self.aggregation_score,
            'pair_count': self.pair_count,
            'condition': self.condition,
            'hydrophobic_count': self.hydrophobic_count,
            'aromatic_count': self.aromatic_count,
            'pairs': [{'interaction': p} for p in self.pair_interactions],
            'nearby_hydrophobics': [{'position': p} for p in self.nearby_hydrophobic_positions],
        }
        

# Define Protocol evaluator for mutagenesis rules' classes
@runtime_checkable
class ClusterEvaluator(Protocol):
    """
    Protocol defining the interface for aggregation rule evaluators.
    
    Any class implementing these attributes and methods can be used
    as a ClusterEvaluator, regardless of inheritance hierarchy.
    
    The @runtime_checkable decorator enables isinstance() checks,
    useful for validation in the RuleRegistry.
    
    Attributes (as properties):
        name: Unique identifier for this rule (e.g., 'hydrophobic_aliphatic')
        description: Human-readable explanation of the rule
        aggregation_score: Base score for clusters matching this rule
        residues: Amino acids targeted by this rule
    
    Methods:
        find_clusters: Identify qualifying clusters in a sequence region
    """
    
    @property
    def name(self) -> str:
        """
        Unique identifier for this rule.
        
        Used as dictionary key in results and for rule selection.
        Convention: lowercase with underscores (e.g., 'hydrophobic_aliphatic')
        """
        ...  # The ellipsis indicates "this is a protocol method stub"
    
    @property
    def description(self) -> str:
        """
        Human-readable description of what this rule detects.
        
        Displayed in help text and analysis reports.
        Example: "V, I, L, A, M clustered (≥3 residues within 4 positions)"
        """
        ...
    
    @property
    def aggregation_score(self) -> int:
        """
        Base aggregation propensity score for this rule.
        
        Higher scores indicate greater aggregation risk.
        Typical range: 1-3 for single rules, 4-8 for multi-rule clusters.
        """
        ...
    
    @property
    def residues(self) -> FrozenSet[str]:
        """
        Set of single-letter amino acid codes targeted by this rule.
        
        Using frozenset ensures immutability and enables set operations.
        Example: frozenset('VILAM') for hydrophobic aliphatic residues.
        """
        ...
    
    def find_clusters(
        self,
        sequence: str,
        start: int,
        stop: int
    ) -> List[Cluster]:
        """
        Identify all qualifying clusters within a sequence region.
        
        This is the core method that implements the rule's detection logic.
        
        Args:
            sequence: Complete protein sequence (not just the region).
                      Full sequence is provided to allow context-aware
                      analysis if needed.
            start: Start position of region to analyze (1-indexed, inclusive)
            stop: End position of region to analyze (1-indexed, inclusive)
        
        Returns:
            List of Cluster objects meeting this rule's criteria.
            Empty list if no qualifying clusters found.
        
        Implementation notes:
            - Extract region with: region_seq = sequence[start-1:stop]
            - Positions in returned Clusters should be absolute (1-indexed
              relative to full sequence), not relative to region
            - Only return clusters meeting minimum size/proximity criteria
        """
        ...
        

# Define rules for mutagenesis with clustering requirement
class BaseClusterEvaluator:
    """
    Base implementation providing common clustering logic.
    
    Subclasses override class attributes to define rule-specific behavior.
    """
    
    # Override in subclasses
    NAME: str = "base"
    DESCRIPTION: str = "Base evaluator"
    RESIDUES: frozenset = frozenset()
    MIN_CLUSTER_SIZE: int = 2
    MAX_GAP: int = 3
    AGGREGATION_SCORE: int = 1
    
    @property
    def name(self) -> str:
        return self.NAME
    
    @property
    def description(self) -> str:
        return self.DESCRIPTION
    
    @property
    def aggregation_score(self) -> int:
        return self.AGGREGATION_SCORE
    
    @property
    def residues(self) -> frozenset:
        return self.RESIDUES
    
    def find_clusters(
        self,
        sequence: str,
        start: int,
        stop: int
    ) -> List[Cluster]:
        """Standard clustering implementation."""
        region_seq = sequence[start - 1:stop]
        
        # Find positions of matching residues
        matching_positions = [
            start + i
            for i, aa in enumerate(region_seq)
            if aa in self.RESIDUES
        ]
        
        # Cluster nearby positions
        raw_clusters = self._cluster_positions(matching_positions)
        
        # Filter by minimum size and create Cluster objects
        return [
            Cluster(
                positions=tuple(cluster),
                residues=tuple(sequence[p - 1] for p in cluster),
                rule_name=self.NAME,
                aggregation_score=self.AGGREGATION_SCORE
            )
            for cluster in raw_clusters
            if len(cluster) >= self.MIN_CLUSTER_SIZE
        ]
    
    def _cluster_positions(self, positions: List[int]) -> List[List[int]]:
        """Group positions into clusters based on MAX_GAP threshold."""
        if not positions:
            return []
        
        clusters = []
        current = [positions[0]]
        
        for pos in positions[1:]:
            if pos - current[-1] <= self.MAX_GAP:
                current.append(pos)
            else:
                if len(current) >= 2:
                    clusters.append(current)
                current = [pos]
        
        if len(current) >= 2:
            clusters.append(current)
        
        return clusters


class HydrophobicAliphaticEvaluator(BaseClusterEvaluator):
    """
    Detects clusters of hydrophobic aliphatic residues (V, I, L, A, M).
    
    Biological basis: Aliphatic side chains drive aggregation through
    hydrophobic collapse and subsequent β-sheet formation. The β-branched
    residues (V, I) have particularly high intrinsic aggregation propensity
    due to their conformational preferences favoring extended structures.
    
    Triggering criteria: ≥3 residues within a span of 4 positions.
    """
    NAME = "hydrophobic_aliphatic"
    DESCRIPTION = "V, I, L, A, M clustered (≥3 residues within 4 positions)"
    RESIDUES = frozenset('VILAM')
    MIN_CLUSTER_SIZE = 3
    MAX_GAP = 4
    AGGREGATION_SCORE = 3


class AromaticEvaluator(BaseClusterEvaluator):
    """
    Detects clusters of aromatic residues (F, Y, W).
    
    Biological basis: Aromatic residues contribute to aggregation through
    π-π stacking and hydrophobic interactions. Phenylalanine is particularly
    aggregation-prone. Tyrosine can modulate aggregation through its hydroxyl
    group (hydrogen bonding capability).
    """
    NAME = "aromatic"
    DESCRIPTION = "F, Y, W clustered (≥2 residues within 3 positions)"
    RESIDUES = frozenset('FYW')
    MIN_CLUSTER_SIZE = 2
    MAX_GAP = 3
    AGGREGATION_SCORE = 2


class AmideEvaluator(BaseClusterEvaluator):
    """
    Detects clusters of amide-containing residues (Q, N).
    
    Biological basis: Q/N-rich regions are hallmarks of prion-like domains
    and intrinsically disordered regions prone to liquid-liquid phase
    separation (LLPS). The amide side chains form "polar zipper" hydrogen
    bond networks that can nucleate and stabilize cross-β structure.
    
    References:
    - Alberti et al., Cell 2009
    - King et al., Science 2012
    """
    NAME = "amide"
    DESCRIPTION = "Q, N clustered (≥2 residues within 3 positions)"
    RESIDUES = frozenset('QN')
    MIN_CLUSTER_SIZE = 2
    MAX_GAP = 3
    AGGREGATION_SCORE = 1


class HydrophobicAromaticEvaluator(BaseClusterEvaluator):
    """
    Detects hydrophobic-aromatic adjacency patterns.
    
    This rule uses special logic beyond simple clustering:
    - Condition A: ≥2 hydrophobic-aromatic adjacent pairs
    - Condition B: 1 pair + additional hydrophobic within 3 positions
    
    Biological basis: Adjacent hydrophobic-aromatic residues create
    particularly stable β-strand segments in amyloid fibrils. The aromatic
    ring intercalates between aliphatic chains, optimizing van der Waals
    packing in the fibril core.
    """
    NAME = "hydrophobic_and_aromatic"
    DESCRIPTION = "Hydrophobic (V,I,L,A,M) adjacent to aromatic (F,Y,W)"
    RESIDUES = frozenset('VILAMFYW')
    MIN_CLUSTER_SIZE = 2
    MAX_GAP = 1
    AGGREGATION_SCORE = 2
    
    HYDROPHOBIC = frozenset('VILAM')
    AROMATIC = frozenset('FYW')
    
    def find_clusters(
        self,
        sequence: str,
        start: int,
        stop: int
    ) -> List[Union[Cluster, HydrophobicAromaticCluster]]:
        """Override base implementation with special pair-detection logic."""
        region_seq = sequence[start - 1:stop]
        
        # Find adjacent hydrophobic-aromatic pairs
        pairs = []
        for i in range(len(region_seq) - 1):
            aa1, aa2 = region_seq[i], region_seq[i + 1]
            pos1, pos2 = start + i, start + i + 1
            
            is_pair = (
                (aa1 in self.HYDROPHOBIC and aa2 in self.AROMATIC) or
                (aa1 in self.AROMATIC and aa2 in self.HYDROPHOBIC)
            )
            if is_pair:
                pairs.append({
                    'positions': (pos1, pos2),
                    'residues': (aa1, aa2),
                    'interaction': f"{aa1}{pos1}-{aa2}{pos2}"
                })
        
        if not pairs:
            return []
        
        # Collect all hydrophobic positions for condition B
        hydrophobic_positions = {
            start + i
            for i, aa in enumerate(region_seq)
            if aa in self.HYDROPHOBIC
        }
        
        clusters = []
        
        # Condition A: ≥2 pairs close together
        if len(pairs) >= 2:
            pair_positions = set()
            pair_interactions = []
            for pair in pairs:
                pair_positions.update(pair['positions'])
                pair_interactions.append(pair['interaction'])
            
            sorted_positions = tuple(sorted(pair_positions))
            cluster_residues = tuple(sequence[p - 1] for p in sorted_positions)
            
            clusters.append(HydrophobicAromaticCluster(
                positions=sorted_positions,
                residues=cluster_residues,
                rule_name=self.NAME,
                aggregation_score=self.AGGREGATION_SCORE,
                pair_count=len(pairs),
                condition='at_least_2_pairs',
                hydrophobic_count=sum(1 for r in cluster_residues if r in self.HYDROPHOBIC),
                aromatic_count=sum(1 for r in cluster_residues if r in self.AROMATIC),
                pair_interactions=tuple(pair_interactions),
                nearby_hydrophobic_positions=()
            ))
        
        # Condition B: 1 pair + nearby hydrophobic
        for pair in pairs:
            pair_position_set = set(pair['positions'])
            
            nearby = []
            for hp in hydrophobic_positions:
                if hp in pair_position_set:
                    continue
                min_distance = min(abs(hp - p) for p in pair['positions'])
                if min_distance <= 3:
                    nearby.append(hp)
            
            if nearby:
                all_positions = tuple(sorted(pair_position_set | set(nearby)))
                cluster_residues = tuple(sequence[p - 1] for p in all_positions)
                
                clusters.append(HydrophobicAromaticCluster(
                    positions=all_positions,
                    residues=cluster_residues,
                    rule_name=self.NAME,
                    aggregation_score=self.AGGREGATION_SCORE,
                    pair_count=1,
                    condition='1_pair_plus_hydrophobic',
                    hydrophobic_count=sum(1 for r in cluster_residues if r in self.HYDROPHOBIC),
                    aromatic_count=sum(1 for r in cluster_residues if r in self.AROMATIC),
                    pair_interactions=(pair['interaction'],),
                    nearby_hydrophobic_positions=tuple(nearby)
                ))
        
        # Deduplicate
        seen = set()
        unique = []
        for cluster in clusters:
            if cluster.positions not in seen:
                seen.add(cluster.positions)
                unique.append(cluster)
        
        return unique
        

class RuleRegistry:
    """
    Registry for managing cluster evaluation rules.
    
    Provides centralized registration and retrieval of evaluators,
    enabling dynamic rule addition without modifying core logic.
    """
    
    def __init__(self):
        self._evaluators: Dict[str, ClusterEvaluator] = {}
    
    def register(self, evaluator: ClusterEvaluator) -> None:
        """Register an evaluator under its name."""
        if evaluator.name in self._evaluators:
            logger.warning(f"Overwriting existing evaluator: {evaluator.name}")
        self._evaluators[evaluator.name] = evaluator
        logger.debug(f"Registered evaluator: {evaluator.name}")
    
    def get(self, name: str) -> ClusterEvaluator:
        """Retrieve evaluator by name."""
        if name not in self._evaluators:
            raise KeyError(f"Unknown rule: {name}. Available: {list(self._evaluators.keys())}")
        return self._evaluators[name]
    
    def list_rules(self) -> List[str]:
        """Get list of all registered rule names."""
        return list(self._evaluators.keys())
    
    def evaluate_region(
        self,
        sequence: str,
        start: int,
        stop: int,
        rules: Optional[List[str]] = None
    ) -> Dict[str, List[Cluster]]:
        """
        Evaluate all (or selected) rules on a region.
        
        Args:
            sequence: Full protein sequence
            start: Region start (1-indexed)
            stop: Region end (1-indexed)
            rules: Optional list of rule names; if None, all rules applied
        
        Returns:
            Dict mapping rule name to list of identified clusters
        """
        target_rules = rules or self.list_rules()
        
        results = {}
        for rule_name in target_rules:
            if rule_name not in self._evaluators:
                logger.warning(f"Skipping unknown rule: {rule_name}")
                continue
            
            evaluator = self._evaluators[rule_name]
            clusters = evaluator.find_clusters(sequence, start, stop)
            results[rule_name] = clusters
            
            logger.debug(
                f"Rule '{rule_name}' found {len(clusters)} clusters in {start}:{stop}"
            )
        
        return results


# Create global registry with default rules
def create_default_registry() -> RuleRegistry:
    """Create registry with all standard aggregation rules."""
    registry = RuleRegistry()
    registry.register(HydrophobicAliphaticEvaluator())
    registry.register(AromaticEvaluator())
    registry.register(AmideEvaluator())
    registry.register(HydrophobicAromaticEvaluator())
    return registry


# Module-level default registry
DEFAULT_REGISTRY = create_default_registry()

# Compatibility layer: Generate RULES dict from registry for backward compatibility
def _generate_rules_dict(registry: RuleRegistry) -> Dict[str, Dict]:
    """Generate RULES dictionary from registry for backward compatibility."""
    rules = {}
    for rule_name in registry.list_rules():
        evaluator = registry.get(rule_name)
        rules[rule_name] = {
            'description': evaluator.description,
            'residues': set(evaluator.residues),
            'min_cluster_size': getattr(evaluator, 'MIN_CLUSTER_SIZE', 2),
            'max_gap': getattr(evaluator, 'MAX_GAP', 3),
            'mutations': DEFAULT_MUTATIONS,
            'aggregation_score': evaluator.aggregation_score,
        }
    return rules

RULES = _generate_rules_dict(DEFAULT_REGISTRY)


@dataclass(frozen=True, slots=True)
class MultiRuleCluster:
    """
    Represents a cluster matching multiple aggregation rules.
    
    These clusters represent regions with compounded aggregation risk
    where multiple physicochemical features converge.
    """
    positions: Tuple[int, ...]
    residues: Tuple[str, ...]
    rules: Tuple[str, ...]  # Immutable tuple of rule names
    combined_aggregation_score: int
    
    def __post_init__(self):
        if len(self.positions) != len(self.residues):
            raise ValueError("Position and residue counts must match")
        if len(self.rules) < 2:
            raise ValueError("MultiRuleCluster requires at least 2 rules")
    
    @property
    def size(self) -> int:
        return len(self.positions)
    
    @property
    def span(self) -> int:
        return max(self.positions) - min(self.positions) + 1 if self.positions else 0
    
    @property
    def is_multi_rule(self) -> bool:
        """Always True for MultiRuleCluster; aids duck typing."""
        return True
    
    @property
    def rule_string(self) -> str:
        """Joined rule names for display."""
        return '+'.join(sorted(self.rules))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'positions': list(self.positions),
            'residues': list(self.residues),
            'size': self.size,
            'span': self.span,
            'rules': list(self.rules),
            'combined_aggregation_score': self.combined_aggregation_score,
            'is_multi_rule': True,
        }


class UnionFind:
    """
    Disjoint Set Union (Union-Find) with path compression and union by rank.
    
    Provides near-constant time operations for:
    - Finding which set an element belongs to
    - Merging two sets
    
    Time complexity: O(α(n)) amortized per operation, where α is the
    inverse Ackermann function (effectively ≤4 for any practical input size).
    
    Usage in mutagenesis context:
    - Each cluster is an element
    - Overlapping clusters are unioned into the same set
    - Final sets represent merged aggregation-prone regions
    """
    
    __slots__ = ('parent', 'rank', '_size')
    
    def __init__(self, n: int):
        """
        Initialize Union-Find structure for n elements.
        
        Args:
            n: Number of elements (0 to n-1)
        """
        self.parent = list(range(n))  # Each element is its own parent initially
        self.rank = [0] * n  # All trees start with height 0
        self._size = [1] * n  # Track size of each set
    
    def find(self, x: int) -> int:
        """
        Find the representative (root) of element x's set.
        
        Uses path compression: all nodes along the path to root
        are updated to point directly to root, flattening the tree.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """
        Merge the sets containing elements x and y.
        
        Uses union by rank: the shorter tree is attached under the
        taller tree to maintain balance.
        
        Returns:
            True if sets were merged, False if already in same set
        """
        px, py = self.find(x), self.find(y)
        
        if px == py:
            return False  # Already in same set
        
        # Union by rank: attach smaller tree under larger
        if self.rank[px] < self.rank[py]:
            px, py = py, px  # Ensure px is the larger tree
        
        self.parent[py] = px  # Attach py under px
        self._size[px] += self._size[py]  # Update size
        
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1  # Tree grew taller
        
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """Check if x and y are in the same set."""
        return self.find(x) == self.find(y)
    
    def set_size(self, x: int) -> int:
        """Get the size of the set containing x."""
        return self._size[self.find(x)]
    
    def get_groups(self) -> Dict[int, List[int]]:
        """
        Get all disjoint sets as a dictionary.
        
        Returns:
            Dict mapping root element to list of all elements in that set
        """
        groups: Dict[int, List[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        return groups

def create_cluster(
    positions: List[int],
    sequence: str,
    rule_name: str,
    aggregation_score: int
) -> Cluster:
    """
    Factory function to create Cluster from mutable inputs.
    
    Handles conversion from lists to tuples and extracts residues from sequence.
    This provides a clean interface while the Cluster itself remains immutable.
    """
    sorted_positions = tuple(sorted(positions))
    residues = tuple(sequence[p - 1] for p in sorted_positions)
    
    return Cluster(
        positions=sorted_positions,
        residues=residues,
        rule_name=rule_name,
        aggregation_score=aggregation_score
    )

def print_usage_example():
    """Print detailed usage example"""
    example = """
USAGE EXAMPLES:
=========================================================

1. Rule-based mutagenesis in specific regions:
   python AGGRESSOR.py protein.fasta --regions 10:20 30:40 50:60

2. Rule-based with custom mutations:
   python AGGRESSOR.py protein.fasta --regions 5:15 -m A D E

3. Rule-based with specific rules only:
   python AGGRESSOR.py protein.fasta --regions 10:30 --rules hydrophobic_aliphatic aromatic

4. Combined approach (rules + specific positions):
   python AGGRESSOR.py protein.fasta --regions 10:20 --positions 15 25 --mutations P G

5. With insertions and rule-based:
   python AGGRESSOR.py protein.fasta --regions 5:15 --insert-positions 10 --insert-aas K

6. With gatekeeping amino acids (only for edge positions):
   python AGGRESSOR.py protein.fasta --regions 10:20 --gatekeeping Y K

7. Detailed verbose output:
   python AGGRESSOR.py protein.fasta --regions 10:20 -v

8. Analyze the entire sequence (all residues):
   python AGGRESSOR.py protein.fasta --regions all

AVAILABLE RULES:
=========================================================
• hydrophobic_aliphatic    : Triggers if ≥3 V, I, L, A, M residues within 4 positions of each other
• aromatic                 : Triggers if ≥2 F, Y, W residues within 3 positions of each other
• amide                    : Triggers if ≥2 Q, N residues within 3 positions of each other
• hydrophobic_and_aromatic : Triggers if ≥2 hydrophobic-aromatic adjacent pairs 
                             OR 1 pair + at least 1 hydrophobic residue within 3 positions

AGGREGATION SCORE RANKING (highest to lowest):
1. hydrophobic_aliphatic: 3
2. hydrophobic_and_aromatic: 2
3. aromatic: 2
4. amide: 1

GATEKEEPING AMINO ACIDS:
=========================================================
• Gatekeeping amino acids (default: Y) are only applied to positions at the edge of motifs
  or directly adjacent to them (within 1 position of motif boundary)
• Regular mutations are applied to all positions in motifs
• Use --gatekeeping option to specify custom gatekeeping amino acids

REQUIRED PARAMETERS:
=========================================================
• input_file    : Input FASTA file containing protein sequence (the multifasta format is not supported!)
• Either --positions OR --regions must be specified
"""
    print(example)

def read_fasta(filepath: str) -> Tuple[str, str]:
    """
    Read a single sequence FASTA file
    Returns: (header, sequence)
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        header = ""
        sequence = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                if header:  # If we already have a header, we're done (single sequence)
                    break
                header = line
            elif line and not header:
                raise ValueError("FASTA file must start with header line (>)")
            elif header:
                sequence += line.upper()
        
        if not header or not sequence:
            raise ValueError("Invalid FASTA file or empty sequence")
        
        return header, sequence
    
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Error reading FASTA file: {e}")

def validate_amino_acids(input_data: Union[str, List[str]], name: str = "amino acids", strict: bool = False) -> bool:
    """
    Unified function to validate amino acid codes.
    
    Args:
        input_data: Can be a string (sequence) or list of strings (amino acid codes)
        name: Description for error messages (used only in strict mode)
        strict: If True, raises ValueError on invalid input. If False, returns bool.
    
    Returns:
        bool: True if all amino acids are valid (only when strict=False)
    
    Raises:
        ValueError: If strict=True and invalid amino acids are found
    """
    if isinstance(input_data, str):
        # Validate sequence string: check if all characters are valid AAs
        invalid_chars = [char for char in set(input_data.upper()) if char not in VALID_AAS]
        if invalid_chars:
            if strict:
                raise ValueError(f"Invalid {name}: contains invalid characters {invalid_chars}. "
                               f"Valid amino acids: {', '.join(sorted(VALID_AAS))}")
            return False
        return True
    
    elif isinstance(input_data, list):
        # Validate list of amino acid codes: check if all items are single valid AAs
        invalid = [aa for aa in input_data if len(aa) != 1 or aa.upper() not in VALID_AAS]
        if invalid:
            if strict:
                raise ValueError(f"Invalid {name}: {invalid}. Valid amino acids: {', '.join(sorted(VALID_AAS))}")
            return False
        return True
    
    else:
        raise TypeError(f"input_data must be str or list, got {type(input_data).__name__}")

def validate_sequence(sequence: str) -> bool:
    """Validate that the sequence contains only valid amino acid codes (non-strict)"""
    return validate_amino_acids(sequence, "sequence", strict=False)

def parse_region(region_str: str, seq_length: int) -> Tuple[int, int]:
    """Parse region string in format start:stop (1-indexed)"""
    try:
        if ':' not in region_str:
            raise ValueError("Region must be in format start:stop")
        
        start_str, stop_str = region_str.split(':')
        start = int(start_str.strip())
        stop = int(stop_str.strip())
        
        # Validate bounds
        if start < 1 or stop > seq_length:
            raise ValueError(f"Region {start}:{stop} out of bounds (1-{seq_length})")
        if start > stop:
            raise ValueError(f"Start position {start} cannot be greater than stop position {stop}")
        
        return start, stop
    except ValueError as e:
        raise ValueError(f"Invalid region format '{region_str}': {e}")


def normalize_regions(regions: Optional[List[str]], seq_length: int) -> Optional[List[str]]:
    """
    Normalize --regions values.
    
    Supports a special value "all" (case-insensitive) which expands to "1:<seq_length>".
    If "all" is present alongside other regions, "all" takes precedence.
    """
    if not regions:
        return regions
    
    tokens = [r.strip() for r in regions if r is not None and str(r).strip()]
    if not tokens:
        return None
    
    if any(t.lower() == "all" for t in tokens):
        return [f"1:{seq_length}"]
    
    return tokens

def find_clusters(positions: List[int], max_gap: int = 3) -> List[List[int]]:
    """
    Find clusters of positions where positions are within max_gap of each other
    
    Args:
        positions: Sorted list of positions
        max_gap: Maximum allowed gap between consecutive positions in a cluster
    
    Returns:
        List of clusters, each cluster is a list of positions
    """
    if not positions:
        return []
    
    clusters = []
    current_cluster = [positions[0]]
    
    for i in range(1, len(positions)):
        if positions[i] - current_cluster[-1] <= max_gap:
            current_cluster.append(positions[i])
        else:
            if len(current_cluster) >= 2:  # Only keep clusters with at least 2 positions
                clusters.append(current_cluster)
            current_cluster = [positions[i]]
    
    # Don't forget the last cluster
    if len(current_cluster) >= 2:
        clusters.append(current_cluster)
    
    return clusters

def analyze_region(
    sequence: str,
    start: int,
    stop: int,
    registry: RuleRegistry = None,
    selected_rules: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze a protein region for aggregation-prone clusters.
    
    This is the main analysis function, now using the strategy pattern
    for rule evaluation and Union-Find for cluster merging.
    
    Args:
        sequence: Full protein sequence
        start: Region start position (1-indexed)
        stop: Region end position (1-indexed)
        registry: RuleRegistry containing evaluators to use
        selected_rules: Optional subset of rules to apply
    
    Returns:
        Dictionary containing comprehensive analysis results
    """
    if registry is None:
        registry = DEFAULT_REGISTRY
    
    region_seq = sequence[start - 1:stop]
    
    # Evaluate all applicable rules
    rule_results = registry.evaluate_region(sequence, start, stop, selected_rules)
    
    # Collect all clusters across rules
    all_clusters: List[Cluster] = []
    for rule_name, clusters in rule_results.items():
        all_clusters.extend(clusters)
    
    # Merge overlapping clusters using Union-Find
    merged_clusters = merge_overlapping_clusters_unionfind(
        all_clusters, sequence, gap_tolerance=MAX_GAP_FOR_MERGING
    )
    
    # Identify multi-rule clusters
    multi_rule_clusters = [
        c for c in merged_clusters
        if isinstance(c, MultiRuleCluster)
    ]
    
    # Collect hotspot positions
    hotspot_positions = set()
    for cluster in merged_clusters:
        hotspot_positions.update(cluster.positions)
    
    # Build results structure
    results = {
        'region': (start, stop),
        'sequence': region_seq,
        'length': len(region_seq),
        'rules': {},
        'merged_clusters': [c.to_dict() for c in merged_clusters],
        'multi_rule_clusters': [c.to_dict() for c in multi_rule_clusters],
        'aggregation_hotspots': sorted(hotspot_positions),
    }
    
    # Add per-rule details with backward-compatible fields
    for rule_name, clusters in rule_results.items():
        evaluator = registry.get(rule_name)
        
        rule_entry = {
            'description': evaluator.description,
            'qualifying_clusters': [c.to_dict() for c in clusters],
            'condition_met': len(clusters) > 0,
            'aggregation_score': evaluator.aggregation_score,
            # Backward compatibility fields
            'matching_positions': [],
            'matching_residues': [],
            'clusters': [],
        }
        
        # Populate backward-compatible fields
        for cluster in clusters:
            rule_entry['matching_positions'].extend(cluster.positions)
            rule_entry['matching_residues'].extend(cluster.residues)
            rule_entry['clusters'].append(list(cluster.positions))
        
        # Remove duplicates while preserving order
        rule_entry['matching_positions'] = list(dict.fromkeys(rule_entry['matching_positions']))
        rule_entry['matching_residues'] = list(dict.fromkeys(rule_entry['matching_residues']))
        
        # Special handling for hydrophobic_and_aromatic
        if rule_name == 'hydrophobic_and_aromatic' and clusters:
            rule_entry['special_clusters'] = [c.to_dict() for c in clusters]
        
        results['rules'][rule_name] = rule_entry
    
    return results

def find_overlapping_clusters_optimized(all_clusters: List[Tuple[str, Dict]], sequence: str) -> List[Dict]:
    """
    Optimized O(n log n) overlap detection using spatial sorting.
    Returns list of multi-rule clusters.
    """
    if not all_clusters:
        return []
    
    # Create cluster intervals with metadata (start, end, rule_name, cluster_data)
    intervals = []
    for rule_name, cluster_data in all_clusters:
        positions = cluster_data.get('positions', [])
        if positions:
            # Clean and validate rule_name
            if not rule_name or not isinstance(rule_name, str):
                rule_name = 'unknown'
            
            intervals.append({
                'start': min(positions),
                'end': max(positions),
                'rule_name': rule_name,
                'cluster_data': cluster_data,
                'positions_set': set(positions)
            })
    
    if not intervals:
        return []
    
    # Sort by start position (O(n log n))
    intervals.sort(key=lambda x: x['start'])
    
    # Single pass merge (O(n))
    merged_clusters = []
    current_group = [intervals[0]]
    
    for interval in intervals[1:]:
        # Check if current interval overlaps or is close to the last in current group
        last_in_group = current_group[-1]
        gap = interval['start'] - last_in_group['end'] - 1
        
        if interval['positions_set'].intersection(last_in_group['positions_set']) or gap <= MAX_GAP_FOR_MERGING:
            # Merge into current group
            current_group.append(interval)
        else:
            # Finalize current group and start new one
            if len(current_group) > 1:
                merged_clusters.append(_merge_cluster_group(current_group, sequence))
            current_group = [interval]
    
    # Don't forget the last group
    if len(current_group) > 1:
        merged_clusters.append(_merge_cluster_group(current_group, sequence))
    
    return merged_clusters

def _merge_cluster_group(group: List[Dict], sequence: str) -> Dict:
    """Merge a group of overlapping clusters into a single cluster"""
    all_positions = set()
    all_rule_names = set()  # Use set to collect all individual rule names
    max_score = 0
    total_score = 0
    
    for item in group:
        all_positions.update(item['positions_set'])
        # Handle merged rule names - split by '+' to get individual rules
        rule_name = item.get('rule_name', '')
        if not rule_name:
            rule_name = 'unknown'
        
        # Filter out any empty strings AND only keep valid rule names
        individual_rules = []
        for r in rule_name.split('+'):
            r = r.strip()
            if r and r in RULES:
                individual_rules.append(r)
        
        all_rule_names.update(individual_rules)
        score = item['cluster_data'].get('aggregation_score', 0)
        max_score = max(max_score, score)
        total_score += score
    
    sorted_positions = sorted(all_positions)
    combined_residues = [sequence[pos-1] for pos in sorted_positions]
    
    # Get unique rules (already unique since we used a set)
    unique_rules = sorted(list(all_rule_names))
    
    # If no valid rules found, check if we have any non-standard rule names
    if not unique_rules:
        # Try to extract any rule-like names that might have been created
        for item in group:
            rule_name = item.get('rule_name', '')
            # Look for patterns like rule names in the string
            for possible_rule in RULES.keys():
                if possible_rule in rule_name:
                    unique_rules.append(possible_rule)
        
        # Remove duplicates again
        unique_rules = sorted(list(set(unique_rules)))
    
    # If still no rules, use a placeholder but ensure it's valid for downstream processing
    if not unique_rules:
        # Use the first valid rule as a fallback
        unique_rules = [list(RULES.keys())[0]]
    
    if len(unique_rules) > 1:
        combined_score = total_score + len(unique_rules) - 1
    else:
        combined_score = max_score
    
    if len(unique_rules) > 1:
        return MultiRuleCluster(
            positions=tuple(sorted_positions),
            residues=tuple(combined_residues),
            rules=tuple(unique_rules),
            combined_aggregation_score=combined_score
        )
    else:
        return Cluster(
            positions=tuple(sorted_positions),
            residues=tuple(combined_residues),
            rule_name=unique_rules[0],
            aggregation_score=combined_score
        )

def merge_overlapping_clusters_unionfind(
    clusters: List[Cluster],
    sequence: str,
    gap_tolerance: int = MAX_GAP_FOR_MERGING
) -> List[Union[Cluster, MultiRuleCluster]]:
    """
    Merge overlapping or proximal clusters using Union-Find.
    
    This replaces the iterative while-loop approach with an efficient
    O(n² α(n)) algorithm where α is the inverse Ackermann function.
    
    For typical protein analyses with <1000 clusters, this provides
    significant speedup over the previous O(n²) worst-case approach.
    
    Args:
        clusters: List of Cluster objects to potentially merge
        sequence: Full protein sequence for residue extraction
        gap_tolerance: Maximum gap between clusters to consider for merging
    
    Returns:
        List of merged clusters (Cluster or MultiRuleCluster)
    """
    if not clusters:
        return []
    
    n = len(clusters)
    uf = UnionFind(n)
    
    # Phase 1: Identify all overlapping pairs and union them
    # O(n²) comparisons, but each union is O(α(n))
    for i in range(n):
        for j in range(i + 1, n):
            if clusters[i].overlaps_with(clusters[j], gap_tolerance):
                uf.union(i, j)
    
    # Phase 2: Group clusters by their root
    groups = uf.get_groups()
    
    # Phase 3: Merge each group into a single cluster
    merged_clusters = []
    
    for root, indices in groups.items():
        group_clusters = [clusters[i] for i in indices]
        
        if len(group_clusters) == 1:
            # No merging needed
            merged_clusters.append(group_clusters[0])
        else:
            # Merge multiple clusters
            merged = _merge_cluster_group_to_object(group_clusters, sequence)
            merged_clusters.append(merged)
    
    return merged_clusters

def _merge_cluster_group_to_object(
    clusters: List[Cluster],
    sequence: str
) -> Union[Cluster, MultiRuleCluster]:
    """
    Merge a group of overlapping clusters into a single object.
    
    Returns MultiRuleCluster if multiple rules are involved,
    otherwise returns a regular Cluster with combined positions.
    """
    # Collect all positions and rules
    all_positions: Set[int] = set()
    all_rules: Set[str] = set()
    max_score = 0
    total_score = 0
    
    for cluster in clusters:
        all_positions.update(cluster.positions)
        all_rules.add(cluster.rule_name)
        max_score = max(max_score, cluster.aggregation_score)
        total_score += cluster.aggregation_score
    
    sorted_positions = tuple(sorted(all_positions))
    residues = tuple(sequence[p - 1] for p in sorted_positions)
    
    if len(all_rules) > 1:
        # Multiple rules: create MultiRuleCluster
        # Combined score = sum of individual scores + bonus for convergence
        combined_score = total_score + len(all_rules) - 1
        
        return MultiRuleCluster(
            positions=sorted_positions,
            residues=residues,
            rules=tuple(sorted(all_rules)),
            combined_aggregation_score=combined_score
        )
    else:
        # Single rule: create regular Cluster with merged positions
        rule_name = next(iter(all_rules))
        
        return Cluster(
            positions=sorted_positions,
            residues=residues,
            rule_name=rule_name,
            aggregation_score=max_score
        )

def create_mutated_sequence(sequence: str, position: int, new_aa: str) -> str:
    """
    Efficiently create a mutated sequence by replacing one amino acid.
    Optimized to avoid unnecessary list creation for large sequences.
    """
    if position < 1 or position > len(sequence):
        raise ValueError(f"Position {position} out of range (1-{len(sequence)})")
    
    # For small sequences, list conversion is fine
    # For large sequences, use slicing (more memory efficient)
    if len(sequence) < 1000:
        seq_list = list(sequence)
        seq_list[position - 1] = new_aa
        return ''.join(seq_list)
    else:
        # For large sequences, use string slicing (avoids full list creation)
        return sequence[:position-1] + new_aa + sequence[position:]

def get_mutations_for_position(pos: int, cluster_positions: List[int], 
                               region_start: int, region_end: int,
                               mutations: List[str], gatekeeping_aas: List[str]) -> List[str]:
    """Get list of mutations to apply based on position type (edge vs internal)"""
    if is_edge_position(pos, cluster_positions, region_start, region_end):
        return list(set(mutations + gatekeeping_aas))
    return mutations

def is_edge_position(pos: int, cluster_positions: List[int], region_start: int, region_end: int) -> bool:
    """
    Check if a position is at the edge of a cluster or directly adjacent to it.
    
    Args:
        pos: Position to check
        cluster_positions: All positions in the cluster
        region_start: Start of the region (1-indexed)
        region_end: End of the region (1-indexed)
    
    Returns:
        True if position is at edge of cluster or adjacent to region boundary
    """
    if not cluster_positions:
        return False
    
    # Check if position is at the edge of the cluster
    min_pos = min(cluster_positions)
    max_pos = max(cluster_positions)
    
    # Position is at the edge if it's the minimum or maximum position in the cluster
    if pos == min_pos or pos == max_pos:
        return True
    
    # Check if position is adjacent to region boundaries
    # (within 1 position of region start or end)
    if pos == region_start or pos == region_end:
        return True
    
    # Check if position is directly adjacent to region boundaries
    if pos == region_start + 1 or pos == region_end - 1:
        return True
    
    return False

def apply_rule_mutations(sequence: str, region_analysis: Dict, mutations: List[str], 
                        selected_rules: List[str] = None, gatekeeping_aas: List[str] = None) -> List[Tuple[str, str, int]]:
    """Apply mutations based on rules in analyzed region"""
    results = []
    
    if gatekeeping_aas is None:
        gatekeeping_aas = GATEKEEPING_AAS
    
    # Filter rules if specific ones are selected
    if selected_rules:
        rules_to_apply = {k: v for k, v in region_analysis['rules'].items() if k in selected_rules}
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
                aggregation_score=cluster_dict.get('aggregation_score', rule_data['aggregation_score'])
            ))
    
    # Merge overlapping clusters
    merged_clusters = merge_overlapping_clusters_unionfind(
        cluster_objects, sequence, gap_tolerance=MAX_GAP_FOR_MERGING
    )
    
    # Track mutated positions
    mutated_positions: Set[int] = set()
    
    # Apply mutations to each merged cluster
    for cluster in merged_clusters:
        positions_to_mutate = list(cluster.positions)
        
        # Get aggregation score
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
            
            # Determine mutations based on edge status
            all_mutations = get_mutations_for_position(
                pos, positions_to_mutate, region_start, region_end,
                mutations, gatekeeping_aas
            )
            
            for new_aa in all_mutations:
                if new_aa == original_aa:
                    continue
                
                mutated_seq = create_mutated_sequence(sequence, pos, new_aa)
                
                is_gatekeeping = new_aa in gatekeeping_aas and new_aa not in mutations
                description = f"{original_aa}{pos}{new_aa} | {rule_desc} (agg_score={agg_score})"
                
                if is_gatekeeping:
                    description += f" | GATEKEEPING ({new_aa})"
                
                results.append((description, mutated_seq, agg_score))
    
    return results

def mutate_sequence(sequence: str, positions: List[int], mutations: List[str], 
                   regions: List[str] = None, selected_rules: List[str] = None,
                   insertion_positions: List[int] = None, insertion_aas: List[str] = None,
                   gatekeeping_aas: List[str] = None, verbose: bool = False) -> Tuple[List[Tuple[str, str]], List[Dict]]:
    """Generate mutated sequences with point mutations, rule-based mutations, and insertions"""
    all_results = []  # Will store (description, sequence, agg_score)
    region_analyses = []
    seq_len = len(sequence)
    
    # Use default gatekeeping amino acids if not provided
    if gatekeeping_aas is None:
        gatekeeping_aas = GATEKEEPING_AAS
    
    # Validate positions for direct mutations
    for pos in positions:
        if pos < 1 or pos > seq_len:
            raise ValueError(f"Position {pos} is out of range (sequence length: {seq_len})")
    
    # Analyze regions for rule-based mutations
    if regions:
        for region_str in regions:
            start, stop = parse_region(region_str, seq_len)
            analysis = analyze_region(sequence, start, stop)
            region_analyses.append(analysis)
            
            
            logger.debug(f"Analyzing region {start}:{stop}")
            logger.debug(f"Sequence: {analysis['sequence']}")
            logger.debug(f"Length: {analysis['length']} residues")
            
            for rule_name, rule_data in analysis['rules'].items():
                if rule_name == 'hydrophobic_and_aromatic' and rule_data['condition_met']:
                    logger.debug(f"\n{rule_name.upper()}:")
                    logger.debug(f"  Description: {rule_data['description']}")
                    logger.debug(f"  ✓ Rule triggered!")
                    
                    for i, cluster in enumerate(rule_data['special_clusters'], 1):
                        logger.debug(f"\n  Cluster {i} ({cluster['condition']}):")
                        logger.debug(f"    Positions: {cluster['positions']}")
                        logger.debug(f"    Residues: {''.join(cluster['residues'])}")
                        logger.debug(f"    Hydrophobic count: {cluster['hydrophobic_count']}")
                        logger.debug(f"    Aromatic count: {cluster['aromatic_count']}")
                        
                        if cluster['condition'] == 'at_least_2_pairs':
                            logger.debug(f"    Pairs found: {cluster['pair_count']}")
                            for j, pair in enumerate(cluster.get('pairs', []), 1):
                                logger.debug(f"      Pair {j}: {pair['interaction']}")
                        else:  # 1_pair_plus_hydrophobic
                            if cluster.get('pairs'):
                                pair = cluster['pairs'][0]
                                logger.debug(f"    Pair: {pair['interaction']}")
                            if cluster.get('nearby_hydrophobics'):
                                logger.debug(f"    Nearby hydrophobic residues:")
                                for h in cluster['nearby_hydrophobics']:
                                    position = h.get('position', 'unknown')
                                    residue = sequence[position - 1] if isinstance(position, int) else '?'
                                    logger.debug(f"      {residue}{position}")
                        
                        logger.debug(f"    Size: {cluster['size']}, Span: {cluster['span']}aa")
                
                elif rule_data['matching_positions']:
                    logger.debug(f"\n{rule_name.upper()}:")
                    logger.debug(f"  Description: {rule_data['description']}")
                    residues_str = ''.join(rule_data['matching_residues'])
                    logger.debug(f"  Matching residues: {residues_str}")
                    logger.debug(f"  Positions: {rule_data['matching_positions']}")
                    
                    if rule_data['clusters']:
                        logger.debug(f"  Found {len(rule_data['clusters'])} cluster(s):")
                        for i, cluster in enumerate(rule_data['clusters'], 1):
                            cluster_residues = [sequence[pos-1] for pos in cluster]
                            span = max(cluster) - min(cluster) + 1
                            logger.debug(
                                f"    Cluster {i}: positions {cluster}, "
                                f"residues {''.join(cluster_residues)}, "
                                f"size={len(cluster)}, span={span}aa"
                            )
                    
                    if rule_data['qualifying_clusters']:
                        logger.debug(f"  ✓ QUALIFYING CLUSTERS:")
                        for i, cluster in enumerate(rule_data['qualifying_clusters'], 1):
                            logger.debug(
                                f"    Cluster {i}: positions {cluster['positions']}, "
                                f"residues {''.join(cluster['residues'])}, "
                                f"size={cluster['size']}, span={cluster['span']}aa, "
                                f"agg_score={cluster['aggregation_score']}"
                            )
                    else:
                        logger.debug(f"  ✗ No qualifying clusters")
            
            if analysis['multi_rule_clusters']:
                logger.debug(f"\n{'*'*60}")
                logger.debug(f"MULTI-RULE CLUSTERS (HIGH AGGREGATION RISK):")
                logger.debug(f"{'*'*60}")
                for i, cluster in enumerate(analysis['multi_rule_clusters'], 1):
                    logger.debug(f"\n  Multi-Rule Cluster {i}:")
                    logger.debug(f"    Rules: {', '.join(cluster['rules'])}")
                    logger.debug(f"    Positions: {cluster['positions']}")
                    logger.debug(f"    Residues: {''.join(cluster['residues'])}")
                    logger.debug(f"    Combined Aggregation Score: {cluster['combined_aggregation_score']}/8")
                    
    # Apply rule-based mutations
    for analysis in region_analyses:
        rule_mutations = apply_rule_mutations(sequence, analysis, mutations, selected_rules, gatekeeping_aas)
        all_results.extend(rule_mutations)
    
    # Generate direct point mutations
    for pos in positions:
        original_aa = sequence[pos-1]
        
        # For direct mutations, apply only regular mutations (not gatekeeping)
        for new_aa in mutations:
            if new_aa == original_aa:
                continue
            
            mutated_seq = create_mutated_sequence(sequence, pos, new_aa)
            
            # Standardized description for direct mutations with agg_score=0
            description = f"{original_aa}{pos}{new_aa} | Direct mutation (agg_score=0)"
            all_results.append((description, mutated_seq, 0))
    
    # Generate insertions if specified
    if insertion_positions and insertion_aas:
        if len(insertion_positions) != len(insertion_aas):
            raise ValueError("insertion_positions and insertion_aas must have the same length")
        
        for ins_pos, ins_aa in zip(insertion_positions, insertion_aas):
            if ins_pos < 1 or ins_pos > seq_len + 1:
                raise ValueError(f"Insertion position {ins_pos} is out of range (1-{seq_len+1})")
            
            mutated_seq = sequence[:ins_pos-1] + ins_aa + sequence[ins_pos-1:]
            # Standardized description for insertions with agg_score=0
            description = f"Insertion: {ins_aa} inserted before position {ins_pos} (agg_score=0)"
            all_results.append((description, mutated_seq, 0))
    
    # Sort all results by aggregation score (descending)
    all_results.sort(key=lambda x: x[2], reverse=True)
    
    # Convert to format expected by downstream functions (description, sequence)
    sorted_results = [(desc, seq) for desc, seq, _ in all_results]
    
    return sorted_results, region_analyses

def extract_mutation_info(description: str, regions: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Extract mutation information from description string.
    
    Args:
        description: Mutation description (e.g., "V10F | Rule 'hydrophobic_aliphatic' (agg_score=3)")
        regions: List of region strings (e.g., ["10:20", "30:40"])
    
    Returns:
        Dict with keys: position, is_gatekeeping, region, agg_score, description
    """
    info = {
        'position': None,
        'is_gatekeeping': False,
        'region': None,
        'agg_score': 0,
        'description': description
    }
    
    # Extract position and aggregation score
    match = re.search(r'(\d+)', description)
    if match:
        info['position'] = int(match.group(1))
    
    match_score = re.search(r'agg_score=(\d+)', description)
    if match_score:
        info['agg_score'] = int(match_score.group(1))
    
    info['is_gatekeeping'] = 'GATEKEEPING' in description
    
    # Determine which region this position belongs to
    if regions and info['position']:
        for region_str in regions:
            start, stop = parse_region(region_str, 999999)  # Use large number to avoid validation error
            if start <= info['position'] <= stop:
                if info['region'] is None:
                    info['region'] = region_str
                else:
                    # Position belongs to multiple regions (overlap)
                    info['region'] += f",{region_str}"
    
    return info


def get_region_for_position(position: int, regions: Optional[List[str]] = None) -> Optional[str]:
    """Get the region(s) a position belongs to."""
    if not regions or position is None:
        return None
    
    matching_regions = []
    for region_str in regions:
        start, stop = parse_region(region_str, 999999)
        if start <= position <= stop:
            matching_regions.append(region_str)
    
    return ','.join(matching_regions) if matching_regions else None


def generate_multi_point_mutations(
    mutations: List[Tuple[str, str]],
    sequence: str,
    multi_mutation_levels: List[int],
    regions: Optional[List[str]] = None,
    max_combinations: int = 10000,
    top_variants_per_position: int = 3
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Generate multi-point mutations from single mutations.
    
    Args:
        mutations: List of (description, sequence) tuples for single mutations
        sequence: Original protein sequence
        multi_mutation_levels: Levels to generate (e.g., [2, 3] for double and triple)
        regions: List of region strings for categorization
        max_combinations: Maximum combinations to generate per level
    
    Returns:
        Dict mapping level (2, 3, etc.) to list of dicts with:
          - description, sequence, agg_score
          - positions, regions, is_gatekeeping, is_all_gatekeeping
    """
    # Parse mutation info from descriptions
    mutation_infos = []
    for desc, seq in mutations:
        info = extract_mutation_info(desc, regions)
        if not info['position'] or not seq:
            continue
        
        # Only point mutations can be combined into multi-point substitutions.
        # Skip entries like insertions which don't encode "A123B".
        match = re.search(r'([A-Z])(\d+)([A-Z])', desc)
        if not match:
            continue
        
        new_aa = match.group(3)
        mutation_infos.append({
            'position': info['position'],
            'is_gatekeeping': info['is_gatekeeping'],
            'region': info['region'],
            'agg_score': info['agg_score'],
            'description': desc,
            'sequence': seq,
            'original_aa': sequence[info['position'] - 1],
            'new_aa': new_aa
        })
    
    # Group by position to avoid combinatorial blow-up and invalid duplicate-position combos.
    # Keep only the best (highest agg_score) variants per position.
    by_position: Dict[int, List[Dict[str, Any]]] = {}
    for mi in mutation_infos:
        by_position.setdefault(mi['position'], []).append(mi)
    
    # Deduplicate per-position variants by new amino acid, keeping highest scoring option.
    filtered_by_position: Dict[int, List[Dict[str, Any]]] = {}
    for pos, variants in by_position.items():
        best_by_newaa: Dict[str, Dict[str, Any]] = {}
        for v in variants:
            new_aa = v.get('new_aa')
            if not new_aa:
                continue
            existing = best_by_newaa.get(new_aa)
            if existing is None or v.get('agg_score', 0) > existing.get('agg_score', 0):
                best_by_newaa[new_aa] = v
        
        kept = list(best_by_newaa.values())
        kept.sort(key=lambda x: x.get('agg_score', 0), reverse=True)
        if top_variants_per_position and top_variants_per_position > 0:
            kept = kept[:top_variants_per_position]
        filtered_by_position[pos] = kept
    
    # Only positions with at least 1 viable variant can participate
    viable_positions = sorted([p for p, vs in filtered_by_position.items() if vs])
    
    multi_results = {}
    
    for level in multi_mutation_levels:
        if level < 2 or level > len(viable_positions):
            continue
        
        multi_results[level] = []
        
        # Generate combinations of positions at this level (not raw mutations),
        # then choose one variant per selected position.
        count = 0
        for pos_combo in combinations(viable_positions, level):
            if count >= max_combinations:
                logger.warning(f"Reached maximum combinations ({max_combinations}) for level {level}")
                break
            
            variant_lists = [filtered_by_position[p] for p in pos_combo]
            for variant_combo in product(*variant_lists):
                if count >= max_combinations:
                    break
                
                positions = [v['position'] for v in variant_combo]
                new_aas = [v['new_aa'] for v in variant_combo]
                
                # Apply mutations in reverse order to keep indices stable
                mutations_to_apply = sorted(zip(positions, new_aas), reverse=True, key=lambda x: x[0])
                combined_seq = sequence
                for pos, aa in mutations_to_apply:
                    combined_seq = create_mutated_sequence(combined_seq, pos, aa)
                
                combined_agg_score = sum(v.get('agg_score', 0) for v in variant_combo)
                
                mutations_str = ' + '.join(
                    f"{v['original_aa']}{v['position']}{v['new_aa']}"
                    for v in sorted(variant_combo, key=lambda x: x['position'])
                )
                
                is_all_gatekeeping = all(v.get('is_gatekeeping', False) for v in variant_combo)
                regions_involved = sorted({v['region'] for v in variant_combo if v.get('region')})
                regions_str = ', '.join(regions_involved) if regions_involved else "Direct"
                
                description = f"{mutations_str} | {regions_str} (level={level}, agg_score={combined_agg_score})"
                if is_all_gatekeeping:
                    description += " | GATEKEEPING_COMBINED"
                
                multi_results[level].append({
                    'description': description,
                    'sequence': combined_seq,
                    'agg_score': combined_agg_score,
                    'positions': positions,
                    'regions': regions_involved,
                    'level': level,
                    'is_all_gatekeeping': is_all_gatekeeping
                })
                count += 1
        
        # Sort by aggregation score (descending)
        multi_results[level].sort(key=lambda x: x.get('agg_score', 0), reverse=True)
        logger.info(f"Generated {len(multi_results[level])} level-{level} mutations")
    
    return multi_results


def categorize_multi_mutations(
    multi_mutations: Dict[int, List[Union[Tuple[str, str, int], Dict[str, Any]]]],
    regions: Optional[List[str]] = None
) -> Dict[int, Dict[str, List[Tuple[str, str, int]]]]:
    """
    Categorize multi-mutations by region(s) and mutation type.
    
    Categories:
    - single_region: All positions in the same region
    - multi_region: Positions span different regions
    - gatekeeping: Only gatekeeping mutations
    """
    categorized = {}
    
    for level, mutations in multi_mutations.items():
        categorized[level] = {
            'single_region': [],
            'multi_region': [],
            'gatekeeping': []
        }
        
        for item in mutations:
            # Accept both legacy tuple format and new dict format
            if isinstance(item, dict):
                desc = item.get('description', '')
                seq = item.get('sequence', '')
                agg_score = int(item.get('agg_score', 0) or 0)
                regions_involved = item.get('regions') or []
                is_gatekeeping = bool(item.get('is_all_gatekeeping')) or ('GATEKEEPING_COMBINED' in desc)
            else:
                desc, seq, agg_score = item
                regions_involved = re.findall(r'(\d+:\d+)', desc)
                is_gatekeeping = 'GATEKEEPING_COMBINED' in desc
            
            if is_gatekeeping:
                categorized[level]['gatekeeping'].append((desc, seq, agg_score))
                continue
            
            if not regions_involved or len(set(regions_involved)) == 1:
                categorized[level]['single_region'].append((desc, seq, agg_score))
            else:
                categorized[level]['multi_region'].append((desc, seq, agg_score))
    
    return categorized


def create_output_directory(base_path: str, multi_mutation_levels: Optional[List[int]] = None) -> Path:
    """
    Create organized output directory structure.
    
    Args:
        base_path: Base directory path
        multi_mutation_levels: Levels to create directories for (e.g., [2, 3])
    
    Returns:
        Path to the created directory
    """
    output_dir = Path(base_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if multi_mutation_levels:
        for level in sorted(multi_mutation_levels):
            level_dir = output_dir / f"{_level_to_text(level)}_mutations"
            level_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {level_dir}")
    
    return output_dir


def _level_to_text(level: int) -> str:
    """Convert level number to text (2 -> 'double', 3 -> 'triple', etc.)"""
    level_names = {
        2: 'double',
        3: 'triple',
        4: 'quadruple',
        5: 'quintuple',
        6: 'sextuple'
    }
    return level_names.get(level, f'{level}x')


def write_multi_mutations_by_category(
    output_dir: str,
    original_header: str,
    original_seq: str,
    categorized_mutations: Dict[int, Dict[str, List[Tuple[str, str, int]]]],
    include_original: bool = True
):
    """
    Write categorized multi-mutations to separate FASTA files.
    
    Creates files:
    - {level}_mutations/single_region.fasta
    - {level}_mutations/multi_region.fasta
    - {level}_mutations/gatekeeping.fasta
    """
    output_base = Path(output_dir)
    
    for level in sorted(categorized_mutations.keys()):
        level_dir = output_base / f"{_level_to_text(level)}_mutations"
        level_dir.mkdir(parents=True, exist_ok=True)
        
        categories = categorized_mutations[level]
        
        # Write single region mutations
        if categories['single_region']:
            file_path = level_dir / 'single_region.fasta'
            _write_fasta_file(
                str(file_path), original_header, original_seq,
                categories['single_region'], include_original
            )
            logger.info(f"Wrote {len(categories['single_region'])} single-region {_level_to_text(level)} mutations")
        
        # Write multi region mutations
        if categories['multi_region']:
            file_path = level_dir / 'multi_region.fasta'
            _write_fasta_file(
                str(file_path), original_header, original_seq,
                categories['multi_region'], include_original
            )
            logger.info(f"Wrote {len(categories['multi_region'])} multi-region {_level_to_text(level)} mutations")
        
        # Write gatekeeping mutations
        if categories['gatekeeping']:
            file_path = level_dir / 'gatekeeping.fasta'
            _write_fasta_file(
                str(file_path), original_header, original_seq,
                categories['gatekeeping'], include_original
            )
            logger.info(f"Wrote {len(categories['gatekeeping'])} gatekeeping {_level_to_text(level)} mutations")


def _write_fasta_file(
    output_file: str,
    original_header: str,
    original_seq: str,
    mutations: List[Tuple[str, str, int]],
    include_original: bool = True
):
    """Internal function to write FASTA file from mutations with scores."""
    with open(output_file, 'w') as f:
        if include_original:
            f.write(f"{original_header}\n")
            for i in range(0, len(original_seq), FASTA_LINE_LENGTH):
                f.write(f"{original_seq[i:i+FASTA_LINE_LENGTH]}\n")
        
        # Handle both 2-tuple and 3-tuple formats
        for item in mutations:
            if len(item) == 3:
                description, mutated_seq, _ = item
            else:
                description, mutated_seq = item
            
            # Extract protein name from original header (remove '>')
            protein_name = original_header[1:].strip()
            f.write(f">{protein_name}_{description}\n")
            for j in range(0, len(mutated_seq), FASTA_LINE_LENGTH):
                f.write(f"{mutated_seq[j:j+FASTA_LINE_LENGTH]}\n")


def write_fasta(output_file: str, original_header: str, original_seq: str, 
                mutations: List[Tuple[str, str]], include_original: bool = True):
    """Write results to FASTA file"""
    with open(output_file, 'w') as f:
        if include_original:
            f.write(f"{original_header}\n")
            for i in range(0, len(original_seq), FASTA_LINE_LENGTH):
                f.write(f"{original_seq[i:i+FASTA_LINE_LENGTH]}\n")
        
        for i, (description, mutated_seq) in enumerate(mutations, 1):
            # Extract protein name from original header (remove '>')
            protein_name = original_header[1:].strip()
            # Keep spaces in description for readability
            f.write(f">{protein_name}_{description}\n")
            for j in range(0, len(mutated_seq), FASTA_LINE_LENGTH):
                f.write(f"{mutated_seq[j:j+FASTA_LINE_LENGTH]}\n")

def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup and return argument parser"""
    parser = argparse.ArgumentParser(
        description='Perform rule-based in silico mutagenesis on protein sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        usage='python AGGRESSOR.py <input_file> [options]'
    )
    
    # Required arguments
    required = parser.add_argument_group('REQUIRED ARGUMENTS')
    required.add_argument('input_file', nargs='?', help='Input FASTA file containing protein sequence')
    
    # Region and rule arguments
    region_args = parser.add_argument_group('REGION-BASED MUTAGENESIS')
    region_args.add_argument('-r', '--regions', type=str, nargs='+',
                            help='Regions to analyze for rule-based mutations (format: start:stop) or "all"')
    region_args.add_argument('--rules', type=str, nargs='+',
                        choices=['hydrophobic_aliphatic', 'aromatic', 'amide', 'hydrophobic_and_aromatic'],
                        help='Specific rules to apply')
    
    # Direct mutation arguments
    mutation_args = parser.add_argument_group('DIRECT MUTATIONS')
    mutation_args.add_argument('-p', '--positions', type=int, nargs='+', default=[],
                              help='Specific positions to mutate (1-indexed)')
    mutation_args.add_argument('-m', '--mutations', type=str, nargs='+', default=DEFAULT_MUTATIONS,
                              help=f'Amino acids to mutate to (default: {DEFAULT_MUTATIONS})')
    
    # Gatekeeping amino acids argument
    gatekeeping_args = parser.add_argument_group('GATEKEEPING AMINO ACIDS')
    gatekeeping_args.add_argument('-g', '--gatekeeping', type=str, nargs='+', default=GATEKEEPING_AAS,
                                 help=f'Amino acids to use as gatekeepers for edge positions (default: {GATEKEEPING_AAS})')
    
    # Insertion arguments
    insertion_args = parser.add_argument_group('INSERTIONS')
    insertion_args.add_argument('--insert-positions', type=int, nargs='+',
                               help='Positions for insertions (insertion happens BEFORE this position)')
    insertion_args.add_argument('--insert-aas', type=str, nargs='+',
                               help='Amino acids to insert at insertion positions')
    
    # Multi-point mutation arguments
    multi_mut_args = parser.add_argument_group('MULTI-POINT MUTATIONS')
    multi_mut_args.add_argument('--multi-mutations', type=int, nargs='+',
                               help='Levels of multi-point mutations to generate (e.g., 2 3 for double and triple)')
    multi_mut_args.add_argument('--multi-top-per-position', type=int, default=3,
                               help=('Limit the number of substitution variants kept PER POSITION (ranked by agg_score) '
                                     'when generating multi-mutations. This does NOT restrict which positions are used; '
                                     'it only reduces combinatorial explosion when a position has many possible substitutions. '
                                     'Set very large to effectively disable (default: 3).'))
    multi_mut_args.add_argument('--multi-output', default='mutated_sequences',
                               help='Output directory for mutations (default: mutated_sequences)')
    
    # Aggregation analysis arguments
    agg_args = parser.add_argument_group('AGGREGATION ANALYSIS')
    agg_args.add_argument('--agg-only', action='store_true',
                         help='Only identify aggregation hotspots without generating mutations')
    agg_args.add_argument('--min-agg-score', type=int, default=4,
                         help='Minimum aggregation score to flag as hotspot (default: 4)')
    
    # Output arguments
    output_args = parser.add_argument_group('OUTPUT')
    output_args.add_argument('-o', '--output', default='mutated_sequences.fasta',
                            help='Output FASTA file for single mutations (default: mutated_sequences.fasta)')
    output_args.add_argument('--no-original', action='store_true',
                            help='Do not include original sequence in output')
    
    # Verbosity and help
    other_args = parser.add_argument_group('OTHER OPTIONS')
    other_args.add_argument('-v', '--verbose', action='store_true',
                           help='Show detailed analysis of regions including clusters')
    other_args.add_argument('-h', '--help', action='store_true',
                           help='Show this help message and exit')
    
    return parser

def print_help_info(parser: argparse.ArgumentParser):
    """Print detailed help information"""
    print("=" * 70)
    print("RULE-BASED IN SILICO MUTAGENESIS SCRIPT (WITH CLUSTERING)")
    print("WITH MULTI-POINT MUTATION SUPPORT")
    print("=" * 70)
    print("\nDESCRIPTION:")
    print("Performs rule-based mutagenesis on protein sequences.")
    print("Rules apply ONLY when amino acids are clustered together.")
    print("Multiple rules can apply simultaneously to the same motif.")
    print("Motifs with multiple rule matches are flagged as aggregation hotspots.")
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
    print("  • --multi-top-per-position limits how many substitution variants are kept PER POSITION")
    print("    (ranked by agg_score) before building multi-mutation combinations.")
    print("  • This does NOT restrict which positions can be used; it only reduces combinatorial explosion")
    print("    when a single position has many possible substitutions.")
    print("  • To effectively disable limiting, set a very large value (e.g., --multi-top-per-position 9999).")
    print("\nExample usage:")
    print("  python AGGRESSOR.py protein.fasta --regions 10:30 -m P G D K --multi-mutations 2 3")
    print("\nThis generates:")
    print("  mutated_sequences/")
    print("  ├── single_mutations.fasta")
    print("  ├── double_mutations/")
    print("  │   ├── single_region.fasta (2 mutations in same region)")
    print("  │   ├── multi_region.fasta (2 mutations in different regions)")
    print("  │   └── gatekeeping.fasta (2 gatekeeping mutations combined)")
    print("  └── triple_mutations/")
    print("      ├── single_region.fasta (3 mutations in same region)")
    print("      ├── multi_region.fasta (3 mutations in different regions)")
    print("      └── gatekeeping.fasta (3 gatekeeping mutations combined)")
    print("\nCategorization Rules:")
    print("  • single_region: All positions belong to the same input region")
    print("  • multi_region: Positions span multiple different input regions (non-overlapping)")
    print("  • gatekeeping: Combinations of gatekeeping amino acids only")
    print("\n" + "=" * 70)
    print("RULE DETAILS AND AGGREGATION SCORES:")
    print("=" * 70)
    for rule_name, rule in RULES.items():
        print(f"\n{rule_name}:")
        print(f"  {rule['description']}")
        if rule_name == 'hydrophobic_and_aromatic':
            print(f"  Conditions:")
            print(f"    1. At least 2 hydrophobic-aromatic adjacent pairs")
            print(f"    2. OR 1 hydrophobic-aromatic pair + at least 1 hydrophobic within 3 positions")
        else:
            print(f"  Residues: {', '.join(sorted(rule['residues']))}")
        print(f"  Min cluster size: {rule['min_cluster_size']}")
        print(f"  Max gap: {rule['max_gap']} positions")
        print(f"  Aggregation score: {rule['aggregation_score']}")
    
    print("\n" + "=" * 70)
    print("OVERLAP HANDLING PROCEDURE:")
    print("=" * 70)
    print("Overlapping motifs are resolved by keeping the UNION of motif positions.")
    print("Additionally, motifs within 2 positions of each other are also merged.")
    print("When clusters overlap or are close (≤2 positions apart), the script will:")
    print("1. Merge all overlapping clusters into a single unified cluster")
    print("2. Merge clusters that are within 2 positions of each other")
    print("3. Apply mutations to the unified motif")
    print("This creates larger mutagenesis regions covering all adjacent risk areas.")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("GATEKEEPING AMINO ACIDS:")
    print("=" * 70)
    print("Gatekeeping amino acids are only applied to positions at the edge of motifs")
    print("or directly adjacent to region boundaries (within 1 position).")
    print(f"Default gatekeeping amino acids: {GATEKEEPING_AAS}")
    print("Use --gatekeeping option to specify custom gatekeeping amino acids.")
    print("=" * 70)

def validate_arguments(args) -> None:
    """Validate command line arguments"""
    # Check if at least one type of mutation is requested
    if not args.agg_only and not args.positions and not args.regions and not args.insert_positions:
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
    
    # Validate gatekeeping amino acids list
    try:
        validate_amino_acids(args.gatekeeping, "gatekeeping amino acids", strict=True)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate that insert-positions and insert-aas are both provided or both omitted
    if bool(args.insert_positions) != bool(args.insert_aas):
        print("\nERROR: Both --insert-positions and --insert-aas must be provided together", file=sys.stderr)
        print("Example: --insert-positions 10 20 --insert-aas K M")
        sys.exit(1)
    
    # Validate multi-mutations argument
    if args.multi_mutations:
        if not isinstance(args.multi_mutations, list):
            args.multi_mutations = [args.multi_mutations]
        
        # Check that all mutation levels are >= 2
        for level in args.multi_mutations:
            if level < 2:
                print(f"\nERROR: Multi-mutation levels must be >= 2 (got {level})", file=sys.stderr)
                sys.exit(1)
        
        if hasattr(args, 'multi_top_per_position'):
            if args.multi_top_per_position is not None and args.multi_top_per_position < 1:
                print("\nERROR: --multi-top-per-position must be >= 1", file=sys.stderr)
                sys.exit(1)
        
        # Warn if very high mutation levels
        if any(level > 6 for level in args.multi_mutations):
            print("\nWARNING: High mutation levels (>6) may generate very large numbers of combinations")
            print("This could consume significant memory and disk space.")
        
        logger.debug(f"Multi-mutation levels: {sorted(args.multi_mutations)}")

def print_mutation_summary(mutations: List[Tuple[str, str]]):
    """Print summary of generated mutations"""
    if not mutations:
        print("\nNo mutations generated. Check your criteria.")
        return
    
    print(f"\n{'='*70}")
    print("MUTATION SUMMARY (Sorted by aggregation score)")
    print(f"{'='*70}")
    
    # First, extract aggregation scores and group mutations
    mutation_data = []
    for desc, seq in mutations:
        # Extract agg_score from description
        agg_score = 0
        if "(agg_score=" in desc:
            try:
                agg_str = desc.split("(agg_score=")[1].split(")")[0]
                agg_score = int(agg_str)
            except (IndexError, ValueError):
                agg_score = 0
        mutation_data.append((desc, seq, agg_score))
    
    # Count by type using the new standardized headers
    direct = sum(1 for d, _, _ in mutation_data if "Direct mutation" in d)
    rule_based = sum(1 for d, _, _ in mutation_data if "Rule '" in d and "GATEKEEPING" not in d)
    merged_rules = sum(1 for d, _, _ in mutation_data if "MERGED RULES" in d and "GATEKEEPING" not in d)
    insertions = sum(1 for d, _, _ in mutation_data if "Insertion" in d)
    gatekeeping = sum(1 for d, _, _ in mutation_data if "GATEKEEPING" in d)
    
    # Count specific rule types for detailed breakdown
    hydrophobic = sum(1 for d, _, _ in mutation_data if "Rule 'hydrophobic_aliphatic'" in d and "GATEKEEPING" not in d)
    aromatic = sum(1 for d, _, _ in mutation_data if "Rule 'aromatic'" in d and "GATEKEEPING" not in d)
    amide = sum(1 for d, _, _ in mutation_data if "Rule 'amide'" in d and "GATEKEEPING" not in d)
    hydrophobic_arom = sum(1 for d, _, _ in mutation_data if "Rule 'hydrophobic_and_aromatic'" in d and "GATEKEEPING" not in d)
    
    print(f"Direct mutations: {direct} (agg_score: 0)")
    print(f"Rule-based mutations: {rule_based}")
    print(f"  • Hydrophobic mutations: {hydrophobic} (agg_score: 3)")
    print(f"  • Hydrophobic-aromatic mutations: {hydrophobic_arom} (agg_score: 2)")
    print(f"  • Aromatic mutations: {aromatic} (agg_score: 2)")
    print(f"  • Amide mutations: {amide} (agg_score: 1)")
    print(f"Merged rule mutations: {merged_rules} (agg_score: 4-8)")
    print(f"Insertions: {insertions} (agg_score: 0)")
    print(f"Gatekeeping mutations: {gatekeeping}")
    print(f"TOTAL: {len(mutations)}")
    
    # Show top 5 mutations by aggregation score
    if mutation_data:
        print(f"\nTOP 5 MUTATIONS BY AGGREGATION SCORE:")
        for i, (desc, _, agg_score) in enumerate(mutation_data[:5], 1):
            if len(desc) > 80:
                desc = desc[:77] + "..."
            print(f"{i}. {desc}")
    
    if hydrophobic_arom > 0:
        print(f"\nHydrophobic-aromatic mutations found:")
        for desc, _, _ in mutation_data:
            if "Rule 'hydrophobic_and_aromatic'" in desc and "GATEKEEPING" not in desc:
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                print(f"  • {desc}")
                break
    
    if gatekeeping > 0:
        print(f"\nGatekeeping mutations (edge positions only):")
        gatekeeping_shown = 0
        for i, (desc, _, _) in enumerate(mutation_data):
            if "GATEKEEPING" in desc:
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                print(f"  {gatekeeping_shown+1}. {desc}")
                gatekeeping_shown += 1
                if gatekeeping_shown >= 5:  # Show first 5 gatekeeping mutations
                    if gatekeeping > 5:
                        print(f"  ... and {gatekeeping - 5} more gatekeeping mutations")
                    break
    
    print(f"\nAll mutations (first 10, sorted by agg_score):")
    shown = 0
    for i, (desc, _, agg_score) in enumerate(mutation_data):
        print(f"{i+1}. {desc}")
        shown += 1
        if shown >= 10:
            if len(mutation_data) > 10:
                print(f"... and {len(mutation_data) - 10} more mutations")
            break


def print_aggregation_summary(region_analyses: List[Dict], sequence: str):
    """Print summary of aggregation analysis results"""
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
        
        print(f"\nREGION {region_start}:{region_end} ({region_end - region_start + 1} residues)")
        print("-" * 70)
        print(f"Sequence: {analysis['sequence']}")
        print(f"Total clusters found: {len(clusters)}")
        print(f"Aggregation hotspot positions: {', '.join(map(str, hotspots)) if hotspots else 'None'}")
        
        # Show rule-by-rule breakdown
        print(f"\nRule Breakdown:")
        for rule_name, rule_data in analysis['rules'].items():
            if rule_data['condition_met']:
                num_clusters = len(rule_data['qualifying_clusters'])
                positions = rule_data['matching_positions']
                print(f"  • {rule_name}: {num_clusters} cluster(s) at positions {positions}")
        
        # Show cluster details
        if clusters:
            print(f"\nDetailed Cluster Information:")
            for i, cluster in enumerate(clusters, 1):
                positions = cluster['positions']
                residues = ''.join(cluster['residues'])
                size = cluster['size']
                span = cluster['span']
                score = cluster.get('aggregation_score', cluster.get('combined_aggregation_score', 0))
                rule = cluster.get('rule_name', ', '.join(cluster.get('rules', [])))
                
                print(f"  Cluster {i}:")
                print(f"    Positions: {positions}")
                print(f"    Residues:  {residues}")
                print(f"    Size: {size}, Span: {span} aa")
                print(f"    Rule(s): {rule}")
                print(f"    Aggregation Score: {score}")
        
        # Highlight multi-rule clusters (highest risk)
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
                print(f"    Combined Aggregation Score: {score}/8 (HIGHEST RISK)")
        
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
        print("These regions match multiple aggregation rules simultaneously.")
        print("Mutations in these areas are highly recommended for aggregation suppression.")
    
    if total_hotspots > 0:
        print(f"\nRecommended mutation targets: {', '.join(map(str, sorted(all_hotspot_positions)))}")
    else:
        print("\nNo aggregation-prone hotspots detected in specified regions.")
    
    print("=" * 70)


def main():
    """Main entry point with integrated optimizations."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    if args.help or not args.input_file:
        print_help_info(parser)
        if not args.input_file:
            logger.error("Input FASTA file is required")
            sys.exit(1)
        sys.exit(0)
    
    validate_arguments(args)
    
    try:
        logger.info("=" * 70)
        logger.info("RULE-BASED MUTAGENESIS WITH OPTIMIZED CLUSTERING")
        logger.info("=" * 70)
        
        header, sequence = read_fasta(args.input_file)
        logger.info(f"Input: {args.input_file}")
        logger.info(f"Sequence length: {len(sequence)} residues")
        
        # Expand special region token(s) now that sequence length is known
        args.regions = normalize_regions(args.regions, len(sequence))
        
        if args.regions:
            logger.info(f"Regions to analyze: {args.regions}")
        if args.positions:
            logger.info(f"Direct mutation positions: {args.positions}")
        logger.info(f"Mutations: {args.mutations}")
        logger.info(f"Gatekeeping amino acids: {args.gatekeeping}")
        
        # Use existing mutate_sequence function
        if not args.agg_only:
            mutations, region_analyses = mutate_sequence(
                sequence,
                args.positions,
                [m.upper() for m in args.mutations],
                args.regions,
                args.rules,
                args.insert_positions,
                [aa.upper() for aa in args.insert_aas] if args.insert_aas else None,
                [aa.upper() for aa in args.gatekeeping],
                args.verbose
            )
            
            logger.info(f"Generated {len(mutations)} mutations")
            
            # Determine output path
            if args.multi_mutations:
                # Use directory structure for multi-mutations
                output_base = Path(args.multi_output)
                output_base.mkdir(parents=True, exist_ok=True)
                
                # Write single mutations to single_mutations.fasta
                single_output = output_base / 'single_mutations.fasta'
                write_fasta(str(single_output), header, sequence, mutations, not args.no_original)
                logger.info(f"Single mutations written to {single_output}")
                
                # Generate multi-point mutations
                logger.info(f"Generating multi-point mutations (levels: {args.multi_mutations})")
                multi_mutations = generate_multi_point_mutations(
                    mutations,
                    sequence,
                    args.multi_mutations,
                    regions=args.regions,
                    top_variants_per_position=getattr(args, 'multi_top_per_position', 3)
                )
                
                # Categorize multi-mutations
                categorized = categorize_multi_mutations(multi_mutations, regions=args.regions)
                
                # Create output directory structure
                create_output_directory(str(output_base), args.multi_mutations)
                
                # Write categorized multi-mutations
                write_multi_mutations_by_category(
                    str(output_base),
                    header,
                    sequence,
                    categorized,
                    include_original=not args.no_original
                )
                
                logger.info(f"Multi-mutation results written to {output_base}")
            else:
                # Use original single output file
                write_fasta(args.output, header, sequence, mutations, not args.no_original)
                logger.info(f"Results written to {args.output}")
            
            print_mutation_summary(mutations)
        else:
            # Aggregation analysis only
            region_analyses = []
            if args.regions:
                for region_str in args.regions:
                    start, stop = parse_region(region_str, len(sequence))
                    analysis = analyze_region(sequence, start, stop, selected_rules=args.rules)
                    region_analyses.append(analysis)
                    logger.info(f"Region {start}:{stop}: {len(analysis['merged_clusters'])} clusters found")
            
            logger.info("Aggregation analysis completed")
            
            # Print summary of aggregation analysis
            print_aggregation_summary(region_analyses, sequence)
        
        logger.info("=" * 70)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
