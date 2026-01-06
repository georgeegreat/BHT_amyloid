import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import argparse
import sys
import os
from typing import Tuple, List, Optional
from dataclasses import dataclass
import time
import json
import gc


@dataclass
class AnalysisResult:
    """Data class to store analysis results for each mutant."""
    mutant: str
    delta_scores: np.ndarray
    non_zero_delta_scores: np.ndarray  # Store non-zero deltas
    non_zero_positions: np.ndarray     # Positions where delta != 0
    mean_delta: float
    median_delta: float
    std_delta: float
    wilcoxon_stat: float
    p_value: float
    significant: bool


class Region:
    """Class to represent a region for detailed analysis."""
    def __init__(self, name: str, start: int, end: int, positions: List[int] = None):
        self.name = name
        self.start = start
        self.end = end
        # If positions are provided, use it directly (for custom regions)
        # Otherwise generate positions from start to end inclusive
        if positions:
            self.positions = np.array(sorted(set(positions)))
            self.start = min(self.positions)
            self.end = max(self.positions)
        else:
            self.positions = np.arange(start, end + 1)
    
    def __repr__(self):
        return f"Region(name='{self.name}', positions={self.start}-{self.end}, n_positions={len(self.positions)})"
    
    def get_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get subset of dataframe for this region."""
        if 'position' in df.columns:
            return df[df['position'].isin(self.positions)].copy()
        else:
            # Use index if in csv no colimn with positions
            return df.iloc[self.positions - 1 if self.positions[0] > 0 else self.positions].copy()


class MetascoreAnalyzer:
    """Class for metascore analysis."""
    # Color palettes for consistent visualization
    COLORS = {
        'significant': '#2E86AB',
        'nonsignificant': '#D3D3D3',
        'negative': '#E74C3C',
        'positive': '#2ECC71',
        'success': '#ABEBC6',
        'failure': '#F5B7B1',
        'grid': '#F0F0F0',
        'wt': '#2ECC71',
        'mutant': '#E74C3C'
    }


    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.df = None
        self.mutants = []
        self.results = []
        self.results_df = None
        self.output_dir = None
        self.non_zero_positions = None  # Store global non-zero positions
        self.regions = []  # List of Region objects for detailed analysis
        self.region_original_indices = {}  # Store original indices for each region
        
        plt.close('all')


    def detect_delimiter(self, filepath: str) -> str:
        """Detect the delimiter used in a CSV file."""
        try:
            with open(filepath, 'r') as f:
                first_lines = [f.readline() for _ in range(5)]
                
                # Try to detect delimiter
                for line in first_lines:
                    if ';' in line and ',' in line:
                        # If both exist, count occurrences
                        semicolon_count = line.count(';')
                        comma_count = line.count(',')
                        return ';' if semicolon_count > comma_count else ','
                    elif ';' in line:
                        return ';'
                    elif ',' in line:
                        return ','
                # Default to comma if no delimiter found
                return ','
        except Exception as e:
            print(f"Warning: Could not detect delimiter, defaulting to comma: {e}")
            return ','


    def parse_region_argument(self, region_str: str) -> Optional[Region]:
        """Parse a region specification string."""
        try:
            # Format: name:start-end or name:positions
            if ':' not in region_str:
                raise ValueError("Region must contain ':' separator")
            
            name_part, pos_part = region_str.split(':', 1)
            name = name_part.strip()
            
            if '-' in pos_part:
                # Range format: start-end
                start_str, end_str = pos_part.split('-')
                start = int(start_str.strip())
                end = int(end_str.strip())
                return Region(name, start, end)
            else:
                # List format: positions separated by commas
                positions = [int(p.strip()) for p in pos_part.split(',')]
                return Region(name, 0, 0, positions)
        except Exception as e:
            print(f"Warning: Could not parse region '{region_str}': {e}")
            return None


    def read_csv(self, filepath: str) -> pd.DataFrame:
        """Read and validate CSV"""
        try:
            # Detect delimiter
            delimiter = self.detect_delimiter(filepath)
            print(f"Detected delimiter: '{delimiter}'")
            df = pd.read_csv(filepath, sep=delimiter, low_memory=False)
            df.columns = df.columns.str.strip()
            print(f"Found columns: {list(df.columns)}")
            # Check for required columns with flexible naming
            has_aa = any(col.lower() == 'aa' for col in df.columns)
            has_aa_name = any(col.lower().replace('_', '').replace('-', '') == 'aaname' 
                            for col in df.columns)
            if not has_aa:
                print("Warning: 'aa' column not found. Using index as position.")
            if not has_aa_name:
                print("Warning: 'aa_name' column not found. Using empty strings.")
            return df
        except Exception as e:
            print(f"Error reading CSV: {e}")
            sys.exit(1)


    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Extract and prepare metascore data with flexible column naming."""
        metascore_cols = []
        for col in df.columns:
            col_lower = col.lower()
            # Look for columns containing 'metascore' or 'score'
            if 'metascore' in col_lower or ('score' in col_lower and len(col) > 5):
                metascore_cols.append(col)
        
        print(f"Found metascore columns: {metascore_cols}")
        if not metascore_cols:
            raise ValueError("No metascore columns found")
        
        wt_cols = []
        for col in metascore_cols:
            col_lower = col.lower()
            # Match various WT column naming conventions
            if ('wt' in col_lower or 'wild' in col_lower or 
                col_lower.endswith('_wt') or 'wildtype' in col_lower):
                wt_cols.append(col)
        
        if not wt_cols:
            # If no explicit WT column, try to find one with common WT patterns
            for col in metascore_cols:
                if any(pattern in col.lower() for pattern in ['control', 'ref', 'base']):
                    wt_cols.append(col)
        
        if not wt_cols:
            # If still no WT column found, assume the first metascore column is WT
            print("Warning: No WT column identified, using first metascore column as WT")
            wt_col = metascore_cols[0]
        else:
            wt_col = wt_cols[0]
            if len(wt_cols) > 1:
                print(f"Warning: Multiple WT-like columns found. Using '{wt_col}'")
        
        print(f"Using WT column: {wt_col}")
        
        # Get mutant columns (all metascore columns except WT)
        mutant_cols = [col for col in metascore_cols if col != wt_col]
        
        if not mutant_cols:
            raise ValueError("No mutant columns found")
        
        print(f"Mutant columns: {mutant_cols}")
        
        # Clean mutant names
        clean_names = []
        for col in mutant_cols:
            # Remove common prefixes
            name = col.replace('metascore_', '').replace('score_', '').replace('_score', '')
            name = name.replace('metascore', '').replace('score', '')
            while name.startswith('_'):
                name = name[1:]
            while name.endswith('_'):
                name = name[:-1]
            if not name:
                name = col
            clean_names.append(name.strip())
        

        new_data = {}
        new_data['position'] = df.get('aa', pd.Series(range(1, len(df) + 1)))
        name_cols = [col for col in df.columns if any(x in col.lower() for x in ['name', 'residue'])]
        if 'aa_name' in df.columns:
            new_data['amino_acid'] = df['aa_name']
        elif name_cols:
            new_data['amino_acid'] = df[name_cols[0]]
        else:
            new_data['amino_acid'] = pd.Series([''] * len(df))
        
        new_data['WT'] = df[wt_col]
        for orig_col, clean_name in zip(mutant_cols, clean_names):
            new_data[clean_name] = df[orig_col]
        result_df = pd.DataFrame(new_data)
        
        # Debug: Print data shape and columns
        print(f"Prepared data shape: {result_df.shape}")
        print(f"Prepared columns: {list(result_df.columns)}")
        return result_df, clean_names


    def find_top_regions(self, df: pd.DataFrame, mutants: List[str], 
                         n_regions: int = 2, min_size: int = 4, max_size: int = 8,
                         threshold: float = 0.03) -> List[Region]:
        """Automatically find top regions with most significant changes."""
        print(f"\nFinding top {n_regions} regions (size {min_size}-{max_size} aa)...")
        
        n_positions = len(df)
        positions = df['position'].values if 'position' in df.columns else np.arange(1, n_positions + 1)
        
        # Calculate total delta magnitude for each position
        position_scores = np.zeros(n_positions)
        for mutant in mutants:
            if mutant in df.columns:
                position_scores += np.abs((df[mutant] - df['WT']).values)
        
        # Sliding window for regions
        region_windows = []
        
        for window_size in range(min_size, min(max_size + 1, n_positions + 1)):
            if window_size <= n_positions:
                window_sums = np.convolve(position_scores, np.ones(window_size), mode='valid')
                for start in range(len(window_sums)):
                    end = start + window_size - 1
                    window_score = window_sums[start]
                    # Get positions
                    if 'position' in df.columns:
                        window_positions = df['position'].iloc[start:end+1].values
                    else:
                        window_positions = np.arange(start + 1, end + 2)
                    
                    region_windows.append({
                        'start_idx': start,
                        'end_idx': end,
                        'start_pos': window_positions[0],
                        'end_pos': window_positions[-1],
                        'positions': window_positions,
                        'score': window_score,
                        'size': window_size})
        
        if not region_windows:
            print("Warning: No valid regions found")
            return []
        
        # Sort by score (descending) and remove overlapping regions
        region_windows.sort(key=lambda x: x['score'], reverse=True)
        
        selected_regions = []
        used_positions = set()
        
        for window in region_windows:
            # Check if this window overlaps significantly with already selected regions
            window_pos_set = set(window['positions'])
            overlap = len(window_pos_set.intersection(used_positions))
            overlap_ratio = overlap / len(window_pos_set)
            
            if overlap_ratio <= 0.2:  # Allow up to 20% overlap
                region_name = f"Auto_Region_{len(selected_regions)+1}"
                region = Region(region_name, window['start_pos'], window['end_pos'])
                selected_regions.append(region)
                used_positions.update(window_pos_set)
                
                print(f"  Found {region_name}: positions {window['start_pos']}-{window['end_pos']} "
                      f"(score: {window['score']:.2f}, size: {window['size']} aa)")
                
                if len(selected_regions) >= n_regions:
                    break
        
        return selected_regions


    def find_non_zero_regions(self, df: pd.DataFrame, mutants: List[str], threshold: float = 0.03) -> Tuple[np.ndarray, pd.DataFrame]:
        """Find positions where at least one mutant has delta score above threshold."""
        all_non_zero_positions = []
        for mutant in mutants:
            if mutant in df.columns:
                delta_scores = df[mutant] - df['WT']
                # Find positions with non-zero delta (above threshold)
                non_zero_idx = np.where(np.abs(delta_scores) > threshold)[0]
                all_non_zero_positions.extend(non_zero_idx)
        
        # Get unique positions
        non_zero_positions = np.unique(all_non_zero_positions)

        if len(non_zero_positions) == 0:
            print("Warning: No non-zero delta positions found. Using all positions.")
            non_zero_positions = np.arange(len(df))
        print(f"Found {len(non_zero_positions)} positions with non-zero delta scores")
        
        # Create subset DataFrame for non-zero region
        region_df = df.iloc[non_zero_positions].copy()
        return non_zero_positions, region_df


    def perform_analysis(self, df: pd.DataFrame, mutants: List[str], 
                         use_non_zero_only: bool = True, threshold: float = 0.03) -> List[AnalysisResult]:
        """Perform Wilcoxon analysis for mutants with optional non-zero filtering."""
        results = []
        all_non_zero_positions = []
        
        print(f"Analyzing {len(mutants)} mutants...")
        
        for i, mutant in enumerate(mutants, 1):
            if i % 10 == 0 or i == len(mutants):
                print(f"  Progress: {i}/{len(mutants)} mutants")
            try:
                if mutant not in df.columns:
                    print(f"Warning: Mutant '{mutant}' not found in dataframe columns")
                    continue
                delta_scores = (df[mutant] - df['WT']).values
                
                # Find non-zero delta scores
                if use_non_zero_only:
                    non_zero_mask = np.abs(delta_scores) > threshold
                    non_zero_delta_scores = delta_scores[non_zero_mask]
                    non_zero_pos = np.where(non_zero_mask)[0]
                    all_non_zero_positions.extend(non_zero_pos)
                else:
                    non_zero_mask = np.ones_like(delta_scores, dtype=bool)
                    non_zero_delta_scores = delta_scores
                    non_zero_pos = np.arange(len(delta_scores))
                
                if len(non_zero_delta_scores) == 0:
                    print(f"Warning: {mutant} has no non-zero delta scores (threshold={threshold})")
                    continue
                
                if np.all(non_zero_delta_scores == non_zero_delta_scores[0]):
                    print(f"Warning: {mutant} has invalid data for analysis")
                    continue
                
                non_zero_differences = np.count_nonzero(non_zero_delta_scores)
                if non_zero_differences < 2:
                    print(f"Warning: {mutant} has insufficient non-zero differences ({non_zero_differences})")
                    stat, p_value = 0.0, 1.0
                else:
                    stat, p_value = wilcoxon(
                        non_zero_delta_scores, 
                        zero_method='wilcox', 
                        alternative='two-sided')
                
                mean_delta = np.nanmean(non_zero_delta_scores)
                median_delta = np.nanmedian(non_zero_delta_scores)
                std_delta = np.nanstd(non_zero_delta_scores)
                
                result = AnalysisResult(
                    mutant=mutant,
                    delta_scores=delta_scores,
                    non_zero_delta_scores=non_zero_delta_scores,
                    non_zero_positions=non_zero_pos,
                    mean_delta=mean_delta,
                    median_delta=median_delta,
                    std_delta=std_delta,
                    wilcoxon_stat=stat,
                    p_value=p_value,
                    significant=p_value < self.alpha)
                results.append(result)
                
            except Exception as e:
                print(f"Error analyzing {mutant}: {str(e).split(':')[0]}")
                result = AnalysisResult(
                    mutant=mutant,
                    delta_scores=np.array([np.nan] * len(df)),
                    non_zero_delta_scores=np.array([]),
                    non_zero_positions=np.array([]),
                    mean_delta=np.nan,
                    median_delta=np.nan,
                    std_delta=np.nan,
                    wilcoxon_stat=np.nan,
                    p_value=np.nan,
                    significant=False)
                results.append(result)
        
        # Store global non-zero positions
        if all_non_zero_positions:
            self.non_zero_positions = np.unique(all_non_zero_positions)
        else:
            self.non_zero_positions = np.arange(len(df))
        return results


    def create_summary_dataframe(self, results: List[AnalysisResult]) -> pd.DataFrame:
        summary_data = []
        for result in results:
            summary_data.append({
                'Mutant': result.mutant,
                'Mean_Δ': result.mean_delta,
                'Median_Δ': result.median_delta,
                'Std_Δ': result.std_delta,
                'Wilcoxon_Statistic': result.wilcoxon_stat,
                'P_value': result.p_value,
                'Significant': result.significant,
                'NonZero_Count': len(result.non_zero_delta_scores)})
        return pd.DataFrame(summary_data)


    def plot_pvalues(self, results_df: pd.DataFrame, output_path: str, region_name: str = ""):
        """Plot p-values for a specific region."""
        if region_name:
            title_suffix = f" ({region_name})"
        else:
            title_suffix = ""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        x = np.arange(len(results_df))
        p_values = results_df['P_value'].values
        significant = results_df['Significant'].values
        colors = np.where(significant, self.COLORS['significant'], self.COLORS['nonsignificant'])
        
        ax1.bar(x, p_values, color=colors, edgecolor='black', width=0.7)
        ax1.axhline(y=self.alpha, color='red', linestyle='--', linewidth=2, label=f'α={self.alpha}')
        ax1.set_yscale('log')
        ax1.set_ylim(max(p_values.min() * 0.1, 1e-4), 1)
        ax1.set_xlabel('Mutant')
        ax1.set_ylabel('P-value (log scale)')
        ax1.set_title(f'Wilcoxon Test P-values{title_suffix}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(results_df['Mutant'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y', color=self.COLORS['grid'])
        
        mean_deltas = results_df['Mean_Δ'].values
        std_deltas = results_df['Std_Δ'].values
        colors_delta = np.where(mean_deltas < 0, self.COLORS['negative'], self.COLORS['positive'])
        
        bars = ax2.bar(x, mean_deltas, yerr=std_deltas, capsize=5, color=colors_delta, edgecolor='black', width=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        for bar, mean_d, p_val in zip(bars, mean_deltas, p_values):
            height = bar.get_height()
            if not np.isnan(height):
                y_pos = height + (0.01 if height >= 0 else -0.01)
                va = 'bottom' if height >= 0 else 'top'
                sig_star = '*' if p_val < self.alpha else ''
                ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                        f'{mean_d:.3f}{sig_star}', ha='center', va=va,
                        fontsize=9, fontweight='bold')
        
        ax2.set_xlabel('Mutant')
        ax2.set_ylabel('Mean Δ Score')
        ax2.set_title(f'Mean Δ Scores with Error Bars{title_suffix}')
        ax2.set_xticks(x)
        ax2.set_xticklabels(results_df['Mutant'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y', color=self.COLORS['grid'])
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close(fig)


    def plot_delta_boxplot(self, results_df: pd.DataFrame, results: List[AnalysisResult], 
                           output_path: str, region_name: str = ""):
        """Create boxplot of delta scores for each mutant."""
        if region_name:
            title_suffix = f" ({region_name})"
        else:
            title_suffix = ""
        fig, ax = plt.subplots(figsize=(12, 6))
    
        box_data = []
        box_labels = []
        box_colors = []
        for idx, (result, row) in enumerate(zip(results, results_df.iterrows())):
            row_data = row[1]
            mutant = row_data['Mutant']
            
            delta_scores = result.non_zero_delta_scores
        
            # Filtering NaN
            valid_scores = delta_scores[~np.isnan(delta_scores)]
            if len(valid_scores) > 0:
                box_data.append(valid_scores)
                box_labels.append(mutant)
                box_colors.append(self.COLORS['significant'] if row_data['Significant'] else self.COLORS['nonsignificant'])
    
        if not box_data:
            return
    
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for i, scores in enumerate(box_data):
            x_pos = np.random.normal(i + 1, 0.04, len(scores))
            ax.scatter(x_pos, scores, alpha=0.6, s=30, color='black', edgecolor='black', linewidth=0.5)

        # Add mean markers
        mean_marker = None
        for i, scores in enumerate(box_data):
            mean_val = np.mean(scores)
            if i == 0:
                mean_marker = ax.scatter(i + 1, mean_val, color='red', s=100, marker='D', 
                                   edgecolor='white', linewidth=1.5, zorder=10,
                                   label='Mean Δ')
            else:
                ax.scatter(i + 1, mean_val, color='red', s=100, marker='D', 
                     edgecolor='white', linewidth=1.5, zorder=10)
    
        zero_line = ax.axhline(y=0, color='red', linestyle='--', linewidth=2, 
                          alpha=0.7, label='Δ = 0')
    
        ax.set_xlabel('Mutant')
        ax.set_ylabel('Δ Score')
        ax.set_title(f'Distribution of Δ Scores (Boxplot){title_suffix}')
        ax.set_xticklabels(box_labels, rotation=45, ha='right')
    
        if mean_marker is not None:
            ax.legend([zero_line, mean_marker], ['Δ = 0', 'Mean Δ'], loc='upper right')

        ax.grid(True, alpha=0.3, axis='y', color=self.COLORS['grid'])
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


    def plot_initial_scores(self, df: pd.DataFrame, mutants: List[str], output_path: str):
        """Plot initial WT and mutant scores across amino acid sequence with improved layout."""
        fig = plt.figure(figsize=(16, 12))
        
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.1)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        
        if 'position' in df.columns and 'amino_acid' in df.columns:
            positions = [f"{pos}-{aa}" for pos, aa in zip(df['position'], df['amino_acid'])]
        else:
            positions = [f"Pos{i+1}" for i in range(len(df))]
        
        x = np.arange(len(positions))
        
        # Calculate sampling interval for x-ticks to avoid overcrowding
        n_positions = len(positions)
        if n_positions > 50:
            tick_interval = max(1, n_positions // 20)
            show_ticks = x[::tick_interval]
            show_labels = [positions[i] for i in range(0, n_positions, tick_interval)]
        else:
            show_ticks = x
            show_labels = positions
        
        # Line plot of all scores
        ax1.plot(x, df['WT'], label='WT', color=self.COLORS['wt'], 
                linewidth=2.5, marker='o', markersize=4)
        
        # Plot all mutants
        for mutant in mutants:
            if mutant in df.columns:
                ax1.plot(x, df[mutant], label=mutant, alpha=0.7, 
                        linewidth=1.5, marker='s', markersize=2)
        
        ax1.set_ylabel('Metascore', fontsize=11)
        ax1.set_title('Metascore Distribution Across Amino Acid Sequence', fontsize=12, pad=10)
        ax1.legend(fontsize=9, ncol=3, loc='upper right', framealpha=0.8)
        ax1.grid(True, alpha=0.3, color=self.COLORS['grid'])
        ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # Hide x-axis labels
        
        # Heatmap of all scores
        all_variants = ['WT'] + [m for m in mutants if m in df.columns]
        score_matrix = df[all_variants].T.values
        
        im = ax2.imshow(score_matrix, cmap='viridis', aspect='auto', 
                       interpolation='nearest')
        
        cbar = fig.colorbar(im, ax=[ax1, ax2], orientation='vertical', pad=0.02)
        cbar.set_label('Metascore', fontsize=10)
        
        ax2.set_yticks(np.arange(len(all_variants)))
        ax2.set_yticklabels(all_variants, fontsize=9)
        ax2.set_xticks(show_ticks)
        ax2.set_xticklabels(show_labels, rotation=90, ha='center', fontsize=8)
        
        # Only show x-axis label on bottom plot
        ax2.set_xlabel('Position - Amino Acid', fontsize=11)
        ax2.set_title('Metascore Heatmap (Raw Scores)', fontsize=12, pad=10)
        
        # plt.tight_layout()
        fig.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close(fig)


    def plot_delta_heatmap(self, df: pd.DataFrame, results: List[AnalysisResult], 
                          output_path: str, show_labels: bool = False, 
                          region_name: str = "", is_specific_region: bool = False,
                          region_indices: np.ndarray = None):
        """Plot delta scores heatmap with optional labels."""
        if region_name:
            title_suffix = f" ({region_name})"
        else:
            title_suffix = ""
        
        # Calculate delta matrix
        delta_matrix = []
        valid_results = []
        
        for r in results:
            if len(r.non_zero_delta_scores) > 0:  # Only include mutants with non-zero deltas
                # For specific regions, we need to extract the subset of delta scores
                if is_specific_region and region_indices is not None:
                    # Use provided region indices to get correct subset
                    delta_subset = r.delta_scores[region_indices]
                else:
                    delta_subset = r.delta_scores
                
                delta_matrix.append(delta_subset)
                valid_results.append(r)
        
        if not delta_matrix:
            print("Warning: No delta data to plot in heatmap")
            return
        
        delta_matrix = np.array(delta_matrix)
        
        # Determine figure size based on number of positions
        n_positions = len(df)
        n_mutants = len(valid_results)
        
        # For specific regions, adjust figure size
        if is_specific_region:
            fig_width = max(8, min(16, n_positions * 0.5))  # Wider for better visibility
            fig_height = max(6, min(12, n_mutants * 0.6))
        else:
            fig_width = max(12, min(24, n_positions * 0.3))
            fig_height = max(6, min(18, n_mutants * 0.4))
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Calculate vmin and vmax for symmetric colormap
        delta_max = np.nanmax(np.abs(delta_matrix))
        vmin = -delta_max if delta_max > 0 else -0.1
        vmax = delta_max if delta_max > 0 else 0.1
        
        im = ax.imshow(delta_matrix, cmap='RdBu_r', aspect='auto',
                      vmin=vmin, vmax=vmax)
        
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('Δ Value', fontsize=10)

        mutants = [r.mutant for r in valid_results]
        
        # Format positions for x-axis labels
        if 'position' in df.columns and 'amino_acid' in df.columns:
            positions = []
            for pos, aa in zip(df['position'], df['amino_acid']):
                positions.append(f"{pos}-{aa}")
        else:
            positions = [f"Pos{i+1}" for i in range(len(df))]
        
        # Set ticks - for specific regions, show all positions
        if is_specific_region or n_positions <= 30:
            x_ticks = np.arange(n_positions)
            x_labels = positions
            font_size = 9 if n_positions <= 20 else 8
        else:
            # For large datasets, sample ticks
            x_tick_interval = max(1, n_positions // 20)
            x_ticks = np.arange(0, n_positions, x_tick_interval)
            x_labels = [positions[i] for i in range(0, n_positions, x_tick_interval)]
            font_size = 8
        
        # Set y-ticks
        if n_mutants <= 30:
            y_ticks = np.arange(n_mutants)
            y_labels = mutants
            y_font_size = 9
        else:
            y_tick_interval = max(1, n_mutants // 15)
            y_ticks = np.arange(0, n_mutants, y_tick_interval)
            y_labels = [mutants[i] for i in range(0, n_mutants, y_tick_interval)]
            y_font_size = 8
        
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_labels, rotation=45 if is_specific_region else 90, 
                          ha='right' if is_specific_region else 'center', 
                          fontsize=font_size)
        ax.set_yticklabels(y_labels, fontsize=y_font_size)
        
        # Add cell text annotations for specific regions or when requested
        if (show_labels or is_specific_region) and n_positions <= 30 and n_mutants <= 20:
            for i in range(len(mutants)):
                for j in range(len(positions)):
                    if not np.isnan(delta_matrix[i, j]):
                        delta_val = delta_matrix[i, j]
                        norm_val = abs(delta_val) / delta_max if delta_max > 0 else 0
                        color = 'white' if norm_val > 0.5 else 'black'
                        ax.text(j, i, f'{delta_val:.2f}',
                               ha='center', va='center', color=color,
                               fontsize=7, fontweight='bold')
        
        ax.set_title(f'Δ Scores Heatmap{title_suffix}', fontsize=12, pad=10)
        ax.set_xlabel('Position', fontsize=11)
        ax.set_ylabel('Mutant', fontsize=11)
        
        # Add grid for better readability in specific regions
        if is_specific_region:
            ax.set_xticks(np.arange(-0.5, n_positions, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n_mutants, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close(fig)


    def plot_distribution(self, results: List[AnalysisResult], output_path: str, region_name: str = ""):
        """Plot distribution of delta scores using only non-zero values."""
        if region_name:
            title_suffix = f" ({region_name})"
        else:
            title_suffix = ""
        
        fig, ax = plt.subplots(figsize=(10, 6))
    
        plot_data = []
        for result in results:
            # Use only non-zero delta scores
            for delta in result.non_zero_delta_scores:
                if not np.isnan(delta):
                    plot_data.append({'Mutant': result.mutant, 'Δ': delta})
    
        if not plot_data:
            print("Warning: No non-zero delta data for distribution plot")
            return
    
        plot_df = pd.DataFrame(plot_data)
    
        # Assign color palette for each mutant
        palette_dict = {}
        for result in results:
            palette_dict[result.mutant] = self.COLORS['significant'] if result.significant else self.COLORS['nonsignificant']
    
        # Violin plot vs boxplot if number of points is suffice
        unique_mutants = plot_df['Mutant'].unique()
        if len(unique_mutants) > 1 and len(plot_df) > len(unique_mutants) * 2:
            sns.violinplot(x='Mutant', y='Δ', data=plot_df, hue='Mutant', 
                          ax=ax, palette=palette_dict, inner='quartile', legend=False)
        else:
            sns.boxplot(x='Mutant', y='Δ', data=plot_df, hue='Mutant',
                       ax=ax, palette=palette_dict, legend=False)

        sns.stripplot(x='Mutant', y='Δ', data=plot_df, ax=ax, 
                     color='black', alpha=0.5, size=3, jitter=0.2)
        
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Mutant', fontsize=11)
        ax.set_ylabel('Δ Score', fontsize=11)
        ax.set_title(f'Distribution of Δ Scores (Non-zero only){title_suffix}', fontsize=12, pad=10)
        
        x_positions = np.arange(len(unique_mutants))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(unique_mutants, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y', color=self.COLORS['grid'])
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close(fig)


    def create_output_directory(self, input_file: str, output_prefix: str) -> str:
        """Create output directory and return path."""
        input_filename = os.path.splitext(os.path.basename(input_file))[0]
        dir_name = f"{output_prefix}_{input_filename}"
        os.makedirs(dir_name, exist_ok=True)
        return dir_name


    def export_results(self, df: pd.DataFrame, results: List[AnalysisResult], output_dir: str):
        """Export all results to CSV files."""
        # Export delta scores
        delta_df = df.copy()
        for result in results:
            delta_df[f'{result.mutant}_delta'] = result.delta_scores
        delta_df.to_csv(os.path.join(output_dir, 'delta_scores.csv'), index=False)
        # Export summary statistics
        summary_df = self.create_summary_dataframe(results)
        summary_df.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
        # Export raw data
        df.to_csv(os.path.join(output_dir, 'original_data.csv'), index=False)
        # Export non-zero region data if available
        if self.non_zero_positions is not None:
            non_zero_df = df.iloc[self.non_zero_positions].copy()
            non_zero_df.to_csv(os.path.join(output_dir, 'non_zero_region_data.csv'), index=False)
        # Export region information
        if self.regions:
            region_info = []
            for region in self.regions:
                region_info.append({
                    'name': region.name,
                    'start': region.start,
                    'end': region.end,
                    'n_positions': len(region.positions),
                    'positions': ','.join(map(str, region.positions))})
            region_df = pd.DataFrame(region_info)
            region_df.to_csv(os.path.join(output_dir, 'regions.csv'), index=False)
        # Save a README file with analysis information
        self._save_readme(output_dir, summary_df)


    def _save_readme(self, output_dir: str, summary_df: pd.DataFrame):
        """Save a README file with analysis information."""
        num_positions = len(self.df) if self.df is not None else 0
        num_mutants = len(self.mutants) if self.mutants else 0
        n_significant = sum(summary_df['Significant']) if 'Significant' in summary_df.columns else 0
        n_total = len(summary_df) if not summary_df.empty else 0
        
        if self.non_zero_positions is not None:
            n_non_zero = len(self.non_zero_positions)
            non_zero_percentage = (n_non_zero / num_positions * 100) if num_positions > 0 else 0
        else:
            n_non_zero = "N/A"
            non_zero_percentage = "N/A"
        
        region_info = ""
        if self.regions:
            region_info = "\nRegions for Detailed Analysis:\n"
            for region in self.regions:
                region_info += f"  {region.name}: positions {region.start}-{region.end} ({len(region.positions)} aa)\n"
        
        readme_content = f"""Wilcoxon Test Analysis Results

Analysis Parameters
Significance level (alpha): {self.alpha}
Number of positions analyzed: {num_positions}
Number of mutants analyzed: {num_mutants}
Non-zero delta positions: {n_non_zero} ({non_zero_percentage:.1f}% of total)
{region_info}
Analysis Strategy
1. Whole sequence analysis: Visualizations include raw scores and delta heatmaps
2. Non-zero region analysis: Statistical tests and detailed visualizations using only positions with non-zero delta scores
   (Threshold for non-zero: |Δ| > 0.03)
3. Specific region analysis: Detailed analysis of user-specified or automatically detected regions

Files Generated
Whole Sequence Analysis:
1. original_data.csv: Cleaned original metascore data
2. initial_scores.png: Raw scores distribution across sequence
3. delta_heatmap.png: Delta scores heatmap for whole sequence

Non-Zero Region Analysis:
4. non_zero_region_data.csv: Data for positions with non-zero delta scores
5. summary_statistics.csv: Statistical analysis results (using non-zero deltas only)
6. delta_scores.csv: Delta scores for all positions
7. pvalues_plot.png: P-values and mean delta scores visualization
8. delta_boxplot.png: Box plot of delta scores distribution
9. delta_distribution.png: KDE distribution of delta scores by violin plot

Specific Region Analysis:
1. region_[name]_heatmap.png: Detailed delta heatmap for the region
2. region_[name]_pvalues.png: P-values plot for the region
3. region_[name]_boxplot.png: Box plot for the region

Statistical Summary
Total significant mutants (p < {self.alpha}): {n_significant}
Total non-significant mutants: {n_total - n_significant}

Interpretation
Delta = Mutant score - WT score
Negative Delta: Mutant has LOWER metascore than WT
Positive Delta: Mutant has HIGHER metascore than WT
p < {self.alpha}: Significant difference from WT
p >= {self.alpha}: No significant difference from WT

Note: Statistical tests use only non-zero delta scores to avoid bias from zero-inflated data.

Analysis performed on: {time.strftime('%Y-%m-%d %H:%M:%S')}
Default DPI for pictures used: 600 
"""
        with open(os.path.join(output_dir, 'README.txt'), 'w') as f:
            f.write(readme_content)

    def analyze_region(self, region: Region, results: List[AnalysisResult], output_dir: str):
        """Perform detailed analysis for a specific region."""
        print(f"\nAnalyzing region: {region.name} (positions {region.start}-{region.end})")
        
        # Get subset of data for this region
        region_df = region.get_subset(self.df)
        if len(region_df) == 0:
            print(f"  Warning: No data found for region {region.name}")
            return
        
        # Store original indices for this region
        if 'position' in self.df.columns:
            # Find indices based on position
            region_positions_set = set(region.positions)
            region_indices = []
            for idx, pos in enumerate(self.df['position']):
                if pos in region_positions_set:
                    region_indices.append(idx)
            region_indices = np.array(region_indices)
        else:
            region_indices = np.array(region.positions - 1 if region.positions[0] > 0 else region.positions)
        
        self.region_original_indices[region.name] = region_indices
        
        # Create results for this specific region
        region_results = []
        for result in results:
            if len(result.non_zero_delta_scores) == 0:
                # Skip mutants with no non-zero delta scores
                continue
            
            # Get delta scores for this region using the original indices
            if len(region_indices) > 0:
                try:
                    region_delta_scores = result.delta_scores[region_indices]
                except IndexError as e:
                    print(f"  Warning: Index error for {result.mutant} in region {region.name}: {e}")
                    continue
            else:
                print(f"  Warning: No valid indices found for region {region.name}")
                continue
            
            # Filter non-zero delta scores
            non_zero_mask = np.abs(region_delta_scores) > 0.01
            non_zero_delta_scores = region_delta_scores[non_zero_mask]
            
            if len(non_zero_delta_scores) >= 2:  # Need at least 2 for Wilcoxon
                try:
                    stat, p_value = wilcoxon(non_zero_delta_scores, 
                                            zero_method='wilcox', 
                                            alternative='two-sided')
                except:
                    stat, p_value = 0.0, 1.0
            else:
                stat, p_value = 0.0, 1.0
            
            region_result = AnalysisResult(
                mutant=result.mutant,
                delta_scores=region_delta_scores,
                non_zero_delta_scores=non_zero_delta_scores,
                non_zero_positions=np.where(non_zero_mask)[0],
                mean_delta=np.nanmean(non_zero_delta_scores),
                median_delta=np.nanmedian(non_zero_delta_scores),
                std_delta=np.nanstd(non_zero_delta_scores),
                wilcoxon_stat=stat,
                p_value=p_value,
                significant=p_value < self.alpha)
            region_results.append(region_result)
        
        if not region_results:
            print(f"  Warning: No valid results for region {region.name}")
            return
        
        # Create summary dataframe
        region_summary_df = self.create_summary_dataframe(region_results)
        # Create visualizations for this region
        region_dir = os.path.join(output_dir, f"region_{region.name}")
        os.makedirs(region_dir, exist_ok=True)
        # Plot delta heatmap for this specific region
        self.plot_delta_heatmap(region_df, region_results,
                              os.path.join(region_dir, f'delta_heatmap.png'),
                              show_labels=True, region_name=region.name,
                              is_specific_region=True,
                              region_indices=None)
        # Plot p-values for this region
        self.plot_pvalues(region_summary_df,
                         os.path.join(region_dir, f'pvalues_plot.png'),
                         region_name=region.name)
        # Plot boxplot for this region
        self.plot_delta_boxplot(region_summary_df, region_results,
                              os.path.join(region_dir, f'delta_boxplot.png'),
                              region_name=region.name)
        # Export region-specific data
        region_df.to_csv(os.path.join(region_dir, f'region_data.csv'), index=False)
        region_summary_df.to_csv(os.path.join(region_dir, f'region_summary.csv'), index=False)
        print(f"  Region analysis saved to: {region_dir}/")
    
    def cleanup(self):
        """Clean up resources to free memory."""
        self.df = None
        self.results.clear()
        self.results_df = None
        self.regions.clear()
        self.region_original_indices.clear()
        plt.close('all')
        gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description='Wilcoxon test analysis for amyloidogenic delta scores calculations.\n'
        'This script will conduct statistical analysis, results visualization\n'
        'and save all data to a specified directory.',
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True)
    
    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('input_file', help='Path to CSV file')

    # Analysis parameters
    analysis = parser.add_argument_group('analysis parameters')
    analysis.add_argument('--alpha', type=float, default=0.05, 
                         help='Significance level (default: 0.05)')
    analysis.add_argument('--delta-threshold', type=float, default=0.03,
                         help='Threshold for considering delta scores as non-zero (default: 0.03)')
    
    # Output parameters
    output = parser.add_argument_group('output parameters')
    output.add_argument('--output', default='analysis', 
                       help='Output directory prefix (default: analysis)')
    output.add_argument('--no-plot', action='store_true', 
                       help='Skip plotting (faster analysis)')
    
    # Region detection parameters
    regions = parser.add_argument_group('region detection')
    regions.add_argument('--regions', nargs='+', default=[],
                        help='Custom regions for detailed analysis.\n'
                        'Format: "name:start-end" or "name:pos1,pos2,pos3"\n'
                        'Example: --regions "Hotspot1:30-45" "BindingSite:50,51,52,53,54"')
    regions.add_argument('--auto-regions', type=int, default=2,
                        help='Number of regions to automatically detect (default: 2)\n'
                        'Set to 0 to disable automatic region detection')
    regions.add_argument('--region-min-size', type=int, default=4,
                        help='Minimum region size for automatic detection (default: 4)')
    regions.add_argument('--region-max-size', type=int, default=8,
                        help='Maximum region size for automatic detection (default: 8)')
    
    args = parser.parse_args()
    
    # Validate arguments early
    if not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' not found")
        sys.exit(1)
    if args.alpha <= 0 or args.alpha >= 1:
        print(f"Error: Alpha must be between 0 and 1, got {args.alpha}")
        sys.exit(1)
    
    print("="*60)
    print("WILCOXON TEST ANALYSIS")
    print("="*60)
    
    analyzer = MetascoreAnalyzer(alpha=args.alpha)
    
    print(f"Reading data from: {args.input_file}")
    raw_df = analyzer.read_csv(args.input_file)
    df, mutants = analyzer.prepare_data(raw_df)
    
    analyzer.df = df
    analyzer.mutants = mutants
    
    print(f"Mutants to analyze: {mutants}")
    print(f"Delta threshold for non-zero: {args.delta_threshold}")
    
    output_dir = analyzer.create_output_directory(args.input_file, args.output)
    print(f"Output directory: {output_dir}")
    
    # Parse custom regions
    if args.regions:
        print("\nParsing custom regions...")
        for region_str in args.regions:
            region = analyzer.parse_region_argument(region_str)
            if region:
                analyzer.regions.append(region)
                print(f"  Added region: {region}")
    
    # Find non-zero regions
    print("\nIdentifying non-zero delta regions...")
    non_zero_positions, _ = analyzer.find_non_zero_regions(df, mutants, args.delta_threshold)
    analyzer.non_zero_positions = non_zero_positions
    
    # Find top regions automatically if requested
    if args.auto_regions > 0:
        auto_regions = analyzer.find_top_regions(
            df, mutants, 
            n_regions=args.auto_regions,
            min_size=args.region_min_size,
            max_size=args.region_max_size,
            threshold=args.delta_threshold
        )
        analyzer.regions.extend(auto_regions)
    
    print(f"\nTotal regions for detailed analysis: {len(analyzer.regions)}")
    for region in analyzer.regions:
        print(f"  {region}")
    
    print("\nPerforming Wilcoxon analysis (using non-zero deltas only)...")
    results = analyzer.perform_analysis(df, mutants, use_non_zero_only=True, threshold=args.delta_threshold)
    analyzer.results = results
    
    if not results:
        print("Error: No valid results obtained from analysis")
        sys.exit(1)
    
    summary_df = analyzer.create_summary_dataframe(results)
    analyzer.results_df = summary_df
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(summary_df[['Mutant', 'Mean_Δ', 'P_value', 'Significant', 'NonZero_Count']].to_string(
        index=False, 
        float_format=lambda x: f'{x:.6f}' if isinstance(x, float) else str(x)))
    
    n_significant = sum(summary_df['Significant'])
    n_total = len(summary_df)
    print(f"\nSignificant mutants (p < {args.alpha}): {n_significant}/{n_total} ({n_significant/n_total*100:.1f}%)")
    
    if not args.no_plot:
        print("\nCreating visualizations...")
        
        # Raw metascores mapped on whole sequence
        analyzer.plot_initial_scores(df, mutants, os.path.join(output_dir, 'initial_scores.png'))
        print("  Initial scores plot saved")
        # Heatmap for delta scores
        analyzer.plot_delta_heatmap(df, results, os.path.join(output_dir, 'delta_heatmap.png'), 
                                  show_labels=False, region_name="Full Sequence")
        print("  Full sequence delta heatmap saved")

        # Non-zero region visualizations
        analyzer.plot_pvalues(summary_df, os.path.join(output_dir, 'pvalues_plot.png'), 
                            region_name="Non-zero Region")
        print("  P-values plot saved (non-zero region)")
        
        analyzer.plot_delta_boxplot(summary_df, results, os.path.join(output_dir, 'delta_boxplot.png'),
                                  region_name="Non-zero Region")
        print("  Delta scores boxplot saved (non-zero region)")
        
        analyzer.plot_distribution(results, os.path.join(output_dir, 'delta_distribution.png'),
                                 region_name="Non-zero Region")
        print("  Delta distribution plot saved (non-zero region)")
        
        # Region-specific analyses
        for region in analyzer.regions:
            analyzer.analyze_region(region, results, output_dir)
    
    analyzer.export_results(df, results, output_dir)
    print("\nResults exported to CSV files")
    
    # Cleanup resources
    analyzer.cleanup()
    
    print("\n" + "="*60)
    print(f"ANALYSIS COMPLETE")
    print(f"Results saved in: {output_dir}/")
    print("="*60)
if __name__ == "__main__":
    main()