import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import warnings
import argparse
import sys
import os
from typing import Tuple, List
from dataclasses import dataclass
import time
import pathlib


@dataclass
class AnalysisResult:
    """Data class to store analysis results for each mutant."""
    mutant: str
    delta_scores: np.ndarray
    mean_delta: float
    median_delta: float
    std_delta: float
    wilcoxon_stat: float
    p_value: float
    significant: bool


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


    def read_csv(self, filepath: str) -> pd.DataFrame:
        """Read and validate CSV"""
        try:
            df = pd.read_csv(filepath, sep=';', low_memory=False)
            
            if 'aa' not in df.columns or 'aa_name' not in df.columns:
                print("Warning: Some expected columns are missing")
            
            return df
        except Exception as e:
            print(f"Error reading CSV: {e}")
            sys.exit(1)


    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Extract and prepare metascore data"""
        metascore_mask = df.columns.str.contains('metascore', case=False, na=False)
        metascore_cols = df.columns[metascore_mask].tolist()
        
        if not metascore_cols:
            raise ValueError("No metascore columns found")
        
        wt_mask = df.columns.str.contains('metascore_wt', case=False, na=False)
        wt_cols = df.columns[wt_mask].tolist()
        
        if not wt_cols:
            raise ValueError("WT column not found")
        wt_col = wt_cols[0]
        mutant_cols = [col for col in metascore_cols if col != wt_col]
        if not mutant_cols:
            raise ValueError("No mutant columns found")
        
        clean_names = [col.replace('metascore_', '').strip() for col in mutant_cols]
        
        new_data = {
            'position': df.get('aa', pd.Series(range(1, len(df) + 1))),
            'amino_acid': df.get('aa_name', pd.Series([''] * len(df))),
            'WT': df[wt_col]
        }
        
        for orig_col, clean_name in zip(mutant_cols, clean_names):
            new_data[clean_name] = df[orig_col]
        
        return pd.DataFrame(new_data), clean_names


    def perform_analysis(self, df: pd.DataFrame, mutants: List[str]) -> List[AnalysisResult]:
        """Perform Wilcoxon analysis for mutants"""
        results = []
        
        for mutant in mutants:
            try:
                delta_scores = (df[mutant] - df['WT']).values
                
                if np.isnan(delta_scores).all() or np.all(delta_scores == delta_scores[0]):
                    print(f"Warning: {mutant} has invalid data for analysis")
                    continue
                
                non_zero_differences = np.count_nonzero(delta_scores)
                if non_zero_differences < 2:
                    print(f"Warning: {mutant} has insufficient non-zero differences ({non_zero_differences})")
                    stat, p_value = 0.0, 1.0
                else:
                    stat, p_value = wilcoxon(
                        delta_scores, 
                        zero_method='wilcox', 
                        alternative='two-sided'
                    )
                
                mean_delta = np.nanmean(delta_scores)
                median_delta = np.nanmedian(delta_scores)
                std_delta = np.nanstd(delta_scores)
                
                result = AnalysisResult(
                    mutant=mutant,
                    delta_scores=delta_scores,
                    mean_delta=mean_delta,
                    median_delta=median_delta,
                    std_delta=std_delta,
                    wilcoxon_stat=stat,
                    p_value=p_value,
                    significant=p_value < self.alpha
                )
                
                results.append(result)
                
            except Exception as e:
                print(f"Error analyzing {mutant}: {str(e).split(':')[0]}")
                result = AnalysisResult(
                    mutant=mutant,
                    delta_scores=np.array([np.nan] * len(df)),
                    mean_delta=np.nan,
                    median_delta=np.nan,
                    std_delta=np.nan,
                    wilcoxon_stat=np.nan,
                    p_value=np.nan,
                    significant=False
                )
                results.append(result)
        
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
                'Significant': result.significant
            })
        
        return pd.DataFrame(summary_data)


    def plot_pvalues(self, results_df: pd.DataFrame, output_path: str):
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
        ax1.set_title('Wilcoxon Test P-values')
        ax1.set_xticks(x)
        ax1.set_xticklabels(results_df['Mutant'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y', color=self.COLORS['grid'])
        
        mean_deltas = results_df['Mean_Δ'].values
        std_deltas = results_df['Std_Δ'].values
        
        colors_delta = np.where(mean_deltas < 0, self.COLORS['negative'], self.COLORS['positive'])
        
        bars = ax2.bar(x, mean_deltas, yerr=std_deltas, capsize=5, 
                      color=colors_delta, edgecolor='black', width=0.7)
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
        ax2.set_title('Mean Δ Scores with Error Bars')
        ax2.set_xticks(x)
        ax2.set_xticklabels(results_df['Mutant'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y', color=self.COLORS['grid'])
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close(fig)


    def plot_delta_boxplot(self, results_df: pd.DataFrame, results: List[AnalysisResult], output_path: str):
        """Create boxplot of delta scores for each mutant with legend for mean markers"""
        fig, ax = plt.subplots(figsize=(12, 6))
    
        box_data = []
        box_labels = []
        box_colors = []
    
        for idx, (result, row) in enumerate(zip(results, results_df.iterrows())):
            row_data = row[1]
            mutant = row_data['Mutant']
            delta_scores = result.delta_scores
        
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
        # Creating two markers - zero line and a point of mean delta value and adding this to the legend
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
        ax.set_title('Distribution of Δ Scores (Boxplot)')
        ax.set_xticklabels(box_labels, rotation=45, ha='right')
    
        if mean_marker is not None:
            ax.legend([zero_line, mean_marker], ['Δ = 0', 'Mean Δ'], loc='upper right')

        ax.grid(True, alpha=0.3, axis='y', color=self.COLORS['grid'])
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


    def plot_initial_scores(self, df: pd.DataFrame, mutants: List[str], output_path: str):
        """Plot initial WT and mutant scores across amino acid sequence"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        if 'position' in df.columns and 'amino_acid' in df.columns:
            positions = [f"{pos}-{aa}" for pos, aa in zip(df['position'], df['amino_acid'])]
        else:
            positions = [f"Pos{i+1}" for i in range(len(df))]
        
        x = np.arange(len(positions))
        
        # Line plot of all scores
        ax1.plot(x, df['WT'], label='WT', color=self.COLORS['wt'], 
                linewidth=2.5, marker='o', markersize=6)
        
        for mutant in mutants:
            ax1.plot(x, df[mutant], label=mutant, alpha=0.7, 
                    linewidth=1.5, marker='s', markersize=4)
        
        ax1.set_xlabel('Position - Amino Acid')
        ax1.set_ylabel('Metascore')
        ax1.set_title('Metascore Distribution Across Amino Acid Sequence')
        ax1.set_xticks(x)
        ax1.set_xticklabels(positions, rotation=45, ha='right', fontsize=8)
        ax1.legend(fontsize=9, ncol=3, loc='upper right')
        ax1.grid(True, alpha=0.3, color=self.COLORS['grid'])
        
        # Heatmap of all scores
        all_variants = ['WT'] + mutants
        score_matrix = df[all_variants].T.values
        
        im = ax2.imshow(score_matrix, cmap='viridis', aspect='auto', 
                       interpolation='nearest')
        
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Metascore', fontsize=10)
        
        ax2.set_xticks(x)
        ax2.set_yticks(np.arange(len(all_variants)))
        ax2.set_xticklabels(positions, rotation=45, ha='right', fontsize=8)
        ax2.set_yticklabels(all_variants, fontsize=9)
        
        # Some text annotations for all cells
        for i in range(len(all_variants)):
            for j in range(len(positions)):
                score = score_matrix[i, j]
                # Determine text color based on background brightness
                # Normalize score to 0-1 for colormap
                norm_score = (score - np.nanmin(score_matrix)) / (np.nanmax(score_matrix) - np.nanmin(score_matrix))
                # Use white text for dark backgrounds (norm_score < 0.5), black otherwise
                color = 'white' if norm_score < 0.5 else 'black'
                ax2.text(j, i, f'{score:.3f}', ha='center', va='center', 
                        color=color, fontsize=7, fontweight='bold')
        
        ax2.set_xlabel('Position - Amino Acid')
        ax2.set_title('Metascore Heatmap (Raw Scores)')
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close(fig)


    def plot_delta_heatmap(self, df: pd.DataFrame, results: List[AnalysisResult], output_path: str):
        delta_matrix = np.array([r.delta_scores for r in results])
        if delta_matrix.size == 0 or np.all(np.isnan(delta_matrix)):
            return
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(delta_matrix, cmap='RdBu_r', aspect='auto',
                      vmin=-np.nanmax(np.abs(delta_matrix)),
                      vmax=np.nanmax(np.abs(delta_matrix)))
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('Δ Value')

        mutants = [r.mutant for r in results]
        if 'position' in df.columns and 'amino_acid' in df.columns:
            positions = [f"{pos}-{aa}" for pos, aa in zip(df['position'], df['amino_acid'])]
        else:
            positions = [f"Pos{i+1}" for i in range(len(df))]
        
        ax.set_xticks(np.arange(len(positions)))
        ax.set_yticks(np.arange(len(mutants)))
        ax.set_xticklabels(positions, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(mutants, fontsize=9)
        
        for i in range(len(mutants)):
            for j in range(len(positions)):
                if not np.isnan(delta_matrix[i, j]):
                    # For delta scores, use absolute value > 0.5 * max for white text
                    norm_val = abs(delta_matrix[i, j]) / np.nanmax(np.abs(delta_matrix))
                    color = 'white' if norm_val > 0.5 else 'black'
                    ax.text(j, i, f'{delta_matrix[i, j]:.2f}',
                           ha='center', va='center', color=color,
                           fontsize=7, fontweight='bold')
        ax.set_title('Δ Scores Heatmap by Position')
        plt.tight_layout()
        fig.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close(fig)


    def plot_distribution(self, results: List[AnalysisResult], output_path: str):
        fig, ax = plt.subplots(figsize=(10, 6))
    
        plot_data = []
        for result in results:
            for delta in result.delta_scores:
                if not np.isnan(delta):
                    plot_data.append({'Mutant': result.mutant, 'Δ': delta})
    
        if not plot_data:
            return
    
        plot_df = pd.DataFrame(plot_data)
    
        # Assiging color palette for each mutant
        palette_dict = {}
        for result in results:
            palette_dict[result.mutant] = self.COLORS['significant'] if result.significant else self.COLORS['nonsignificant']
    
        sns.violinplot(x='Mutant', y='Δ', data=plot_df, hue='Mutant', ax=ax, palette=palette_dict, inner='quartile', legend=False)
        sns.stripplot(x='Mutant', y='Δ', data=plot_df, ax=ax, color='black', alpha=0.5, size=3, jitter=0.2)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Mutant')
        ax.set_ylabel('Δ Score')
        ax.set_title('Distribution of Δ Scores (Violin Plot)')
        unique_mutants = plot_df['Mutant'].unique()
        x_positions = np.arange(len(unique_mutants))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(unique_mutants, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y', color=self.COLORS['grid'])
        plt.tight_layout()
        fig.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close(fig)


    def create_output_directory(self, input_file: str, output_prefix: str) -> str:
        """Create output directory and return path"""
        input_filename = os.path.splitext(os.path.basename(input_file))[0]
        dir_name = f"{output_prefix}_{input_filename}"
        os.makedirs(dir_name, exist_ok=True)
        return dir_name


    def export_results(self, df: pd.DataFrame, results: List[AnalysisResult], output_dir: str):
        """Export all results to CSV files"""
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
        
        # Save a README file with analysis information
        self._save_readme(output_dir, summary_df)
    
    def _save_readme(self, output_dir: str, summary_df: pd.DataFrame):
        """Save a README file with analysis information"""
        num_positions = len(self.df) if self.df is not None else 0
        num_mutants = len(self.mutants) if self.mutants else 0
        n_significant = sum(summary_df['Significant']) if 'Significant' in summary_df.columns else 0
        n_total = len(summary_df) if not summary_df.empty else 0
        
        readme_content = f"""Wilcoxon Test Analysis Results

Analysis Parameters
Significance level (alpha): {self.alpha}
Number of positions analyzed: {num_positions}
Number of mutants analyzed: {num_mutants}

Files Generated
1. original_data.csv: Cleaned original metascore data
2. delta_scores.csv: Delta scores (Mutant - WT) for each position
3. summary_statistics.csv: Statistical analysis results
4. pvalues_plot.png: P-values and mean delta scores visualization
5. delta_boxplot.png: Box plot of delta scores distribution
6. initial_scores.png: Raw scores distribution across sequence
7. delta_heatmap.png: Delta scores heatmap by position
8. delta_distribution.png: Distribution of delta scores (violin plot)

Statistical Summary
Total significant mutants (p < {self.alpha}): {n_significant}
Total non-significant mutants: {n_total - n_significant}

Interpretation
Delta = Mutant score - WT score
Negative Delta: Mutant has LOWER metascore than WT
Positive Delta: Mutant has HIGHER metascore than WT
p < {self.alpha}: Significant difference from WT
p >= {self.alpha}: No significant difference from WT

Analysis performed on: {time.strftime('%Y-%m-%d %H:%M:%S')}
Default DPI for pictures used: 600 
"""

        with open(os.path.join(output_dir, 'README.txt'), 'w') as f:
            f.write(readme_content)


def main():
    parser = argparse.ArgumentParser(description='Wilcoxon test analysis for amyloidogenic delta scores calculations.\n'
    'This script will conduct statistical analysis, results vizualization\n'
    'and save all data to a specified directory.', 
    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_file', help='Path to CSV file')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    parser.add_argument('--output', default='analysis', help='Output prefix')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' not found")
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
    
    output_dir = analyzer.create_output_directory(args.input_file, args.output)
    print(f"Output directory: {output_dir}")
    
    print("\nPerforming Wilcoxon analysis...")
    results = analyzer.perform_analysis(df, mutants)
    analyzer.results = results
    
    if not results:
        print("Error: No valid results obtained from analysis")
        sys.exit(1)
    
    summary_df = analyzer.create_summary_dataframe(results)
    analyzer.results_df = summary_df
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(summary_df[['Mutant', 'Mean_Δ', 'P_value', 'Significant']].to_string(index=False, 
          float_format=lambda x: f'{x:.6f}' if isinstance(x, float) else str(x)))
    
    n_significant = sum(summary_df['Significant'])
    n_total = len(summary_df)
    print(f"\nSignificant mutants (p < {args.alpha}): {n_significant}/{n_total} ({n_significant/n_total*100:.1f}%)")
    
    if not args.no_plot:
        print("\nCreating visualizations...")
        
        analyzer.plot_initial_scores(df, mutants, os.path.join(output_dir, 'initial_scores.png'))
        print("  Initial scores plot saved")
        
        analyzer.plot_pvalues(summary_df, os.path.join(output_dir, 'pvalues_plot.png'))
        print("  P-values violin plot saved")
        
        analyzer.plot_delta_boxplot(summary_df, results, os.path.join(output_dir, 'delta_boxplot.png'))
        print("  Delta scores boxplot saved")
             
        analyzer.plot_delta_heatmap(df, results, os.path.join(output_dir, 'delta_heatmap.png'))
        print("  Delta scores heatmap saved")
        
        analyzer.plot_distribution(results, os.path.join(output_dir, 'delta_distribution.png'))
        print("  Delta distribution plot saved")
    
    analyzer.export_results(df, results, output_dir)
    print("\nResults exported to CSV files")
    
    print("\n" + "="*60)
    print(f"ANALYSIS COMPLETE")
    print(f"Results saved in: {output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()