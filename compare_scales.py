"""
Multi-Scale Comparison Analysis
Compares results across 5, 10, and 20 agent simulations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List


class ScaleComparator:
    """Compare simulation results across different scales"""
    
    def __init__(self):
        self.scales = {
            5: Path("data/run_5_agents"),
            10: Path("data/run_10_agents"),
            20: Path("data/run_20_agents")
        }
        self.results = {}
    
    def load_all_data(self):
        """Load data from all scales"""
        print("\n📂 Loading data from all scales...")
        
        for scale, data_dir in self.scales.items():
            if not data_dir.exists():
                print(f"   ⚠️  {scale}-agent data not found")
                continue
            
            self.results[scale] = self._load_scale_data(scale, data_dir)
            print(f"   ✅ Loaded {scale}-agent data")
    
    def _load_scale_data(self, scale: int, data_dir: Path) -> Dict:
        """Load data for one scale"""
        data = {
            'scale': scale,
            'similarity_matrices': {},
            'coalitions': {},
            'agent_count': scale
        }
        
        # Load similarity matrices
        sim_dir = data_dir / "similarity"
        if sim_dir.exists():
            for file_path in sim_dir.glob("matrix_round_*.json"):
                with open(file_path, 'r') as f:
                    sim_data = json.load(f)
                    data['similarity_matrices'][sim_data['round']] = np.array(sim_data['matrix'])
        
        # Load coalitions
        coal_dir = data_dir / "coalitions"
        if coal_dir.exists():
            for file_path in coal_dir.glob("coalitions_round_*.json"):
                with open(file_path, 'r') as f:
                    coal_data = json.load(f)
                    data['coalitions'][coal_data['round']] = coal_data['coalitions']
        
        return data
    
    def compare_coalition_formation(self) -> Dict:
        """Compare coalition formation across scales"""
        print("\n📊 Analyzing coalition formation patterns...")
        
        comparison = {}
        
        for scale, data in self.results.items():
            if not data['coalitions']:
                continue
            
            # Get first coalition formation
            first_round = min(data['coalitions'].keys())
            coalitions = data['coalitions'][first_round]
            
            comparison[scale] = {
                'num_coalitions': len(coalitions),
                'avg_coalition_size': np.mean([len(c) for c in coalitions]),
                'min_coalition_size': min([len(c) for c in coalitions]),
                'max_coalition_size': max([len(c) for c in coalitions]),
                'coalition_size_std': np.std([len(c) for c in coalitions])
            }
            
            print(f"\n   {scale} agents:")
            print(f"      Coalitions formed: {comparison[scale]['num_coalitions']}")
            print(f"      Avg size: {comparison[scale]['avg_coalition_size']:.1f}")
        
        return comparison
    
    def compare_similarity_evolution(self) -> Dict:
        """Compare how similarity evolves across scales"""
        print("\n📊 Analyzing similarity evolution...")
        
        comparison = {}
        
        for scale, data in self.results.items():
            if not data['similarity_matrices']:
                continue
            
            rounds = sorted(data['similarity_matrices'].keys())
            avg_similarities = []
            
            for round_num in rounds:
                matrix = data['similarity_matrices'][round_num]
                upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
                avg_similarities.append(np.mean(upper_tri))
            
            comparison[scale] = {
                'rounds': rounds,
                'avg_similarities': avg_similarities,
                'initial_similarity': avg_similarities[0] if avg_similarities else 0,
                'final_similarity': avg_similarities[-1] if avg_similarities else 0,
                'similarity_increase': (avg_similarities[-1] - avg_similarities[0]) if len(avg_similarities) > 1 else 0
            }
            
            print(f"\n   {scale} agents:")
            print(f"      Initial: {comparison[scale]['initial_similarity']:.3f}")
            print(f"      Final: {comparison[scale]['final_similarity']:.3f}")
            print(f"      Change: {comparison[scale]['similarity_increase']:+.3f}")
        
        return comparison
    
    def generate_comparison_visualizations(self):
        """Create comparison charts"""
        print("\n🎨 Generating comparison visualizations...")
        
        output_dir = Path("visualizations/scale_comparison")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Chart 1: Coalition counts
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        scales = sorted(self.results.keys())
        
        # Number of coalitions
        coalition_counts = []
        for scale in scales:
            if self.results[scale]['coalitions']:
                first_round = min(self.results[scale]['coalitions'].keys())
                coalition_counts.append(len(self.results[scale]['coalitions'][first_round]))
            else:
                coalition_counts.append(0)
        
        ax1.bar(scales, coalition_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], 
               edgecolor='black', alpha=0.8)
        ax1.set_xlabel('Number of Agents', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Coalitions', fontsize=12, fontweight='bold')
        ax1.set_title('Coalition Count by Scale', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Average coalition size
        avg_sizes = []
        for scale in scales:
            if self.results[scale]['coalitions']:
                first_round = min(self.results[scale]['coalitions'].keys())
                coalitions = self.results[scale]['coalitions'][first_round]
                avg_sizes.append(np.mean([len(c) for c in coalitions]))
            else:
                avg_sizes.append(0)
        
        ax2.bar(scales, avg_sizes, color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
               edgecolor='black', alpha=0.8)
        ax2.set_xlabel('Number of Agents', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Average Coalition Size', fontsize=12, fontweight='bold')
        ax2.set_title('Avg Coalition Size by Scale', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Similarity evolution
        for scale in scales:
            if self.results[scale]['similarity_matrices']:
                rounds = sorted(self.results[scale]['similarity_matrices'].keys())
                avg_sims = []
                
                for round_num in rounds:
                    matrix = self.results[scale]['similarity_matrices'][round_num]
                    upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
                    avg_sims.append(np.mean(upper_tri))
                
                ax3.plot(rounds, avg_sims, marker='o', linewidth=2, 
                        markersize=8, label=f'{scale} agents')
        
        ax3.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Average Similarity', fontsize=12, fontweight='bold')
        ax3.set_title('Similarity Evolution by Scale', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = output_dir / 'scale_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {save_path}")
    
    def generate_report(self) -> str:
        """Generate text comparison report"""
        
        report = []
        report.append("="*80)
        report.append("MULTI-SCALE SIMULATION COMPARISON")
        report.append("="*80)
        report.append("")
        
        for scale in sorted(self.results.keys()):
            data = self.results[scale]
            
            report.append(f"\n{'─'*80}")
            report.append(f"{scale} AGENTS")
            report.append(f"{'─'*80}")
            
            # Coalition info
            if data['coalitions']:
                first_round = min(data['coalitions'].keys())
                coalitions = data['coalitions'][first_round]
                
                report.append(f"\nCoalition Formation (Round {first_round}):")
                report.append(f"  Number of coalitions: {len(coalitions)}")
                report.append(f"  Average size: {np.mean([len(c) for c in coalitions]):.1f}")
                report.append(f"  Size range: {min([len(c) for c in coalitions])} - {max([len(c) for c in coalitions])}")
            
            # Similarity info
            if data['similarity_matrices']:
                rounds = sorted(data['similarity_matrices'].keys())
                
                initial_matrix = data['similarity_matrices'][rounds[0]]
                final_matrix = data['similarity_matrices'][rounds[-1]]
                
                initial_avg = np.mean(initial_matrix[np.triu_indices_from(initial_matrix, k=1)])
                final_avg = np.mean(final_matrix[np.triu_indices_from(final_matrix, k=1)])
                
                report.append(f"\nSimilarity Evolution:")
                report.append(f"  Initial (Round {rounds[0]}): {initial_avg:.3f}")
                report.append(f"  Final (Round {rounds[-1]}): {final_avg:.3f}")
                report.append(f"  Change: {final_avg - initial_avg:+.3f}")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def save_report(self, report: str):
        """Save comparison report"""
        output_dir = Path("visualizations/scale_comparison")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / "scale_comparison_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Report saved: {report_path}")


def main():
    print("\n" + "📊 "*20)
    print("MULTI-SCALE COMPARISON ANALYSIS")
    print("📊 "*20)
    
    comparator = ScaleComparator()
    comparator.load_all_data()
    
    if not comparator.results:
        print("\n❌ No simulation data found. Run simulations first!")
        return
    
    # Run analyses
    coalition_comparison = comparator.compare_coalition_formation()
    similarity_comparison = comparator.compare_similarity_evolution()
    
    # Generate visualizations
    comparator.generate_comparison_visualizations()
    
    # Generate report
    report = comparator.generate_report()
    print("\n" + report)
    
    # Save report
    comparator.save_report(report)
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()