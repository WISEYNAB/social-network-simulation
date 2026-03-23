"""
Research Reporter
Generates human-readable reports from metrics
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class ResearchReporter:
    """
    Generates markdown and text reports from simulation metrics.
    """

    def __init__(self, output_dir: str = "visualizations/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(self, stats: Dict[str, Any]) -> Path:
        """
        Generate a markdown report with all statistics.
        
        Args:
            stats: Statistics dictionary from MetricsCalculator
        
        Returns:
            Path to the generated report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"research_report_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Social Network Simulation - Research Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # Overview
            f.write("## Overview\n\n")
            f.write(f"- **Total Rounds:** {stats.get('total_rounds', 0)}\n")
            f.write(f"- **Total Agents:** {stats.get('total_agents', 0)}\n")
            f.write(f"- **Coalition Formation Events:** {stats.get('coalition_events', 0)}\n\n")
            
            if 'agent_names' in stats:
                f.write("### Agent Roster\n\n")
                for i, name in enumerate(stats['agent_names'], 1):
                    f.write(f"{i}. {name}\n")
                f.write("\n")
            
            # Round-by-round metrics
            f.write("## Round-by-Round Analysis\n\n")
            
            for round_num in sorted(stats.get('per_round_metrics', {}).keys()):
                round_stats = stats['per_round_metrics'][round_num]
                
                f.write(f"### Round {round_num}\n\n")
                f.write(f"**Similarity Metrics:**\n")
                f.write(f"- Average similarity: {round_stats['avg_similarity']:.3f}\n")
                f.write(f"- Maximum similarity: {round_stats['max_similarity']:.3f}\n")
                f.write(f"- Minimum similarity: {round_stats['min_similarity']:.3f}\n\n")
                
                if 'num_coalitions' in round_stats:
                    f.write(f"**Coalition Structure:**\n")
                    f.write(f"- Number of coalitions: {round_stats['num_coalitions']}\n")
                    f.write(f"- Coalition sizes: {round_stats['coalition_sizes']}\n")
                    f.write(f"- Average cohesion: {round_stats['avg_cohesion']:.3f}\n")
                    f.write(f"- Modularity Q: {round_stats.get('modularity', 0):.3f}\n\n")
            
            # Churn analysis
            if 'churn_analysis' in stats:
                f.write("## Coalition Stability Analysis\n\n")
                f.write(f"**Average Churn Rate:** {stats.get('avg_churn_rate', 0)*100:.1f}%\n\n")
                
                for churn_event in stats['churn_analysis']:
                    f.write(f"### Round {churn_event['from_round']} → Round {churn_event['to_round']}\n\n")
                    f.write(f"- Churn rate: {churn_event['churn_rate']*100:.1f}%\n")
                    f.write(f"- Agents migrated: {churn_event['num_migrations']}\n\n")
                    
                    if churn_event['migrations']:
                        f.write("**Migrations:**\n\n")
                        for migration in churn_event['migrations']:
                            f.write(f"- {migration['agent']}: Coalition {migration['from_coalition']+1} → Coalition {migration['to_coalition']+1}\n")
                        f.write("\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            
            if stats.get('avg_churn_rate', 1.0) < 0.2:
                f.write("✅ **High Coalition Stability** - Churn rate below 20%, coalitions are stable.\n\n")
            elif stats.get('avg_churn_rate', 0) < 0.5:
                f.write("⚠️ **Moderate Coalition Stability** - Some agent migration between coalitions.\n\n")
            else:
                f.write("❌ **Low Coalition Stability** - High churn rate, coalitions are unstable.\n\n")
            
            # Check if modularity is improving
            per_round = stats.get('per_round_metrics', {})
            if len(per_round) >= 2:
                rounds_with_modularity = [r for r in per_round.values() if 'modularity' in r]
                if len(rounds_with_modularity) >= 2:
                    first_mod = rounds_with_modularity[0]['modularity']
                    last_mod = rounds_with_modularity[-1]['modularity']
                    
                    if last_mod > first_mod:
                        f.write(f"📈 **Improving Community Structure** - Modularity increased from {first_mod:.3f} to {last_mod:.3f}\n\n")
                    else:
                        f.write(f"📉 **Decreasing Community Structure** - Modularity decreased from {first_mod:.3f} to {last_mod:.3f}\n\n")
            
            f.write("---\n\n")
            f.write("*End of Report*\n")
        
        print(f"📄 Report saved to: {report_path}")
        return report_path

    def save_json_metrics(self, stats: Dict[str, Any]) -> Path:
        """
        Save raw metrics as JSON for further processing.
        
        Args:
            stats: Statistics dictionary
        
        Returns:
            Path to saved JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"metrics_{timestamp}.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"💾 Metrics saved to: {json_path}")
        return json_path