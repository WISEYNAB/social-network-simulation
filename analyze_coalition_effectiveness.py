"""
Coalition Effectiveness Analysis
Compares coalition-based interactions vs non-coalition interactions

Analyzes:
- Within-coalition vs between-coalition similarity
- Pre-coalition vs post-coalition metrics
- Engagement, agreement, and interaction quality
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analysis.metrics import MetricsCalculator


class CoalitionEffectivenessAnalyzer:
    """
    Analyzes the effectiveness of coalition formation by comparing:
    - Within-coalition interactions vs between-coalition interactions
    - Pre-coalition period vs post-coalition period
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.similarity_dir = self.data_dir / "similarity"
        self.coalition_dir = self.data_dir / "coalitions"
        self.conversation_dir = self.data_dir / "conversations"
        
        # Load data
        self.similarity_matrices = self._load_similarity_matrices()
        self.coalitions = self._load_coalitions()
        self.conversation_data = self._load_conversation_data()
        self.agent_names = self._get_agent_names()
    
    def _load_similarity_matrices(self) -> Dict[int, np.ndarray]:
        """Load all similarity matrices"""
        matrices = {}
        for file_path in sorted(self.similarity_dir.glob("matrix_round_*.json")):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                matrices[data['round']] = np.array(data['matrix'])
        return matrices
    
    def _load_coalitions(self) -> Dict[int, List[List[str]]]:
        """Load coalition formations"""
        coalitions = {}
        for file_path in sorted(self.coalition_dir.glob("coalitions_round_*.json")):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                coalitions[data['round']] = data['coalitions']
        return coalitions
    
    def _load_conversation_data(self) -> Dict:
        """Load conversation data from most recent session"""
        # Find most recent session
        session_dirs = sorted([d for d in self.conversation_dir.iterdir() if d.is_dir()])
        if not session_dirs:
            return None
        
        latest_session = session_dirs[-1]
        session_file = latest_session / "session.json"
        
        if session_file.exists():
            with open(session_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _get_agent_names(self) -> List[str]:
        """Get agent names from similarity matrix"""
        first_file = sorted(self.similarity_dir.glob("matrix_round_*.json"))[0]
        with open(first_file, 'r', encoding='utf-8') as f:
            return json.load(f)['agent_names']
    
    def analyze_similarity_comparison(self) -> Dict[str, Any]:
        """
        Compare similarity scores:
        - Within coalitions vs between coalitions
        - Pre-coalition vs post-coalition
        """
        print("\n📊 Analyzing Similarity Patterns...")
        
        results = {
            'pre_coalition': {},
            'post_coalition': {},
            'within_vs_between': {}
        }
        
        # Determine coalition formation rounds
        coalition_rounds = sorted(self.coalitions.keys())
        if not coalition_rounds:
            return results
        
        first_coalition_round = coalition_rounds[0]
        
        # Pre-coalition analysis (rounds before first coalition)
        pre_rounds = [r for r in self.similarity_matrices.keys() if r < first_coalition_round]
        if pre_rounds:
            pre_similarities = []
            for round_num in pre_rounds:
                matrix = self.similarity_matrices[round_num]
                # Get upper triangle (exclude diagonal)
                upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
                pre_similarities.extend(upper_tri)
            
            results['pre_coalition'] = {
                'avg_similarity': float(np.mean(pre_similarities)),
                'std_similarity': float(np.std(pre_similarities)),
                'min_similarity': float(np.min(pre_similarities)),
                'max_similarity': float(np.max(pre_similarities)),
                'rounds': pre_rounds
            }
        
        # Post-coalition analysis (rounds after first coalition)
        post_rounds = [r for r in self.similarity_matrices.keys() if r >= first_coalition_round]
        
        for round_num in post_rounds:
            if round_num not in self.coalitions:
                continue
            
            matrix = self.similarity_matrices[round_num]
            coalitions = self.coalitions[round_num]
            
            # Build agent->coalition mapping
            agent_to_coalition = {}
            for coalition_id, members in enumerate(coalitions):
                for agent in members:
                    if agent in self.agent_names:
                        agent_index = self.agent_names.index(agent)
                        agent_to_coalition[agent_index] = coalition_id
            
            # Separate within-coalition and between-coalition similarities
            within_coalition = []
            between_coalition = []
            
            for i in range(len(self.agent_names)):
                for j in range(i + 1, len(self.agent_names)):
                    similarity = matrix[i][j]
                    
                    coalition_i = agent_to_coalition.get(i, -1)
                    coalition_j = agent_to_coalition.get(j, -1)
                    
                    if coalition_i >= 0 and coalition_i == coalition_j:
                        # Same coalition
                        within_coalition.append(similarity)
                    elif coalition_i >= 0 and coalition_j >= 0:
                        # Different coalitions
                        between_coalition.append(similarity)
            
            # Store results
            if f'round_{round_num}' not in results['within_vs_between']:
                results['within_vs_between'][f'round_{round_num}'] = {}
            
            if within_coalition:
                results['within_vs_between'][f'round_{round_num}']['within_coalition'] = {
                    'avg': float(np.mean(within_coalition)),
                    'std': float(np.std(within_coalition)),
                    'count': len(within_coalition)
                }
            
            if between_coalition:
                results['within_vs_between'][f'round_{round_num}']['between_coalition'] = {
                    'avg': float(np.mean(between_coalition)),
                    'std': float(np.std(between_coalition)),
                    'count': len(between_coalition)
                }
            
            # Calculate separation score
            if within_coalition and between_coalition:
                separation = np.mean(within_coalition) - np.mean(between_coalition)
                results['within_vs_between'][f'round_{round_num}']['separation_score'] = float(separation)
        
        # Overall post-coalition averages
        all_within = []
        all_between = []
        
        for round_data in results['within_vs_between'].values():
            if 'within_coalition' in round_data:
                # Reconstruct individual values (approximate)
                count = round_data['within_coalition']['count']
                avg = round_data['within_coalition']['avg']
                all_within.extend([avg] * count)
            
            if 'between_coalition' in round_data:
                count = round_data['between_coalition']['count']
                avg = round_data['between_coalition']['avg']
                all_between.extend([avg] * count)
        
        if all_within and all_between:
            results['post_coalition'] = {
                'within_coalition_avg': float(np.mean(all_within)),
                'between_coalition_avg': float(np.mean(all_between)),
                'separation_score': float(np.mean(all_within) - np.mean(all_between)),
                'improvement_pct': float((np.mean(all_within) - np.mean(all_between)) / np.mean(all_between) * 100)
            }
        
        return results
    
    def analyze_engagement_comparison(self) -> Dict[str, Any]:
        """
        Compare engagement metrics:
        - Pre-coalition vs post-coalition
        - Activity levels in different periods
        """
        print("\n📊 Analyzing Engagement Patterns...")
        
        if not self.conversation_data:
            return {}
        
        results = {
            'pre_coalition': {},
            'post_coalition': {},
            'per_round': {}
        }
        
        coalition_rounds = sorted(self.coalitions.keys())
        first_coalition_round = coalition_rounds[0] if coalition_rounds else 999
        
        # Analyze each round
        for round_data in self.conversation_data['rounds']:
            round_num = round_data['round_number']
            
            total_posts = len(round_data['posts'])
            total_comments = len(round_data['comments'])
            
            # Calculate engagement rate
            active_agents = set()
            for post in round_data['posts']:
                active_agents.add(post['author'])
            for comment in round_data['comments']:
                active_agents.add(comment['commenter'])
            
            engagement_rate = len(active_agents) / len(self.agent_names)
            
            # Agreement analysis
            agreements = 0
            disagreements = 0
            
            for comment in round_data['comments']:
                comment_text = comment['comment'].lower()
                
                if any(kw in comment_text for kw in ['agree', 'exactly', 'yes', 'right', 'love', 'great point']):
                    agreements += 1
                elif any(kw in comment_text for kw in ['disagree', 'but', 'however', 'actually', 'wrong']):
                    disagreements += 1
            
            total_opinions = agreements + disagreements
            agreement_rate = agreements / total_opinions if total_opinions > 0 else 0
            
            round_metrics = {
                'total_posts': total_posts,
                'total_comments': total_comments,
                'active_agents': len(active_agents),
                'engagement_rate': float(engagement_rate),
                'agreements': agreements,
                'disagreements': disagreements,
                'agreement_rate': float(agreement_rate)
            }
            
            results['per_round'][f'round_{round_num}'] = round_metrics
            
            # Categorize as pre or post coalition
            if round_num < first_coalition_round:
                period = 'pre_coalition'
            else:
                period = 'post_coalition'
            
            if period not in results or not isinstance(results[period], dict):
                results[period] = {
                    'total_posts': 0,
                    'total_comments': 0,
                    'total_active': 0,
                    'total_agreements': 0,
                    'total_disagreements': 0,
                    'rounds': []
                }
            
            results[period]['total_posts'] += total_posts
            results[period]['total_comments'] += total_comments
            results[period]['total_active'] += len(active_agents)
            results[period]['total_agreements'] += agreements
            results[period]['total_disagreements'] += disagreements
            results[period]['rounds'].append(round_num)
        
        # Calculate averages
        for period in ['pre_coalition', 'post_coalition']:
            if period in results and isinstance(results[period], dict) and 'rounds' in results[period]:
                num_rounds = len(results[period]['rounds'])
                if num_rounds > 0:
                    results[period]['avg_posts_per_round'] = results[period]['total_posts'] / num_rounds
                    results[period]['avg_comments_per_round'] = results[period]['total_comments'] / num_rounds
                    results[period]['avg_engagement_rate'] = results[period]['total_active'] / (num_rounds * len(self.agent_names))
                    
                    total_opinions = results[period]['total_agreements'] + results[period]['total_disagreements']
                    results[period]['avg_agreement_rate'] = results[period]['total_agreements'] / total_opinions if total_opinions > 0 else 0
        
        return results
    
    def generate_comparison_report(self, similarity_results: Dict, engagement_results: Dict) -> str:
        """Generate a text report comparing coalition vs non-coalition"""
        
        report = []
        report.append("="*80)
        report.append("COALITION EFFECTIVENESS ANALYSIS")
        report.append("="*80)
        report.append("")
        
        # Similarity Analysis
        report.append("📊 SIMILARITY ANALYSIS")
        report.append("-" * 80)
        report.append("")
        
        if 'pre_coalition' in similarity_results and similarity_results['pre_coalition']:
            pre = similarity_results['pre_coalition']
            report.append(f"PRE-COALITION (Rounds {pre['rounds']}):")
            report.append(f"  Average Similarity: {pre['avg_similarity']:.3f}")
            report.append(f"  Std Deviation:      {pre['std_similarity']:.3f}")
            report.append(f"  Range:              {pre['min_similarity']:.3f} - {pre['max_similarity']:.3f}")
            report.append("")
        
        if 'post_coalition' in similarity_results and similarity_results['post_coalition']:
            post = similarity_results['post_coalition']
            report.append("POST-COALITION:")
            report.append(f"  Within-Coalition Avg:  {post['within_coalition_avg']:.3f}")
            report.append(f"  Between-Coalition Avg: {post['between_coalition_avg']:.3f}")
            report.append(f"  Separation Score:      {post['separation_score']:.3f}")
            report.append(f"  Improvement:           {post['improvement_pct']:.1f}%")
            report.append("")
            
            if post['separation_score'] > 0.1:
                report.append("  ✅ STRONG COALITION EFFECT: Within-coalition similarity is significantly")
                report.append("     higher than between-coalition similarity.")
            elif post['separation_score'] > 0.05:
                report.append("  ⚠️  MODERATE COALITION EFFECT: Some clustering is visible.")
            else:
                report.append("  ❌ WEAK COALITION EFFECT: Coalitions do not show strong cohesion.")
            report.append("")
        
        # Round-by-round breakdown
        if 'within_vs_between' in similarity_results:
            report.append("ROUND-BY-ROUND COMPARISON:")
            report.append("")
            for round_key in sorted(similarity_results['within_vs_between'].keys()):
                round_data = similarity_results['within_vs_between'][round_key]
                round_num = round_key.split('_')[1]
                
                report.append(f"  Round {round_num}:")
                if 'within_coalition' in round_data:
                    report.append(f"    Within-Coalition:  {round_data['within_coalition']['avg']:.3f} (n={round_data['within_coalition']['count']})")
                if 'between_coalition' in round_data:
                    report.append(f"    Between-Coalition: {round_data['between_coalition']['avg']:.3f} (n={round_data['between_coalition']['count']})")
                if 'separation_score' in round_data:
                    report.append(f"    Separation:        {round_data['separation_score']:.3f}")
                report.append("")
        
        # Engagement Analysis
        report.append("="*80)
        report.append("📈 ENGAGEMENT ANALYSIS")
        report.append("-" * 80)
        report.append("")
        
        if 'pre_coalition' in engagement_results and 'avg_posts_per_round' in engagement_results['pre_coalition']:
            pre = engagement_results['pre_coalition']
            report.append(f"PRE-COALITION (Rounds {pre['rounds']}):")
            report.append(f"  Avg Posts/Round:     {pre['avg_posts_per_round']:.1f}")
            report.append(f"  Avg Comments/Round:  {pre['avg_comments_per_round']:.1f}")
            report.append(f"  Engagement Rate:     {pre['avg_engagement_rate']*100:.1f}%")
            report.append(f"  Agreement Rate:      {pre['avg_agreement_rate']*100:.1f}%")
            report.append("")
        
        if 'post_coalition' in engagement_results and 'avg_posts_per_round' in engagement_results['post_coalition']:
            post = engagement_results['post_coalition']
            report.append(f"POST-COALITION (Rounds {post['rounds']}):")
            report.append(f"  Avg Posts/Round:     {post['avg_posts_per_round']:.1f}")
            report.append(f"  Avg Comments/Round:  {post['avg_comments_per_round']:.1f}")
            report.append(f"  Engagement Rate:     {post['avg_engagement_rate']*100:.1f}%")
            report.append(f"  Agreement Rate:      {post['avg_agreement_rate']*100:.1f}%")
            report.append("")
            
            # Comparison
            if 'pre_coalition' in engagement_results and 'avg_posts_per_round' in engagement_results['pre_coalition']:
                pre = engagement_results['pre_coalition']
                
                post_change = ((post['avg_posts_per_round'] - pre['avg_posts_per_round']) / pre['avg_posts_per_round'] * 100)
                comment_change = ((post['avg_comments_per_round'] - pre['avg_comments_per_round']) / pre['avg_comments_per_round'] * 100)
                engagement_change = ((post['avg_engagement_rate'] - pre['avg_engagement_rate']) / pre['avg_engagement_rate'] * 100)
                
                report.append("CHANGE AFTER COALITION FORMATION:")
                report.append(f"  Posts:      {post_change:+.1f}%")
                report.append(f"  Comments:   {comment_change:+.1f}%")
                report.append(f"  Engagement: {engagement_change:+.1f}%")
                report.append("")
        
        # Key Findings
        report.append("="*80)
        report.append("🎯 KEY FINDINGS")
        report.append("-" * 80)
        report.append("")
        
        findings = []
        
        if 'post_coalition' in similarity_results and similarity_results['post_coalition']:
            sep_score = similarity_results['post_coalition']['separation_score']
            if sep_score > 0.1:
                findings.append("✅ Coalition formation successfully created distinct communities")
                findings.append(f"   with {sep_score:.1%} higher within-group similarity")
        
        if ('pre_coalition' in engagement_results and 
            'post_coalition' in engagement_results and 
            'avg_engagement_rate' in engagement_results['pre_coalition'] and
            'avg_engagement_rate' in engagement_results['post_coalition']):
            
            pre_eng = engagement_results['pre_coalition']['avg_engagement_rate']
            post_eng = engagement_results['post_coalition']['avg_engagement_rate']
            
            if post_eng > pre_eng:
                findings.append(f"✅ Engagement increased by {((post_eng - pre_eng) / pre_eng * 100):.1f}% after coalition formation")
            else:
                findings.append(f"⚠️  Engagement decreased by {((pre_eng - post_eng) / pre_eng * 100):.1f}% after coalition formation")
        
        for finding in findings:
            report.append(finding)
        
        if not findings:
            report.append("⚠️  Insufficient data for conclusive findings")
        
        report.append("")
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_results(self, similarity_results: Dict, engagement_results: Dict, report: str):
        """Save analysis results"""
        output_dir = Path("visualizations/coalition_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON with UTF-8 encoding
        json_path = output_dir / "coalition_effectiveness.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'similarity_analysis': similarity_results,
                'engagement_analysis': engagement_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved JSON results: {json_path}")
        
        # Save text report with UTF-8 encoding (FIX FOR EMOJI ERROR)
        report_path = output_dir / "coalition_effectiveness_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Saved text report: {report_path}")
        
        return json_path, report_path


def main():
    """Main analysis pipeline"""
    
    print("\n" + "🔬 "*20)
    print("COALITION EFFECTIVENESS ANALYSIS")
    print("🔬 "*20)
    
    analyzer = CoalitionEffectivenessAnalyzer(data_dir="data")
    
    # Run analyses
    similarity_results = analyzer.analyze_similarity_comparison()
    engagement_results = analyzer.analyze_engagement_comparison()
    
    # Generate report
    report = analyzer.generate_comparison_report(similarity_results, engagement_results)
    
    # Print report
    print("\n" + report)
    
    # Save results
    json_path, report_path = analyzer.save_results(similarity_results, engagement_results, report)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📁 Results saved to: visualizations/coalition_analysis/")
    print(f"   - {json_path.name}")
    print(f"   - {report_path.name}")
    print("\n")


if __name__ == "__main__":
    main()