"""
Dashboard Visualizer
Creates summary dashboards with agent statistics and metrics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List


class DashboardVisualizer:
    """
    Creates dashboard-style visualizations summarizing agent behavior.
    """

    def __init__(self, output_dir: str = "visualizations/dashboards"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        sns.set_style("whitegrid")

    def load_conversation_data(
        self,
        conversation_dir: str = "data/conversations"
    ) -> Dict:
        """
        Load conversation data from the most recent session.
        
        Returns:
            Session data dictionary
        """
        conv_path = Path(conversation_dir)
        
        # Find most recent session
        session_dirs = sorted([d for d in conv_path.iterdir() if d.is_dir()])
        
        if not session_dirs:
            return None
        
        latest_session = session_dirs[-1]
        session_file = latest_session / "session.json"
        
        if not session_file.exists():
            return None
        
        with open(session_file, 'r') as f:
            return json.load(f)

    def create_agent_activity_dashboard(self) -> Path:
        """
        Create a dashboard showing agent activity metrics.
        
        Includes:
        - Posts per agent
        - Comments per agent
        - Agreements vs disagreements
        - Total engagement
        """
        print("\n📊 Creating agent activity dashboard...")
        
        data = self.load_conversation_data()
        
        if not data:
            print("   ❌ No conversation data found")
            return None
        
        # Extract agent statistics
        agent_stats = {}
        
        for round_data in data['rounds']:
            for post in round_data['posts']:
                author = post['author']
                if author not in agent_stats:
                    agent_stats[author] = {
                        'posts': 0,
                        'comments': 0,
                        'agreements': 0,
                        'disagreements': 0
                    }
                agent_stats[author]['posts'] += 1
            
            for comment in round_data['comments']:
                commenter = comment['commenter']
                if commenter not in agent_stats:
                    agent_stats[commenter] = {
                        'posts': 0,
                        'comments': 0,
                        'agreements': 0,
                        'disagreements': 0
                    }
                agent_stats[commenter]['comments'] += 1
                
                # Simple keyword detection for agreements/disagreements
                comment_text = comment['comment'].lower()
                
                if any(kw in comment_text for kw in ['agree', 'exactly', 'yes', 'right', 'love']):
                    agent_stats[commenter]['agreements'] += 1
                elif any(kw in comment_text for kw in ['disagree', 'but', 'however', 'actually']):
                    agent_stats[commenter]['disagreements'] += 1
        
        # Sort agents by total activity
        agents = sorted(agent_stats.keys(), 
                       key=lambda a: agent_stats[a]['posts'] + agent_stats[a]['comments'],
                       reverse=True)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Subplot 1: Posts and Comments
        ax1 = fig.add_subplot(gs[0, :])
        
        x = np.arange(len(agents))
        width = 0.35
        
        posts = [agent_stats[a]['posts'] for a in agents]
        comments = [agent_stats[a]['comments'] for a in agents]
        
        ax1.bar(x - width/2, posts, width, label='Posts', color='#4ECDC4', edgecolor='black')
        ax1.bar(x + width/2, comments, width, label='Comments', color='#FF6B6B', edgecolor='black')
        
        ax1.set_xlabel('Agents', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Agent Activity: Posts and Comments', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([a[:15] for a in agents], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Subplot 2: Total Engagement
        ax2 = fig.add_subplot(gs[1, 0])
        
        total_engagement = [posts[i] + comments[i] for i in range(len(agents))]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(agents)))
        
        ax2.barh(agents, total_engagement, color=colors, edgecolor='black')
        ax2.set_xlabel('Total Activity (Posts + Comments)', fontsize=11, fontweight='bold')
        ax2.set_title('Total Engagement by Agent', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Subplot 3: Agreements vs Disagreements
        ax3 = fig.add_subplot(gs[1, 1])
        
        agreements = [agent_stats[a]['agreements'] for a in agents]
        disagreements = [agent_stats[a]['disagreements'] for a in agents]
        
        ax3.bar(x - width/2, agreements, width, label='Agreements', 
               color='#98D8C8', edgecolor='black')
        ax3.bar(x + width/2, disagreements, width, label='Disagreements',
               color='#FFA07A', edgecolor='black')
        
        ax3.set_xlabel('Agents', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax3.set_title('Agreements vs Disagreements', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([a[:15] for a in agents], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Agent Activity Dashboard', fontsize=16, fontweight='bold', y=0.98)
        
        # Save
        save_path = self.output_dir / 'agent_activity_dashboard.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {save_path.name}")
        return save_path

    def create_similarity_evolution_chart(
        self,
        similarity_dir: str = "data/similarity"
    ) -> Path:
        """
        Create a line chart showing how average similarity evolves over rounds.
        
        Shows:
        - Average pairwise similarity per round
        - Max and min similarity per round
        """
        print("\n📈 Creating similarity evolution chart...")
        
        sim_path = Path(similarity_dir)
        
        # Load all matrices
        matrices = {}
        for file_path in sorted(sim_path.glob("matrix_round_*.json")):
            with open(file_path, 'r') as f:
                data = json.load(f)
                matrices[data['round']] = np.array(data['matrix'])
        
        if not matrices:
            print("   ❌ No similarity matrices found")
            return None
        
        rounds = sorted(matrices.keys())
        
        # Calculate statistics per round
        avg_similarities = []
        max_similarities = []
        min_similarities = []
        
        for round_num in rounds:
            matrix = matrices[round_num]
            
            # Get upper triangle (exclude diagonal)
            upper_triangle = matrix[np.triu_indices_from(matrix, k=1)]
            
            avg_similarities.append(np.mean(upper_triangle))
            max_similarities.append(np.max(upper_triangle))
            min_similarities.append(np.min(upper_triangle))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(rounds, avg_similarities, marker='o', linewidth=2.5, 
               markersize=8, label='Average Similarity', color='#4ECDC4')
        ax.plot(rounds, max_similarities, marker='^', linewidth=2, 
               markersize=7, label='Maximum Similarity', color='#98D8C8', 
               linestyle='--', alpha=0.7)
        ax.plot(rounds, min_similarities, marker='v', linewidth=2,
               markersize=7, label='Minimum Similarity', color='#FFA07A',
               linestyle='--', alpha=0.7)
        
        # Fill between min and max
        ax.fill_between(rounds, min_similarities, max_similarities, 
                        alpha=0.2, color='gray')
        
        ax.set_xlabel('Round Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Similarity Score', fontsize=12, fontweight='bold')
        ax.set_title('Similarity Evolution Over Time', fontsize=14, fontweight='bold', pad=15)
        
        ax.set_xticks(rounds)
        ax.set_xticklabels([f'R{r}' for r in rounds])
        ax.set_ylim(0, 1)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / 'similarity_evolution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {save_path.name}")
        return save_path

    def create_summary_dashboard(
        self,
        stats: Dict
    ) -> Path:
        """
        Create an overall summary dashboard with key metrics.
        
        Args:
            stats: Statistics dictionary from MetricsCalculator
        
        Returns:
            Path to saved image
        """
        print("\n📊 Creating summary dashboard...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
        
        # Title
        fig.suptitle('Social Network Simulation - Summary Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # ── PANEL 1: Overview Stats ──
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        overview_text = f"""
        SIMULATION OVERVIEW
        
        Total Rounds: {stats.get('total_rounds', 0)}
        Total Agents: {stats.get('total_agents', 0)}
        Coalition Formation Events: {stats.get('coalition_events', 0)}
        Average Churn Rate: {stats.get('avg_churn_rate', 0)*100:.1f}%
        """
        
        ax1.text(0.1, 0.5, overview_text, fontsize=14, 
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # ── PANEL 2: Coalition Sizes ──
        if stats.get('per_round_metrics'):
            rounds_with_coalitions = [
                r for r, data in stats['per_round_metrics'].items()
                if 'num_coalitions' in data
            ]
            
            if rounds_with_coalitions:
                ax2 = fig.add_subplot(gs[1, 0])
                
                num_coalitions = [
                    stats['per_round_metrics'][r]['num_coalitions']
                    for r in rounds_with_coalitions
                ]
                
                ax2.bar(range(len(rounds_with_coalitions)), num_coalitions,
                       color='#4ECDC4', edgecolor='black')
                ax2.set_xlabel('Coalition Event', fontsize=10, fontweight='bold')
                ax2.set_ylabel('Number of Coalitions', fontsize=10, fontweight='bold')
                ax2.set_title('Coalition Count', fontsize=11, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)
                
                # ── PANEL 3: Average Cohesion ──
                ax3 = fig.add_subplot(gs[1, 1])
                
                cohesions = [
                    stats['per_round_metrics'][r]['avg_cohesion']
                    for r in rounds_with_coalitions
                ]
                
                ax3.plot(range(len(rounds_with_coalitions)), cohesions,
                        marker='o', linewidth=2, markersize=8, color='#FF6B6B')
                ax3.set_xlabel('Coalition Event', fontsize=10, fontweight='bold')
                ax3.set_ylabel('Average Cohesion', fontsize=10, fontweight='bold')
                ax3.set_title('Coalition Cohesion', fontsize=11, fontweight='bold')
                ax3.set_ylim(0, 1)
                ax3.grid(alpha=0.3)
                
                # ── PANEL 4: Modularity ──
                ax4 = fig.add_subplot(gs[1, 2])
                
                modularities = [
                    stats['per_round_metrics'][r].get('modularity', 0)
                    for r in rounds_with_coalitions
                ]
                
                ax4.bar(range(len(rounds_with_coalitions)), modularities,
                       color='#98D8C8', edgecolor='black')
                ax4.set_xlabel('Coalition Event', fontsize=10, fontweight='bold')
                ax4.set_ylabel('Modularity Q', fontsize=10, fontweight='bold')
                ax4.set_title('Community Structure Quality', fontsize=11, fontweight='bold')
                ax4.grid(axis='y', alpha=0.3)
        
        # ── PANEL 5: Churn Analysis ──
        if 'churn_analysis' in stats:
            ax5 = fig.add_subplot(gs[2, :])
            
            churn_events = stats['churn_analysis']
            churn_labels = [
                f"R{c['from_round']}→R{c['to_round']}"
                for c in churn_events
            ]
            churn_values = [c['churn_rate'] * 100 for c in churn_events]
            
            colors_churn = ['#98D8C8' if v < 20 else '#FFA07A' for v in churn_values]
            
            ax5.bar(range(len(churn_events)), churn_values, 
                   color=colors_churn, edgecolor='black')
            ax5.axhline(y=20, color='red', linestyle='--', alpha=0.5,
                       label='20% stability threshold')
            ax5.set_xlabel('Coalition Transition', fontsize=11, fontweight='bold')
            ax5.set_ylabel('Churn Rate (%)', fontsize=11, fontweight='bold')
            ax5.set_title('Coalition Stability (Churn Rate)', fontsize=12, fontweight='bold')
            ax5.set_xticks(range(len(churn_events)))
            ax5.set_xticklabels(churn_labels)
            ax5.legend()
            ax5.grid(axis='y', alpha=0.3)
        
        # Save
        save_path = self.output_dir / 'summary_dashboard.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {save_path.name}")
        return save_path