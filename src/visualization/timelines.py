"""
Timeline Visualizer
Creates Gantt-style charts showing coalition evolution over time
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple


class TimelineVisualizer:
    """
    Creates timeline visualizations showing:
    - Which agents belong to which coalitions over time
    - Agent migrations between coalitions
    - Coalition stability patterns
    """

    def __init__(self, output_dir: str = "visualizations/timelines"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color palette (same as network graphs)
        self.coalition_colors = [
            '#FF6B6B',  # Red
            '#4ECDC4',  # Teal
            '#45B7D1',  # Blue
            '#FFA07A',  # Orange
            '#98D8C8',  # Mint
            '#F7DC6F',  # Yellow
            '#BB8FCE',  # Purple
            '#85C1E2',  # Sky blue
        ]

    def load_coalition_history(
        self,
        coalition_dir: str = "data/coalitions",
        similarity_dir: str = "data/similarity"
    ) -> Tuple[List[int], List[List[List[str]]], List[str]]:
        """
        Load complete coalition history across all rounds.
        
        Returns:
            Tuple of (rounds_with_coalitions, coalition_structures, all_agent_names)
        """
        coalition_path = Path(coalition_dir)
        
        # Load all coalition files
        coalition_data = {}
        
        for file_path in sorted(coalition_path.glob("coalitions_round_*.json")):
            with open(file_path, 'r') as f:
                data = json.load(f)
                coalition_data[data['round']] = data['coalitions']
        
        # Get agent names from similarity matrix (authoritative source)
        sim_path = Path(similarity_dir)
        first_sim_file = sorted(sim_path.glob("matrix_round_*.json"))[0]
        
        with open(first_sim_file, 'r') as f:
            sim_data = json.load(f)
            all_agent_names = sim_data['agent_names']
        
        rounds = sorted(coalition_data.keys())
        structures = [coalition_data[r] for r in rounds]
        
        print(f"📅 Loaded coalition history for rounds: {rounds}")
        
        return rounds, structures, all_agent_names

    def create_coalition_timeline(self) -> Path:
        """
        Create a Gantt-style timeline showing coalition membership over time.
        
        Each row = one agent
        Each colored block = coalition membership in that round
        """
        print("\n📅 Creating coalition timeline...")
        
        rounds, structures, agent_names = self.load_coalition_history()
        
        if len(rounds) < 2:
            print("   ⚠️  Need at least 2 coalition formations for timeline")
            return None
        
        # Build agent membership timeline
        # timeline[agent_name][round] = coalition_id
        timeline = {name: {} for name in agent_names}
        
        for round_idx, round_num in enumerate(rounds):
            coalitions = structures[round_idx]
            
            for coalition_id, members in enumerate(coalitions):
                for agent in members:
                    # Only add if agent exists in our agent_names list
                    if agent in timeline:
                        timeline[agent][round_num] = coalition_id
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Y-axis: agents (one row per agent)
        y_positions = {name: i for i, name in enumerate(agent_names)}
        
        # Draw timeline bars
        bar_height = 0.8
        
        for agent_name in agent_names:
            y_pos = y_positions[agent_name]
            
            for round_num in rounds:
                if round_num in timeline[agent_name]:
                    coalition_id = timeline[agent_name][round_num]
                    color = self.coalition_colors[coalition_id % len(self.coalition_colors)]
                    
                    # Draw a rectangle for this agent's coalition membership
                    # X = round number, width = 1 round
                    rect = mpatches.Rectangle(
                        (round_num - 0.4, y_pos - bar_height/2),
                        width=0.8,
                        height=bar_height,
                        facecolor=color,
                        edgecolor='black',
                        linewidth=1.5,
                        alpha=0.9
                    )
                    ax.add_patch(rect)
        
        # Formatting
        ax.set_xlim(min(rounds) - 0.5, max(rounds) + 0.5)
        ax.set_ylim(-0.5, len(agent_names) - 0.5)
        
        ax.set_xlabel('Round Number', fontsize=14, fontweight='bold')
        ax.set_ylabel('Agents', fontsize=14, fontweight='bold')
        ax.set_title('Coalition Membership Timeline', fontsize=16, fontweight='bold', pad=20)
        
        # Set y-axis labels
        ax.set_yticks(range(len(agent_names)))
        ax.set_yticklabels(agent_names, fontsize=10)
        
        # Set x-axis ticks
        ax.set_xticks(rounds)
        ax.set_xticklabels([f'R{r}' for r in rounds])
        
        # Grid
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Legend
        max_coalitions = max(len(s) for s in structures)
        legend_elements = [
            mpatches.Patch(
                facecolor=self.coalition_colors[i % len(self.coalition_colors)],
                edgecolor='black',
                label=f'Coalition {i+1}'
            )
            for i in range(max_coalitions)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / 'coalition_timeline.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {save_path.name}")
        return save_path

    def create_migration_sankey(self) -> Path:
        """
        Create a Sankey-style diagram showing agent migrations between coalitions.
        
        Shows flow of agents from one coalition to another between clustering events.
        """
        print("\n🔀 Creating migration flow diagram...")
        
        rounds, structures, agent_names = self.load_coalition_history()
        
        if len(rounds) < 2:
            print("   ⚠️  Need at least 2 coalition formations")
            return None
        
        # For simplicity, show migration between first and last clustering
        round_before = rounds[0]
        round_after = rounds[-1]
        
        coalitions_before = structures[0]
        coalitions_after = structures[-1]
        
        # Build migration matrix
        # migration[from_coalition][to_coalition] = count
        n_before = len(coalitions_before)
        n_after = len(coalitions_after)
        
        migration_matrix = np.zeros((n_before, n_after), dtype=int)
        
        # Map agents to coalitions (only for agents in agent_names)
        agent_to_before = {}
        for i, members in enumerate(coalitions_before):
            for agent in members:
                if agent in agent_names:
                    agent_to_before[agent] = i
        
        agent_to_after = {}
        for i, members in enumerate(coalitions_after):
            for agent in members:
                if agent in agent_names:
                    agent_to_after[agent] = i
        
        # Count migrations
        for agent in agent_names:
            before_id = agent_to_before.get(agent, 0)
            after_id = agent_to_after.get(agent, 0)
            migration_matrix[before_id][after_id] += 1
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Simple bar chart representation
        x = np.arange(n_after)
        width = 0.8 / max(n_before, 1)
        
        for i in range(n_before):
            counts = migration_matrix[i]
            offset = (i - n_before/2) * width
            
            color = self.coalition_colors[i % len(self.coalition_colors)]
            ax.bar(
                x + offset,
                counts,
                width,
                label=f'From Coalition {i+1}',
                color=color,
                alpha=0.8,
                edgecolor='black'
            )
        
        ax.set_xlabel(f'Coalition at Round {round_after}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Agents', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Agent Migration: Round {round_before} → Round {round_after}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'Coalition {i+1}' for i in range(n_after)])
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / f'migration_flow_r{round_before}_to_r{round_after}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {save_path.name}")
        return save_path

    def create_stability_chart(self) -> Path:
        """
        Create a line chart showing coalition stability metrics over time.
        
        Shows:
        - Number of coalitions per round
        - Average coalition size
        - Churn rate between rounds
        """
        print("\n📈 Creating stability chart...")
        
        rounds, structures, agent_names = self.load_coalition_history()
        
        if len(rounds) < 2:
            print("   ⚠️  Need at least 2 coalition formations")
            return None
        
        # Calculate metrics per round
        num_coalitions = [len(s) for s in structures]
        avg_coalition_size = [
            np.mean([len(c) for c in s]) for s in structures
        ]
        
        # Calculate churn rates between consecutive rounds
        churn_rates = []
        churn_rounds = []
        
        for i in range(len(rounds) - 1):
            before = structures[i]
            after = structures[i + 1]
            
            # Build mappings (only for valid agents)
            map_before = {}
            for cid, members in enumerate(before):
                for agent in members:
                    if agent in agent_names:
                        map_before[agent] = cid
            
            map_after = {}
            for cid, members in enumerate(after):
                for agent in members:
                    if agent in agent_names:
                        map_after[agent] = cid
            
            # Count migrations
            migrations = sum(
                1 for agent in agent_names
                if map_before.get(agent, -1) != map_after.get(agent, -1)
            )
            
            churn_rate = migrations / len(agent_names) if agent_names else 0
            churn_rates.append(churn_rate * 100)  # Percentage
            churn_rounds.append(rounds[i + 1])
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot 1: Number of coalitions
        ax1.plot(rounds, num_coalitions, marker='o', linewidth=2, 
                 markersize=8, color='#FF6B6B')
        ax1.set_ylabel('Number of\nCoalitions', fontsize=11, fontweight='bold')
        ax1.set_title('Coalition Stability Metrics Over Time', 
                      fontsize=14, fontweight='bold', pad=15)
        ax1.grid(alpha=0.3)
        ax1.set_ylim(0, max(num_coalitions) + 1)
        
        # Plot 2: Average coalition size
        ax2.plot(rounds, avg_coalition_size, marker='s', linewidth=2,
                 markersize=8, color='#4ECDC4')
        ax2.set_ylabel('Avg Coalition\nSize', fontsize=11, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Plot 3: Churn rate
        if churn_rates:
            ax3.bar(churn_rounds, churn_rates, color='#FFA07A', 
                   edgecolor='black', alpha=0.8)
            ax3.set_ylabel('Churn Rate\n(%)', fontsize=11, fontweight='bold')
            ax3.grid(alpha=0.3, axis='y')
            ax3.axhline(y=20, color='red', linestyle='--', alpha=0.5, 
                       label='20% threshold')
            ax3.legend(loc='upper right', fontsize=9)
        
        ax3.set_xlabel('Round Number', fontsize=12, fontweight='bold')
        ax3.set_xticks(rounds)
        ax3.set_xticklabels([f'R{r}' for r in rounds])
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / 'stability_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {save_path.name}")
        return save_path