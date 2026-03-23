"""
Network Visualizer
Creates agent network graphs showing relationships and coalitions
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional


class NetworkVisualizer:
    """
    Creates network graphs where:
    - Nodes = Agents
    - Edges = Similarity (weighted)
    - Colors = Coalition membership
    """

    def __init__(self, output_dir: str = "visualizations/networks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color palette for coalitions
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

    def load_data(
        self,
        round_number: int,
        similarity_dir: str = "data/similarity",
        coalition_dir: str = "data/coalitions"
    ) -> Dict:
        """
        Load similarity matrix and coalition data for a specific round.
        
        Returns:
            Dictionary with matrix, agent_names, and coalitions (if available)
        """
        # Load similarity matrix
        sim_path = Path(similarity_dir) / f"matrix_round_{round_number:02d}.json"
        
        if not sim_path.exists():
            return None
        
        with open(sim_path, 'r') as f:
            sim_data = json.load(f)
        
        data = {
            'matrix': np.array(sim_data['matrix']),
            'agent_names': sim_data['agent_names'],
            'round': round_number,
            'coalitions': None
        }
        
        # Try to load coalitions
        coal_path = Path(coalition_dir) / f"coalitions_round_{round_number:02d}.json"
        
        if coal_path.exists():
            with open(coal_path, 'r') as f:
                coal_data = json.load(f)
                data['coalitions'] = coal_data['coalitions']
        
        return data

    def create_network_graph(
        self,
        round_number: int,
        similarity_threshold: float = 0.4,
        show_edge_labels: bool = False,
        layout: str = 'spring'
    ) -> Path:
        """
        Create a network graph for a specific round.
        
        Args:
            round_number: Which round to visualize
            similarity_threshold: Only show edges above this similarity
            show_edge_labels: Whether to show similarity scores on edges
            layout: 'spring', 'circular', or 'kamada_kawai'
        
        Returns:
            Path to saved image
        """
        print(f"\n🕸️  Creating network graph for round {round_number}...")
        
        data = self.load_data(round_number)
        
        if data is None:
            print(f"   ❌ No data found for round {round_number}")
            return None
        
        matrix = data['matrix']
        agent_names = data['agent_names']
        coalitions = data['coalitions']
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for i, name in enumerate(agent_names):
            G.add_node(i, name=name)
        
        # Add edges (only if similarity > threshold)
        edge_weights = []
        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                similarity = matrix[i][j]
                if similarity > similarity_threshold:
                    G.add_edge(i, j, weight=similarity)
                    edge_weights.append(similarity)
        
        # Determine node colors based on coalitions
        if coalitions:
            # Build agent -> coalition mapping
            agent_to_coalition = {}
            for coalition_id, members in enumerate(coalitions):
                for agent in members:
                    agent_to_coalition[agent] = coalition_id
            
            # Color nodes by coalition
            node_colors = []
            for name in agent_names:
                coalition_id = agent_to_coalition.get(name, 0)
                color = self.coalition_colors[coalition_id % len(self.coalition_colors)]
                node_colors.append(color)
        else:
            # No coalitions - use neutral color
            node_colors = ['#95a5a6'] * len(agent_names)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=3000,
            alpha=0.9,
            edgecolors='black',
            linewidths=2,
            ax=ax
        )
        
        # Draw edges (width proportional to similarity)
        if edge_weights:
            # Normalize edge widths
            max_weight = max(edge_weights)
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            widths = [5 * (w / max_weight) for w in weights]
            
            nx.draw_networkx_edges(
                G, pos,
                width=widths,
                alpha=0.3,
                edge_color='gray',
                ax=ax
            )
            
            # Optionally show edge labels
            if show_edge_labels:
                edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges}
                nx.draw_networkx_edge_labels(
                    G, pos,
                    edge_labels=edge_labels,
                    font_size=8,
                    ax=ax
                )
        
        # Draw labels
        labels = {i: name for i, name in enumerate(agent_names)}
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=10,
            font_weight='bold',
            ax=ax
        )
        
        # Title
        title = f'Agent Network - Round {round_number}'
        if coalitions:
            title += f' ({len(coalitions)} Coalitions)'
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        
        # Legend for coalitions
        if coalitions:
            legend_elements = []
            for i, coalition in enumerate(coalitions):
                color = self.coalition_colors[i % len(self.coalition_colors)]
                label = f"Coalition {i+1} ({len(coalition)} agents)"
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=color, markersize=10, label=label)
                )
            ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / f'network_round_{round_number:02d}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Saved: {save_path.name}")
        return save_path

    def create_all_network_graphs(
        self,
        similarity_threshold: float = 0.4
    ) -> List[Path]:
        """
        Create network graphs for all available rounds.
        
        Returns:
            List of paths to saved images
        """
        print("\n🕸️  Creating network graphs for all rounds...")
        
        # Find all available rounds
        similarity_dir = Path("data/similarity")
        available_rounds = []
        
        for file_path in similarity_dir.glob("matrix_round_*.json"):
            round_num = int(file_path.stem.split('_')[-1])
            available_rounds.append(round_num)
        
        available_rounds.sort()
        saved_paths = []
        
        for round_num in available_rounds:
            path = self.create_network_graph(
                round_number=round_num,
                similarity_threshold=similarity_threshold
            )
            if path:
                saved_paths.append(path)
        
        return saved_paths

    def create_comparison_network(
        self,
        round_before: int,
        round_after: int,
        similarity_threshold: float = 0.4
    ) -> Path:
        """
        Create side-by-side comparison of networks before/after clustering.
        
        Args:
            round_before: Round before clustering
            round_after: Round after clustering
            similarity_threshold: Edge threshold
        
        Returns:
            Path to saved comparison image
        """
        print(f"\n🔀 Creating network comparison (Round {round_before} vs {round_after})...")
        
        data_before = self.load_data(round_before)
        data_after = self.load_data(round_after)
        
        if not data_before or not data_after:
            print("   ❌ Missing data for comparison")
            return None
        
        # Create side-by-side figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        for ax, data, title in [(ax1, data_before, f'Round {round_before}'), 
                               (ax2, data_after, f'Round {round_after}')]:
            
            matrix = data['matrix']
            agent_names = data['agent_names']
            coalitions = data['coalitions']
            
            # Build graph
            G = nx.Graph()
            for i, name in enumerate(agent_names):
                G.add_node(i, name=name)
            
            for i in range(len(agent_names)):
                for j in range(i + 1, len(agent_names)):
                    if matrix[i][j] > similarity_threshold:
                        G.add_edge(i, j, weight=matrix[i][j])
            
            # Colors
            if coalitions:
                agent_to_coalition = {}
                for coalition_id, members in enumerate(coalitions):
                    for agent in members:
                        agent_to_coalition[agent] = coalition_id
                
                node_colors = [
                    self.coalition_colors[agent_to_coalition.get(name, 0) % len(self.coalition_colors)]
                    for name in agent_names
                ]
            else:
                node_colors = ['#95a5a6'] * len(agent_names)
            
            # Layout (use same layout for both for easier comparison)
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            
            # Draw
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                  node_size=2000, alpha=0.9, 
                                  edgecolors='black', linewidths=2, ax=ax)
            
            if G.edges():
                edges = G.edges()
                weights = [G[u][v]['weight'] for u, v in edges]
                max_weight = max(weights) if weights else 1
                widths = [5 * (w / max_weight) for w in weights]
                
                nx.draw_networkx_edges(G, pos, width=widths, 
                                      alpha=0.3, edge_color='gray', ax=ax)
            
            labels = {i: name for i, name in enumerate(agent_names)}
            nx.draw_networkx_labels(G, pos, labels=labels, 
                                   font_size=9, font_weight='bold', ax=ax)
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.axis('off')
        
        plt.suptitle(
            f'Network Comparison: Before & After Clustering',
            fontsize=20,
            fontweight='bold',
            y=0.98
        )
        
        plt.tight_layout()
        
        save_path = self.output_dir / f'comparison_round_{round_before}_vs_{round_after}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Saved: {save_path.name}")
        return save_path