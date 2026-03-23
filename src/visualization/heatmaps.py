"""
Heatmap Visualizer
Creates similarity matrix heatmaps (static and animated)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from typing import Dict, List


class HeatmapVisualizer:
    """
    Creates heatmap visualizations of similarity matrices.
    - Static heatmaps for each round
    - Animated GIF showing evolution
    """

    def __init__(self, output_dir: str = "visualizations/heatmaps"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'

    def load_similarity_matrices(self, data_dir: str = "data/similarity") -> Dict:
        """
        Load all similarity matrices from JSON files.
        
        Returns:
            Dictionary with round numbers as keys
        """
        similarity_dir = Path(data_dir)
        matrices = {}
        
        for file_path in sorted(similarity_dir.glob("matrix_round_*.json")):
            with open(file_path, 'r') as f:
                data = json.load(f)
                matrices[data['round']] = {
                    'matrix': np.array(data['matrix']),
                    'agent_names': data['agent_names']
                }
        
        print(f"📊 Loaded {len(matrices)} matrices for heatmap generation")
        return matrices

    def create_heatmap(
        self,
        matrix: np.ndarray,
        agent_names: List[str],
        round_number: int,
        title: str = None,
        save_path: Path = None
    ) -> Path:
        """
        Create a single heatmap for a similarity matrix.
        
        Args:
            matrix: NxN similarity matrix
            agent_names: List of agent names (for labels)
            round_number: Round number (for title)
            title: Custom title (optional)
            save_path: Where to save (optional)
        
        Returns:
            Path to saved image
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        # Use a diverging colormap centered at 0.5
        sns.heatmap(
            matrix,
            annot=True,          # Show numbers in cells
            fmt='.2f',           # Format: 2 decimal places
            cmap='RdYlGn',       # Red-Yellow-Green colormap
            center=0.5,          # Center at medium similarity
            vmin=0,              # Min value
            vmax=1,              # Max value
            square=True,         # Square cells
            linewidths=0.5,      # Grid lines
            cbar_kws={'label': 'Similarity Score'},
            xticklabels=[name[:15] for name in agent_names],  # Truncate long names
            yticklabels=[name[:15] for name in agent_names],
            ax=ax
        )
        
        # Title
        if title is None:
            title = f'Agent Similarity Matrix - Round {round_number}'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Labels
        ax.set_xlabel('Agents', fontsize=12)
        ax.set_ylabel('Agents', fontsize=12)
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Tight layout
        plt.tight_layout()
        
        # Save
        if save_path is None:
            save_path = self.output_dir / f'heatmap_round_{round_number:02d}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

    def create_all_heatmaps(self, data_dir: str = "data/similarity") -> List[Path]:
        """
        Create heatmaps for all rounds.
        
        Returns:
            List of paths to saved images
        """
        print("\n🎨 Creating similarity heatmaps...")
        
        matrices_data = self.load_similarity_matrices(data_dir)
        saved_paths = []
        
        for round_num in sorted(matrices_data.keys()):
            data = matrices_data[round_num]
            matrix = data['matrix']
            agent_names = data['agent_names']
            
            path = self.create_heatmap(
                matrix=matrix,
                agent_names=agent_names,
                round_number=round_num
            )
            
            saved_paths.append(path)
            print(f"   ✅ Round {round_num}: {path.name}")
        
        return saved_paths

    def create_animated_heatmap(
        self,
        data_dir: str = "data/similarity",
        duration: int = 1000
    ) -> Path:
        """
        Create an animated GIF showing similarity evolution over rounds.
        
        Args:
            data_dir: Directory with similarity matrices
            duration: Milliseconds per frame
        
        Returns:
            Path to saved GIF
        """
        print("\n🎬 Creating animated heatmap GIF...")
        
        # First, create all individual heatmaps
        heatmap_paths = self.create_all_heatmaps(data_dir)
        
        if len(heatmap_paths) < 2:
            print("   ⚠️  Need at least 2 rounds to create animation")
            return None
        
        # Load images
        images = [Image.open(path) for path in heatmap_paths]
        
        # Save as GIF
        gif_path = self.output_dir / 'similarity_evolution.gif'
        
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0  # Loop forever
        )
        
        print(f"   ✅ Animated GIF saved: {gif_path}")
        print(f"      Frames: {len(images)}, Duration: {duration}ms per frame")
        
        return gif_path

    def create_difference_heatmap(
        self,
        round_before: int,
        round_after: int,
        data_dir: str = "data/similarity"
    ) -> Path:
        """
        Create a heatmap showing the CHANGE in similarity between two rounds.
        
        Useful for visualizing what changed after coalition formation.
        
        Args:
            round_before: Earlier round number
            round_after: Later round number
            data_dir: Directory with similarity matrices
        
        Returns:
            Path to saved image
        """
        print(f"\n📊 Creating difference heatmap (Round {round_before} → {round_after})...")
        
        matrices_data = self.load_similarity_matrices(data_dir)
        
        if round_before not in matrices_data or round_after not in matrices_data:
            print("   ❌ One or both rounds not found")
            return None
        
        matrix_before = matrices_data[round_before]['matrix']
        matrix_after = matrices_data[round_after]['matrix']
        agent_names = matrices_data[round_before]['agent_names']
        
        # Calculate difference
        difference = matrix_after - matrix_before
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap (diverging around 0)
        sns.heatmap(
            difference,
            annot=True,
            fmt='+.2f',          # Show + for positive changes
            cmap='RdBu_r',       # Red = decrease, Blue = increase
            center=0,            # Center at no change
            vmin=-0.5,
            vmax=0.5,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Similarity Change'},
            xticklabels=[name[:15] for name in agent_names],
            yticklabels=[name[:15] for name in agent_names],
            ax=ax
        )
        
        ax.set_title(
            f'Similarity Change: Round {round_before} → Round {round_after}',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        
        ax.set_xlabel('Agents', fontsize=12)
        ax.set_ylabel('Agents', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / f'difference_round_{round_before}_to_{round_after}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {save_path.name}")
        return save_path