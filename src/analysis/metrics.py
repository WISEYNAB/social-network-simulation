"""
Metrics Calculator
Computes research-relevant statistics from simulation data
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple


class MetricsCalculator:
    """
    Calculates key metrics for coalition formation research:
    - Coalition stability (churn rate, persistence)
    - Cohesion (intra-group similarity)
    - Separation (inter-group distance)
    - Convergence (when similarity stabilizes)
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.similarity_dir = self.data_dir / "similarity"
        self.coalition_dir = self.data_dir / "coalitions"

    def load_similarity_matrices(self) -> Dict[int, np.ndarray]:
        """
        Load all similarity matrices from saved JSON files.
        
        Returns:
            Dictionary mapping round_number -> similarity_matrix
        """
        matrices = {}
        
        # Find all matrix files
        matrix_files = sorted(self.similarity_dir.glob("matrix_round_*.json"))
        
        for file_path in matrix_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                round_num = data['round']
                matrix = np.array(data['matrix'])
                matrices[round_num] = matrix
        
        print(f"📊 Loaded {len(matrices)} similarity matrices")
        return matrices

    def load_coalitions(self) -> Dict[int, List[List[str]]]:
        """
        Load coalition formations from saved JSON files.
        
        Returns:
            Dictionary mapping round_number -> list_of_coalitions
        """
        coalitions = {}
        
        coalition_files = sorted(self.coalition_dir.glob("coalitions_round_*.json"))
        
        for file_path in coalition_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                round_num = data['round']
                coalition_list = data['coalitions']
                coalitions[round_num] = coalition_list
        
        print(f"🏘️  Loaded {len(coalitions)} coalition formations")
        return coalitions

    def calculate_cohesion(
        self,
        similarity_matrix: np.ndarray,
        coalition: List[str],
        agent_names: List[str]
    ) -> float:
        """
        Calculate intra-coalition cohesion (average similarity within group).
        
        High cohesion = agents in the coalition are very similar to each other
        
        Args:
            similarity_matrix: Full NxN similarity matrix
            coalition: List of agent names in this coalition
            agent_names: Ordered list of all agent names (for indexing)
        
        Returns:
            Average similarity score within the coalition (0-1)
        """
        # Get indices for coalition members
        agent_index = {name: i for i, name in enumerate(agent_names)}
        
        # Filter coalition members to only those that exist in agent_names
        valid_coalition = [name for name in coalition if name in agent_index]
        
        if len(valid_coalition) < 2:
            return 1.0  # Single-member or empty coalition is perfectly cohesive
        
        indices = [agent_index[name] for name in valid_coalition]
        
        # Extract similarities between all pairs in coalition
        similarities = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i = indices[i]
                idx_j = indices[j]
                similarities.append(similarity_matrix[idx_i][idx_j])
        
        return np.mean(similarities) if similarities else 0.0

    def calculate_separation(
        self,
        similarity_matrix: np.ndarray,
        coalition_i: List[str],
        coalition_j: List[str],
        agent_names: List[str]
    ) -> float:
        """
        Calculate inter-coalition separation (average dissimilarity between groups).
        
        High separation = agents in different coalitions are very different
        
        Args:
            similarity_matrix: Full NxN similarity matrix
            coalition_i: First coalition
            coalition_j: Second coalition
            agent_names: Ordered list of all agent names
        
        Returns:
            Average dissimilarity (1 - similarity) between the two coalitions
        """
        agent_index = {name: i for i, name in enumerate(agent_names)}
        
        # Filter to valid names only
        valid_i = [name for name in coalition_i if name in agent_index]
        valid_j = [name for name in coalition_j if name in agent_index]
        
        if not valid_i or not valid_j:
            return 0.0
        
        indices_i = [agent_index[name] for name in valid_i]
        indices_j = [agent_index[name] for name in valid_j]
        
        # Cross-coalition similarities
        similarities = []
        for idx_i in indices_i:
            for idx_j in indices_j:
                similarities.append(similarity_matrix[idx_i][idx_j])
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        return 1.0 - avg_similarity  # Return dissimilarity

    def calculate_churn_rate(
        self,
        coalitions_before: List[List[str]],
        coalitions_after: List[List[str]]
    ) -> Tuple[float, List[Dict]]:
        """
        Calculate churn rate between two coalition configurations.
        
        Churn rate = percentage of agents who switched coalitions
        
        Args:
            coalitions_before: Coalition structure at time T
            coalitions_after: Coalition structure at time T+1
        
        Returns:
            Tuple of (churn_rate, list_of_migrations)
        """
        # Build agent->coalition mapping for both timepoints
        def build_mapping(coalitions):
            mapping = {}
            for coalition_id, members in enumerate(coalitions):
                for agent in members:
                    mapping[agent] = coalition_id
            return mapping
        
        map_before = build_mapping(coalitions_before)
        map_after = build_mapping(coalitions_after)
        
        # Find migrations
        all_agents = set(map_before.keys()) | set(map_after.keys())
        migrations = []
        
        for agent in all_agents:
            before = map_before.get(agent, -1)
            after = map_after.get(agent, -1)
            
            if before != after:
                migrations.append({
                    'agent': agent,
                    'from_coalition': before,
                    'to_coalition': after
                })
        
        churn_rate = len(migrations) / len(all_agents) if all_agents else 0.0
        
        return churn_rate, migrations

    def calculate_modularity(
        self,
        similarity_matrix: np.ndarray,
        coalitions: List[List[str]],
        agent_names: List[str]
    ) -> float:
        """
        Calculate modularity Q - measures quality of coalition structure.
        
        Q close to 1 = strong community structure (high intra-coalition, low inter-coalition similarity)
        Q close to 0 = no meaningful community structure
        
        This is the same metric used by Louvain algorithm.
        
        Args:
            similarity_matrix: Full NxN similarity matrix
            coalitions: List of coalitions
            agent_names: Ordered list of all agent names
        
        Returns:
            Modularity score Q
        """
        agent_index = {name: i for i, name in enumerate(agent_names)}
        n = len(agent_names)
        
        # Build coalition membership array
        membership = np.full(n, -1, dtype=int)  # -1 for unassigned
        
        for coalition_id, members in enumerate(coalitions):
            for agent in members:
                if agent in agent_index:
                    membership[agent_index[agent]] = coalition_id
        
        # Calculate modularity
        # Q = (1/2m) * sum_ij [(A_ij - k_i*k_j/2m) * delta(c_i, c_j)]
        # For weighted networks with similarities as weights
        
        total_weight = np.sum(similarity_matrix) / 2  # Divide by 2 for undirected
        
        if total_weight == 0:
            return 0.0
        
        Q = 0.0
        for i in range(n):
            for j in range(n):
                if membership[i] >= 0 and membership[i] == membership[j]:
                    A_ij = similarity_matrix[i][j]
                    k_i = np.sum(similarity_matrix[i])
                    k_j = np.sum(similarity_matrix[j])
                    Q += A_ij - (k_i * k_j) / (2 * total_weight)
        
        Q /= (2 * total_weight)
        return Q

    def generate_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics for the entire simulation.
        
        Returns:
            Dictionary with all key metrics
        """
        print("\n📊 Calculating summary statistics...")
        
        matrices = self.load_similarity_matrices()
        coalitions = self.load_coalitions()
        
        if not matrices:
            return {"error": "No similarity matrices found"}
        
        # Get agent names from first matrix file
        first_file = sorted(self.similarity_dir.glob("matrix_round_*.json"))[0]
        with open(first_file, 'r') as f:
            agent_names = json.load(f)['agent_names']
        
        stats = {
            'total_rounds': len(matrices),
            'total_agents': len(agent_names),
            'agent_names': agent_names,
            'coalition_events': len(coalitions),
            'per_round_metrics': {}
        }
        
        # Per-round metrics
        for round_num in sorted(matrices.keys()):
            matrix = matrices[round_num]
            
            # Get upper triangle indices (exclude diagonal)
            upper_tri_indices = np.triu_indices_from(matrix, k=1)
            upper_tri_values = matrix[upper_tri_indices]
            
            round_stats = {
                'avg_similarity': float(np.mean(upper_tri_values)),
                'max_similarity': float(np.max(upper_tri_values)),
                'min_similarity': float(np.min(upper_tri_values))
            }
            
            # If coalitions exist for this round, calculate cohesion and modularity
            if round_num in coalitions:
                coalition_list = coalitions[round_num]
                
                # Cohesion per coalition
                cohesions = []
                for coalition in coalition_list:
                    cohesion = self.calculate_cohesion(matrix, coalition, agent_names)
                    cohesions.append(cohesion)
                
                round_stats['num_coalitions'] = len(coalition_list)
                round_stats['coalition_sizes'] = [len(c) for c in coalition_list]
                round_stats['avg_cohesion'] = float(np.mean(cohesions)) if cohesions else 0.0
                round_stats['min_cohesion'] = float(np.min(cohesions)) if cohesions else 0.0
                round_stats['max_cohesion'] = float(np.max(cohesions)) if cohesions else 0.0
                
                # Modularity
                modularity = self.calculate_modularity(matrix, coalition_list, agent_names)
                round_stats['modularity'] = float(modularity)
            
            stats['per_round_metrics'][round_num] = round_stats
        
        # Churn rate between coalition formations
        if len(coalitions) >= 2:
            coalition_rounds = sorted(coalitions.keys())
            churn_rates = []
            
            for i in range(len(coalition_rounds) - 1):
                round_before = coalition_rounds[i]
                round_after = coalition_rounds[i + 1]
                
                churn, migrations = self.calculate_churn_rate(
                    coalitions[round_before],
                    coalitions[round_after]
                )
                
                churn_rates.append({
                    'from_round': round_before,
                    'to_round': round_after,
                    'churn_rate': churn,
                    'num_migrations': len(migrations),
                    'migrations': migrations
                })
            
            stats['churn_analysis'] = churn_rates
            stats['avg_churn_rate'] = float(np.mean([c['churn_rate'] for c in churn_rates]))
        
        print("   ✅ Summary statistics calculated")
        return stats