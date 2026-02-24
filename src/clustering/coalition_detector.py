"""
Coalition Detector
Applies clustering algorithms to similarity matrices to form agent coalitions.

Supports three algorithms:
1. Louvain (modularity-based community detection)
2. Threshold-based (agents with similarity > threshold group together)
3. Hierarchical (agglomerative clustering with dendrogram)
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict


class CoalitionDetector:
    """
    Detects and forms agent coalitions based on behavioral similarity.
    """

    def __init__(
        self,
        agent_names: List[str],
        min_coalition_size: int = 2,
        max_coalitions: int = 5,
        save_dir: str = "data/coalitions"
    ):
        self.agent_names = agent_names
        self.min_coalition_size = min_coalition_size
        self.max_coalitions = max_coalitions
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # History of coalition formations
        self.coalition_history: List[Dict[str, Any]] = []

        print(f"\n🔍 CoalitionDetector initialized")
        print(f"   Agents: {len(agent_names)}")
        print(f"   Min coalition size: {min_coalition_size}")
        print(f"   Max coalitions: {max_coalitions}")

    def detect_coalitions(
        self,
        similarity_matrix: np.ndarray,
        round_number: int,
        method: str = "threshold",
        threshold: float = 0.6,
        n_clusters: int = 3
    ) -> List[List[str]]:
        """
        Detect coalitions using specified clustering method.

        Args:
            similarity_matrix: NxN similarity scores
            round_number: Current round number
            method: "threshold", "louvain", or "hierarchical"
            threshold: Similarity threshold for threshold method
            n_clusters: Number of clusters for hierarchical method

        Returns:
            List of coalitions, where each coalition is a list of agent names
        """

        print(f"\n🔍 Detecting coalitions after round {round_number}...")
        print(f"   Method: {method}")

        if method == "threshold":
            coalitions = self._threshold_clustering(similarity_matrix, threshold)
        elif method == "louvain":
            coalitions = self._louvain_clustering(similarity_matrix)
        elif method == "hierarchical":
            coalitions = self._hierarchical_clustering(similarity_matrix, n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Filter out coalitions that are too small
        coalitions = [c for c in coalitions if len(c) >= self.min_coalition_size]

        # If too many coalitions, merge the smallest ones
        if len(coalitions) > self.max_coalitions:
            coalitions = self._merge_smallest_coalitions(
                coalitions,
                similarity_matrix,
                self.max_coalitions
            )

        # Store in history
        self._save_coalition_snapshot(round_number, coalitions, method)

        # Print results
        self._print_coalitions(coalitions, round_number)

        return coalitions

    # ── CLUSTERING ALGORITHM 1: THRESHOLD-BASED ──────────────────────────────

    def _threshold_clustering(
        self,
        similarity_matrix: np.ndarray,
        threshold: float = 0.6
    ) -> List[List[str]]:
        """
        Group agents where pairwise similarity exceeds threshold.
        Uses a greedy approach to form cohesive groups.
        """

        n = len(self.agent_names)
        assigned = set()
        coalitions = []

        # Sort agents by their average similarity to others (most central first)
        centrality = np.mean(similarity_matrix, axis=1)
        sorted_indices = np.argsort(-centrality)  # Descending order

        for seed_idx in sorted_indices:
            if seed_idx in assigned:
                continue

            # Start new coalition with this seed agent
            coalition_indices = {seed_idx}
            assigned.add(seed_idx)

            # Add agents similar to the seed
            for candidate_idx in range(n):
                if candidate_idx in assigned:
                    continue

                # Check similarity to all existing coalition members
                avg_sim_to_coalition = np.mean([
                    similarity_matrix[candidate_idx][member_idx]
                    for member_idx in coalition_indices
                ])

                if avg_sim_to_coalition > threshold:
                    coalition_indices.add(candidate_idx)
                    assigned.add(candidate_idx)

            # Convert indices to names
            coalition_names = [self.agent_names[idx] for idx in coalition_indices]
            coalitions.append(coalition_names)

        # Handle unassigned agents (isolates)
        if len(assigned) < n:
            isolates = [
                self.agent_names[i]
                for i in range(n)
                if i not in assigned
            ]
            if isolates:
                coalitions.append(isolates)

        return coalitions

    # ── CLUSTERING ALGORITHM 2: LOUVAIN (MODULARITY OPTIMIZATION) ────────────

    def _louvain_clustering(
        self,
        similarity_matrix: np.ndarray
    ) -> List[List[str]]:
        """
        Louvain community detection algorithm.
        Maximizes modularity to find natural communities.
        """

        try:
            import networkx as nx
            from community import community_louvain
        except ImportError:
            print("   ⚠️  Louvain requires 'python-louvain' package")
            print("   ⚠️  Falling back to threshold clustering")
            return self._threshold_clustering(similarity_matrix, threshold=0.5)

        # Build weighted graph
        G = nx.Graph()

        n = len(self.agent_names)
        for i in range(n):
            G.add_node(i, name=self.agent_names[i])

        for i in range(n):
            for j in range(i + 1, n):
                weight = similarity_matrix[i][j]
                if weight > 0.3:  # Only add edges with meaningful similarity
                    G.add_edge(i, j, weight=weight)

        # Detect communities
        partition = community_louvain.best_partition(G, weight='weight')

        # Convert to coalitions
        communities = defaultdict(list)
        for node_idx, community_id in partition.items():
            communities[community_id].append(self.agent_names[node_idx])

        coalitions = list(communities.values())
        return coalitions

    # ── CLUSTERING ALGORITHM 3: HIERARCHICAL CLUSTERING ──────────────────────

    def _hierarchical_clustering(
        self,
        similarity_matrix: np.ndarray,
        n_clusters: int = 3
    ) -> List[List[str]]:
        """
        Hierarchical agglomerative clustering.
        Forms exactly n_clusters coalitions.
        """

        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
        except ImportError:
            print("   ⚠️  Hierarchical clustering requires scipy")
            print("   ⚠️  Falling back to threshold clustering")
            return self._threshold_clustering(similarity_matrix, threshold=0.5)

        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)

        # Ensure valid distance matrix
        distance_matrix = np.clip(distance_matrix, 0, 1)

        # Convert to condensed form for scipy
        condensed_dist = squareform(distance_matrix, checks=False)

        # Compute hierarchical clustering
        linkage_matrix = linkage(condensed_dist, method='ward')

        # Cut dendrogram to get n_clusters
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        # Group agents by cluster
        coalitions_dict = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_labels):
            coalitions_dict[cluster_id].append(self.agent_names[idx])

        coalitions = list(coalitions_dict.values())
        return coalitions

    # ── COALITION MERGING ─────────────────────────────────────────────────────

    def _merge_smallest_coalitions(
        self,
        coalitions: List[List[str]],
        similarity_matrix: np.ndarray,
        max_coalitions: int
    ) -> List[List[str]]:
        """
        If there are too many coalitions, merge the smallest/least cohesive ones.
        """

        while len(coalitions) > max_coalitions:
            # Find the smallest coalition
            sizes = [len(c) for c in coalitions]
            smallest_idx = sizes.index(min(sizes))
            smallest_coalition = coalitions.pop(smallest_idx)

            # Find which remaining coalition it's most similar to
            best_merge_idx = 0
            best_similarity = -1

            agent_index = {name: i for i, name in enumerate(self.agent_names)}

            for idx, coalition in enumerate(coalitions):
                # Compute average similarity between coalitions
                similarities = []
                for agent_a in smallest_coalition:
                    for agent_b in coalition:
                        i = agent_index[agent_a]
                        j = agent_index[agent_b]
                        similarities.append(similarity_matrix[i][j])

                avg_sim = np.mean(similarities) if similarities else 0
                if avg_sim > best_similarity:
                    best_similarity = avg_sim
                    best_merge_idx = idx

            # Merge into best match
            coalitions[best_merge_idx].extend(smallest_coalition)

        return coalitions

    # ── COALITION STABILITY ANALYSIS ──────────────────────────────────────────

    def analyze_stability(self) -> Dict[str, Any]:
        """
        Analyze coalition stability across multiple clustering events.
        Returns metrics like churn rate, persistence, etc.
        """

        if len(self.coalition_history) < 2:
            return {"error": "Need at least 2 clustering events to analyze stability"}

        # Compare last two clustering events
        prev_event = self.coalition_history[-2]
        curr_event = self.coalition_history[-1]

        prev_coalitions = prev_event["coalitions"]
        curr_coalitions = curr_event["coalitions"]

        # Build agent->coalition mapping
        prev_map = {}
        for i, coalition in enumerate(prev_coalitions):
            for agent in coalition:
                prev_map[agent] = i

        curr_map = {}
        for i, coalition in enumerate(curr_coalitions):
            for agent in coalition:
                curr_map[agent] = i

        # Count migrations
        migrations = []
        for agent in self.agent_names:
            prev_coalition = prev_map.get(agent, -1)
            curr_coalition = curr_map.get(agent, -1)

            if prev_coalition != curr_coalition:
                migrations.append({
                    "agent": agent,
                    "from_coalition": prev_coalition,
                    "to_coalition": curr_coalition
                })

        churn_rate = len(migrations) / len(self.agent_names)

        stability_metrics = {
            "total_agents": len(self.agent_names),
            "agents_migrated": len(migrations),
            "churn_rate": churn_rate,
            "migrations": migrations,
            "prev_num_coalitions": len(prev_coalitions),
            "curr_num_coalitions": len(curr_coalitions),
            "coalition_size_change": len(curr_coalitions) - len(prev_coalitions)
        }

        return stability_metrics

    # ── DISPLAY AND SAVE ──────────────────────────────────────────────────────

    def _print_coalitions(self, coalitions: List[List[str]], round_number: int):
        """Print detected coalitions in readable format"""

        print(f"\n🏘️  Detected {len(coalitions)} coalitions after round {round_number}:")

        for i, coalition in enumerate(coalitions, 1):
            print(f"\n   Coalition {i} ({len(coalition)} members):")
            for agent in coalition:
                print(f"      • {agent}")

        # Calculate coalition statistics
        sizes = [len(c) for c in coalitions]
        print(f"\n   📊 Coalition Statistics:")
        print(f"      Average size: {np.mean(sizes):.1f}")
        print(f"      Largest: {max(sizes)}")
        print(f"      Smallest: {min(sizes)}")

    def _save_coalition_snapshot(
        self,
        round_number: int,
        coalitions: List[List[str]],
        method: str
    ):
        """Save coalition snapshot to file"""

        snapshot = {
            "round": round_number,
            "method": method,
            "num_coalitions": len(coalitions),
            "coalitions": coalitions,
            "coalition_sizes": [len(c) for c in coalitions]
        }

        self.coalition_history.append(snapshot)

        # Save to JSON
        save_path = self.save_dir / f"coalitions_round_{round_number:02d}.json"
        with open(save_path, 'w') as f:
            json.dump(snapshot, f, indent=2)

        print(f"   💾 Coalitions saved to: {save_path}")

    def get_current_coalitions(self) -> List[List[str]]:
        """Get the most recent coalition configuration"""
        if not self.coalition_history:
            return [self.agent_names]  # All in one group by default

        return self.coalition_history[-1]["coalitions"]

    def get_coalition_history(self) -> List[Dict[str, Any]]:
        """Get full history of all coalition formations"""
        return self.coalition_history