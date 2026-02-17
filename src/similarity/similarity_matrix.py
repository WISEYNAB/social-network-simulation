"""
Similarity Matrix Calculator
Computes multi-dimensional behavioral similarity between agents
after every discussion round.

Dimensions:
    1. Semantic Similarity    (25%) - text embedding cosine similarity
    2. Topic Overlap          (20%) - shared keyword/topic coverage
    3. Sentiment Alignment    (15%) - emotional tone matching
    4. Interaction Frequency  (15%) - how often they engage each other
    5. Agreement Rate         (25%) - agreement vs disagreement ratio
"""

import json
import numpy as np
import warnings
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings('ignore')


# ── WEIGHTS FOR EACH DIMENSION ────────────────────────────────────────────────
WEIGHTS = {
    "semantic":    0.25,
    "topic":       0.20,
    "sentiment":   0.15,
    "interaction": 0.15,
    "agreement":   0.25,
}


class SimilarityMatrix:
    """
    Maintains and updates a pairwise similarity matrix
    for all agents in the simulation.

    The matrix is recalculated after every discussion round
    using conversation history as input.
    """

    def __init__(
        self,
        agent_names: List[str],
        ollama_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text:latest",
        save_dir: str = "data/similarity"
    ):
        self.agent_names = agent_names
        self.n = len(agent_names)
        self.ollama_url = ollama_url
        self.embed_model = embed_model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Agent index lookup
        self.agent_index = {name: i for i, name in enumerate(agent_names)}

        # The main similarity matrix — starts as identity (1s on diagonal)
        self.matrix = np.eye(self.n, dtype=float)

        # History of matrix snapshots per round
        self.matrix_history: List[Dict[str, Any]] = []

        # Sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Cache embeddings to avoid recomputing
        self.embedding_cache: Dict[str, List[float]] = {}

        print(f"\n📐 SimilarityMatrix initialized for {self.n} agents")
        print(f"   Agents: {', '.join(agent_names)}")
        print(f"   Embed model: {embed_model}")
        print(f"   Weights: {WEIGHTS}")

    # ── PUBLIC METHOD: UPDATE AFTER EACH ROUND ────────────────────────────────

    def update(
        self,
        round_number: int,
        conversation_history: List[Dict[str, Any]],
        agent_stats: List[Dict[str, Any]]
    ):
        """
        Recompute the full similarity matrix after a discussion round.

        Args:
            round_number: Current round number
            conversation_history: All posts and comments so far
            agent_stats: Stats dict from each agent (agreements, disagreements etc.)
        """

        print(f"\n📐 Computing similarity matrix after round {round_number}...")

        # Extract per-agent data from history
        agent_texts      = self._collect_agent_texts(conversation_history)
        interaction_map  = self._build_interaction_map(conversation_history)
        agreement_map    = self._build_agreement_map(agent_stats)

        # Compute each dimension
        semantic_matrix    = self._compute_semantic_similarity(agent_texts)
        topic_matrix       = self._compute_topic_overlap(agent_texts)
        sentiment_matrix   = self._compute_sentiment_alignment(agent_texts)
        interaction_matrix = self._compute_interaction_frequency(interaction_map)
        agreement_matrix   = self._compute_agreement_rate(agreement_map)

        # Combine all dimensions using weights
        self.matrix = (
            WEIGHTS["semantic"]    * semantic_matrix    +
            WEIGHTS["topic"]       * topic_matrix       +
            WEIGHTS["sentiment"]   * sentiment_matrix   +
            WEIGHTS["interaction"] * interaction_matrix +
            WEIGHTS["agreement"]   * agreement_matrix
        )

        # Force diagonal to 1.0 (agent is always 100% similar to itself)
        np.fill_diagonal(self.matrix, 1.0)

        # Force symmetry (average of both directions)
        self.matrix = (self.matrix + self.matrix.T) / 2
        np.fill_diagonal(self.matrix, 1.0)

        # Clip values to valid range
        self.matrix = np.clip(self.matrix, 0.0, 1.0)

        # Save snapshot
        self._save_snapshot(round_number)

        # Print readable matrix
        self._print_matrix(round_number)

        print(f"   ✅ Similarity matrix updated!")

    # ── DIMENSION 1: SEMANTIC SIMILARITY ─────────────────────────────────────

    def _compute_semantic_similarity(
        self,
        agent_texts: Dict[str, str]
    ) -> np.ndarray:
        """
        Compute cosine similarity between agent text embeddings.
        Uses nomic-embed-text via Ollama for local embedding.
        """

        print(f"   [1/5] Computing semantic similarity...", end="", flush=True)

        embeddings = []

        for name in self.agent_names:
            text = agent_texts.get(name, "")

            if not text.strip():
                # No posts yet — use zero vector
                embeddings.append(None)
                continue

            # Check cache first
            cache_key = f"{name}:{hash(text)}"
            if cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
                continue

            # Get embedding from Ollama
            embedding = self._get_ollama_embedding(text)

            if embedding:
                self.embedding_cache[cache_key] = embedding
                embeddings.append(embedding)
            else:
                embeddings.append(None)

        # Build matrix — handle missing embeddings
        matrix = np.zeros((self.n, self.n))

        valid_embeddings = []
        valid_indices = []

        for i, emb in enumerate(embeddings):
            if emb is not None:
                valid_embeddings.append(emb)
                valid_indices.append(i)

        if len(valid_embeddings) >= 2:
            emb_array = np.array(valid_embeddings)
            # Normalize before cosine similarity
            emb_array = normalize(emb_array)
            sim_matrix = cosine_similarity(emb_array)

            # Place scores into full matrix
            for i, vi in enumerate(valid_indices):
                for j, vj in enumerate(valid_indices):
                    matrix[vi][vj] = sim_matrix[i][j]

        print(f" ✓ ({len(valid_embeddings)}/{self.n} agents had posts)")
        return matrix

    def _get_ollama_embedding(self, text: str) -> Optional[List[float]]:
        """
        Call Ollama embedding API to get text vector.
        Uses nomic-embed-text model.
        """
        try:
            # Truncate very long texts
            text = text[:2000] if len(text) > 2000 else text

            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": self.embed_model,
                    "prompt": text
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("embedding", None)
            else:
                print(f"\n   ⚠️  Ollama embedding error: {response.status_code}")
                return None

        except requests.exceptions.ConnectionError:
            print(f"\n   ⚠️  Ollama not reachable — skipping semantic similarity")
            return None
        except Exception as e:
            print(f"\n   ⚠️  Embedding error: {e}")
            return None

    # ── DIMENSION 2: TOPIC OVERLAP ────────────────────────────────────────────

    def _compute_topic_overlap(
        self,
        agent_texts: Dict[str, str]
    ) -> np.ndarray:
        """
        Compute Jaccard similarity between agents' topic keyword sets.
        Extracts meaningful words and compares vocabulary overlap.
        """

        print(f"   [2/5] Computing topic overlap...", end="", flush=True)

        # Extract keyword sets per agent
        agent_keywords = {}

        for name in self.agent_names:
            text = agent_texts.get(name, "")
            agent_keywords[name] = self._extract_keywords(text)

        # Build Jaccard similarity matrix
        matrix = np.zeros((self.n, self.n))

        for i, name_i in enumerate(self.agent_names):
            for j, name_j in enumerate(self.agent_names):
                if i == j:
                    matrix[i][j] = 1.0
                    continue

                set_i = agent_keywords[name_i]
                set_j = agent_keywords[name_j]

                if not set_i or not set_j:
                    matrix[i][j] = 0.0
                    continue

                # Jaccard similarity
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                matrix[i][j] = intersection / union if union > 0 else 0.0

        print(f" ✓")
        return matrix

    def _extract_keywords(self, text: str) -> set:
        """
        Extract meaningful keywords from text.
        Removes stopwords and short words.
        """

        # Common stopwords to ignore
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "was", "are", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "shall",
            "i", "you", "he", "she", "we", "they", "it", "this", "that",
            "my", "your", "his", "her", "our", "their", "its", "what",
            "which", "who", "when", "where", "how", "why", "not", "no",
            "just", "more", "very", "so", "if", "about", "up", "out",
            "there", "then", "than", "also", "into", "like", "as", "all",
            "can", "get", "got", "let", "now", "even", "see", "think",
            "really", "actually", "here", "some", "any", "one", "two",
            "every", "still", "well", "back", "way", "re", "ve", "ll", "s"
        }

        # Clean text
        text = text.lower()
        # Remove punctuation and emojis (keep letters and numbers)
        cleaned = ''.join(c if c.isalpha() or c.isspace() else ' ' for c in text)
        words = cleaned.split()

        # Filter: remove stopwords and short words
        keywords = {
            word for word in words
            if word not in stopwords and len(word) > 3
        }

        return keywords

    # ── DIMENSION 3: SENTIMENT ALIGNMENT ─────────────────────────────────────

    def _compute_sentiment_alignment(
        self,
        agent_texts: Dict[str, str]
    ) -> np.ndarray:
        """
        Compare average sentiment scores between agent pairs.
        Agents with similar emotional tones score higher.
        """

        print(f"   [3/5] Computing sentiment alignment...", end="", flush=True)

        # Get average sentiment per agent
        agent_sentiments = {}

        for name in self.agent_names:
            text = agent_texts.get(name, "")

            if not text.strip():
                agent_sentiments[name] = 0.0  # Neutral default
                continue

            scores = self.sentiment_analyzer.polarity_scores(text)
            # Use compound score: -1 (very negative) to +1 (very positive)
            agent_sentiments[name] = scores["compound"]

        # Build similarity matrix
        matrix = np.zeros((self.n, self.n))

        for i, name_i in enumerate(self.agent_names):
            for j, name_j in enumerate(self.agent_names):
                if i == j:
                    matrix[i][j] = 1.0
                    continue

                sent_i = agent_sentiments[name_i]
                sent_j = agent_sentiments[name_j]

                # Similarity = 1 - normalized absolute difference
                # Max difference is 2.0 (-1 to +1), so divide by 2
                diff = abs(sent_i - sent_j)
                matrix[i][j] = 1.0 - (diff / 2.0)

        print(f" ✓ (scores: { {k: round(v,2) for k, v in agent_sentiments.items()} })")
        return matrix

    # ── DIMENSION 4: INTERACTION FREQUENCY ───────────────────────────────────

    def _compute_interaction_frequency(
        self,
        interaction_map: Dict[str, Dict[str, int]]
    ) -> np.ndarray:
        """
        Compute similarity based on how often agents interact.
        More interactions = higher similarity score.
        """

        print(f"   [4/5] Computing interaction frequency...", end="", flush=True)

        matrix = np.zeros((self.n, self.n))

        # Find max interactions for normalization
        all_counts = []
        for name_i in self.agent_names:
            for name_j in self.agent_names:
                count = interaction_map.get(name_i, {}).get(name_j, 0)
                count += interaction_map.get(name_j, {}).get(name_i, 0)
                all_counts.append(count)

        max_interactions = max(all_counts) if all_counts else 1
        max_interactions = max(max_interactions, 1)  # Avoid division by zero

        for i, name_i in enumerate(self.agent_names):
            for j, name_j in enumerate(self.agent_names):
                if i == j:
                    matrix[i][j] = 1.0
                    continue

                # Count interactions in both directions
                count_ij = interaction_map.get(name_i, {}).get(name_j, 0)
                count_ji = interaction_map.get(name_j, {}).get(name_i, 0)
                total = count_ij + count_ji

                # Normalize to 0-1
                matrix[i][j] = min(total / max_interactions, 1.0)

        print(f" ✓")
        return matrix

    # ── DIMENSION 5: AGREEMENT RATE ───────────────────────────────────────────

    def _compute_agreement_rate(
        self,
        agreement_map: Dict[str, Dict[str, Dict[str, int]]]
    ) -> np.ndarray:
        """
        Compute similarity based on agreement vs disagreement ratios.
        Higher agreement rate = higher similarity.
        """

        print(f"   [5/5] Computing agreement rate...", end="", flush=True)

        matrix = np.zeros((self.n, self.n))

        for i, name_i in enumerate(self.agent_names):
            for j, name_j in enumerate(self.agent_names):
                if i == j:
                    matrix[i][j] = 1.0
                    continue

                # Get agreement/disagreement counts both ways
                agree_ij   = agreement_map.get(name_i, {}).get(name_j, {}).get("agree", 0)
                disagree_ij = agreement_map.get(name_i, {}).get(name_j, {}).get("disagree", 0)
                agree_ji   = agreement_map.get(name_j, {}).get(name_i, {}).get("agree", 0)
                disagree_ji = agreement_map.get(name_j, {}).get(name_i, {}).get("disagree", 0)

                total_agree    = agree_ij + agree_ji
                total_disagree = disagree_ij + disagree_ji
                total          = total_agree + total_disagree

                if total == 0:
                    # No interaction yet — neutral score
                    matrix[i][j] = 0.5
                else:
                    matrix[i][j] = total_agree / total

        print(f" ✓")
        return matrix

    # ── DATA EXTRACTION HELPERS ───────────────────────────────────────────────

    def _collect_agent_texts(
        self,
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Collect all text written by each agent across all rounds.
        Combines posts and comments into one text blob per agent.
        """

        agent_texts = {name: [] for name in self.agent_names}

        for message in conversation_history:
            msg_type = message.get("type", "")

            if msg_type == "post":
                author = message.get("author", "")
                content = message.get("content", "")
                if author in agent_texts and content:
                    agent_texts[author].append(content)

            elif msg_type == "comment":
                commenter = message.get("commenter", "")
                comment = message.get("comment", "")
                if commenter in agent_texts and comment:
                    agent_texts[commenter].append(comment)

        # Join all text per agent
        return {
            name: " ".join(texts)
            for name, texts in agent_texts.items()
        }

    def _build_interaction_map(
        self,
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, int]]:
        """
        Build a map of how many times each agent commented on others.
        Format: interaction_map[commenter][target] = count
        """

        interaction_map = {
            name: {other: 0 for other in self.agent_names}
            for name in self.agent_names
        }

        for message in conversation_history:
            if message.get("type") == "comment":
                commenter = message.get("commenter", "")
                target    = message.get("target_author", "")

                if commenter in interaction_map and target in interaction_map:
                    interaction_map[commenter][target] += 1

        return interaction_map

    def _build_agreement_map(
        self,
        agent_stats: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Dict[str, int]]]:
        """
        Build agreement/disagreement map from agent stats.
        Note: Currently uses global stats — we'll refine this
        to be pairwise in a future iteration.
        """

        # For now we use a simple proxy:
        # agents with similar agreement tendencies score higher
        agreement_map = {
            name: {other: {"agree": 0, "disagree": 0} for other in self.agent_names}
            for name in self.agent_names
        }

        # Build a lookup for stats
        stats_lookup = {s["name"]: s for s in agent_stats}

        for name_i in self.agent_names:
            stats_i = stats_lookup.get(name_i, {})
            agree_rate_i = stats_i.get("agreements", 0)

            for name_j in self.agent_names:
                if name_i == name_j:
                    continue

                stats_j = stats_lookup.get(name_j, {})
                agree_rate_j = stats_j.get("agreements", 0)

                # Proxy: if both agents have similar agreement rates, they likely agree
                total_i = agree_rate_i + stats_i.get("disagreements", 0) + 1
                total_j = agree_rate_j + stats_j.get("disagreements", 0) + 1

                rate_i = agree_rate_i / total_i
                rate_j = agree_rate_j / total_j

                # If both tend to agree, mark as agreeable pair
                avg_rate = (rate_i + rate_j) / 2
                agreement_map[name_i][name_j]["agree"] = int(avg_rate * 10)
                agreement_map[name_i][name_j]["disagree"] = int((1 - avg_rate) * 10)

        return agreement_map

    # ── DISPLAY AND SAVE ──────────────────────────────────────────────────────

    def _print_matrix(self, round_number: int):
        """Print the similarity matrix in a readable table format"""

        print(f"\n📊 Similarity Matrix after Round {round_number}:")
        print(f"{'':>28}", end="")

        # Short names for header
        short_names = [name[:8] for name in self.agent_names]

        for name in short_names:
            print(f"{name:>10}", end="")
        print()

        for i, name in enumerate(self.agent_names):
            print(f"  {name:<26}", end="")
            for j in range(self.n):
                val = self.matrix[i][j]
                # Color coding in terminal
                if i == j:
                    print(f"{'1.00':>10}", end="")
                elif val >= 0.7:
                    print(f"\033[92m{val:>10.2f}\033[0m", end="")  # Green - high similarity
                elif val >= 0.5:
                    print(f"\033[93m{val:>10.2f}\033[0m", end="")  # Yellow - medium
                else:
                    print(f"\033[91m{val:>10.2f}\033[0m", end="")  # Red - low
            print()

    def _save_snapshot(self, round_number: int):
        """Save matrix snapshot to disk after each round"""

        snapshot = {
            "round": round_number,
            "agent_names": self.agent_names,
            "matrix": self.matrix.tolist()
        }

        self.matrix_history.append(snapshot)

        # Save to JSON file
        save_path = self.save_dir / f"matrix_round_{round_number:02d}.json"
        with open(save_path, 'w') as f:
            json.dump(snapshot, f, indent=2)

        print(f"   💾 Matrix saved to: {save_path}")

    def get_top_pairs(self, top_n: int = 5) -> List[Tuple[str, str, float]]:
        """Return the most similar agent pairs"""

        pairs = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                pairs.append((
                    self.agent_names[i],
                    self.agent_names[j],
                    self.matrix[i][j]
                ))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_n]

    def get_matrix_as_dict(self) -> Dict[str, Dict[str, float]]:
        """Return matrix as nested dictionary for easy access"""
        return {
            self.agent_names[i]: {
                self.agent_names[j]: float(self.matrix[i][j])
                for j in range(self.n)
            }
            for i in range(self.n)
        }