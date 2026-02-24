"""
GroupChat - Manages multi-agent discussions
Updated to compute similarity matrix after every round
"""

import random
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from agents.social_agent import SocialAgent
    from utils.logger import ConversationLogger
    from similarity.similarity_matrix import SimilarityMatrix
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agents.social_agent import SocialAgent
    from utils.logger import ConversationLogger
    from similarity.similarity_matrix import SimilarityMatrix


class GroupChat:
    """
    Simulates a social media group discussion.
    Computes similarity matrix after every round.
    """

    def __init__(
        self,
        agents: List[SocialAgent],
        chat_name: str = "Main Community",
        logger: Optional[ConversationLogger] = None
    ):
        self.agents = agents
        self.chat_name = chat_name
        self.logger = logger or ConversationLogger()

        self.conversation_history: List[Dict[str, Any]] = []
        self.current_round = 0
        self.round_summaries: List[Dict[str, Any]] = []

        # Initialize similarity matrix
        self.similarity_matrix = SimilarityMatrix(
            agent_names=[a.name for a in agents]
        )

        print(f"\n🏘️  GroupChat '{self.chat_name}' initialized")
        print(f"   Members: {', '.join([a.name for a in self.agents])}")

    def run_round(
        self,
        topic: str,
        topic_type: str = "general",
        enable_comments: bool = True
    ) -> Dict[str, Any]:
        """Run one full discussion round then update similarity matrix"""

        self.current_round += 1
        round_number = self.current_round

        self.logger.log_round_start(round_number, topic, topic_type)

        # ── PHASE 1: POSTING ──────────────────────────────────────────
        print(f"\n📝 PHASE 1: Agents posting...")

        round_posts = []
        context = self.conversation_history[-10:] if self.conversation_history else []
        posting_order = self.agents.copy()
        random.shuffle(posting_order)

        for agent in posting_order:
            print(f"   Waiting for {agent.name}...", end="", flush=True)

            post = agent.generate_post(prompt=topic, context=context)

            if post:
                post_record = {
                    "round": round_number,
                    "type": "post",
                    "author": agent.name,
                    "archetype": agent.archetype,
                    "content": post,
                    "topic": topic
                }
                round_posts.append(post_record)
                self.conversation_history.append(post_record)
                self.logger.log_post(round_number, agent.name, agent.archetype, post)
            else:
                self.logger.log_skip(round_number, agent.name)

        print(f"\n   ✓ {len(round_posts)} agents posted")

        # ── PHASE 2: COMMENTING ───────────────────────────────────────
        round_comments = []

        if enable_comments and len(round_posts) > 0:
            print(f"\n💬 PHASE 2: Agents commenting...")

            for agent in self.agents:
                comments_made = 0
                max_comments = 2

                posts_to_review = round_posts.copy()
                random.shuffle(posts_to_review)

                for post_record in posts_to_review:
                    if post_record["author"] == agent.name:
                        continue
                    if comments_made >= max_comments:
                        break

                    print(f"   {agent.name} reviewing {post_record['author']}'s post...", end="", flush=True)

                    comment = agent.generate_comment(
                        original_post=post_record["content"],
                        post_author=post_record["author"]
                    )

                    if comment:
                        comment_record = {
                            "round": round_number,
                            "type": "comment",
                            "commenter": agent.name,
                            "target_author": post_record["author"],
                            "target_post": post_record["content"],
                            "comment": comment
                        }
                        round_comments.append(comment_record)
                        self.conversation_history.append(comment_record)
                        comments_made += 1
                        self.logger.log_comment(
                            round_number,
                            agent.name,
                            post_record["author"],
                            comment
                        )
                    else:
                        print(f" [skipped]")

            print(f"\n   ✓ {len(round_comments)} comments made")

        # ── PHASE 3: UPDATE SIMILARITY MATRIX ────────────────────────
        agent_stats = [agent.get_stats() for agent in self.agents]

        self.similarity_matrix.update(
            round_number=round_number,
            conversation_history=self.conversation_history,
            agent_stats=agent_stats
        )

        # Print top similar pairs
        top_pairs = self.similarity_matrix.get_top_pairs(top_n=3)
        print(f"\n🔗 Top 3 most similar pairs after round {round_number}:")
        for name_i, name_j, score in top_pairs:
            print(f"   {name_i} ↔ {name_j}: {score:.3f}")

        # ── ROUND SUMMARY ─────────────────────────────────────────────
        active_agents = set(
            [p["author"] for p in round_posts] +
            [c["commenter"] for c in round_comments]
        )

        stats = {
            "round": round_number,
            "topic": topic,
            "topic_type": topic_type,
            "total_posts": len(round_posts),
            "total_comments": len(round_comments),
            "active_agents": len(active_agents),
            "active_agent_names": list(active_agents),
            "silent_agents": [
                a.name for a in self.agents
                if a.name not in active_agents
            ]
        }

        self.round_summaries.append(stats)
        self.logger.log_round_end(round_number, stats)

        return {
            "round_number": round_number,
            "topic": topic,
            "posts": round_posts,
            "comments": round_comments,
            "stats": stats,
            "similarity_matrix": self.similarity_matrix.get_matrix_as_dict()
        }

    def run_multiple_rounds(
        self,
        topics: List[Dict[str, str]],
        enable_comments: bool = True
    ) -> List[Dict[str, Any]]:
        """Run multiple rounds and return all results"""

        print(f"\n🚀 Starting {len(topics)}-round discussion in '{self.chat_name}'")

        all_results = []

        for i, topic_data in enumerate(topics, 1):
            print(f"\n\n{'🔄'*30}")
            print(f"STARTING ROUND {i} of {len(topics)}")
            print(f"{'🔄'*30}")

            result = self.run_round(
                topic=topic_data["text"],
                topic_type=topic_data.get("type", "general"),
                enable_comments=enable_comments
            )
            all_results.append(result)

        self.logger.finalize()
        self._print_final_summary()

        return all_results

    def _print_final_summary(self):
        """Print final simulation summary"""
        print(f"\n\n{'='*80}")
        print(f"SIMULATION COMPLETE - {self.chat_name}")
        print(f"{'='*80}")
        print(f"Total rounds: {self.current_round}")
        print(f"Total messages: {len(self.conversation_history)}")

        print(f"\nPer-Agent Activity:")
        for agent in self.agents:
            stats = agent.get_stats()
            total = stats['total_posts'] + stats['total_comments']
            bar = "█" * total
            print(f"  {agent.name:<35} Posts: {stats['total_posts']:>2}  "
                  f"Comments: {stats['total_comments']:>2}  {bar}")

        print(f"\n🔗 Final Top 5 Most Similar Pairs:")
        for name_i, name_j, score in self.similarity_matrix.get_top_pairs(5):
            print(f"   {name_i} ↔ {name_j}: {score:.3f}")

    def get_all_interactions(self) -> List[Dict[str, Any]]:
        return self.conversation_history

    def get_similarity_matrix(self) -> SimilarityMatrix:
        return self.similarity_matrix
    
    def get_agents_by_names(self, agent_names: List[str]) -> List[SocialAgent]:
        """Return agent objects for a given list of agent names"""
        return [agent for agent in self.agents if agent.name in agent_names]