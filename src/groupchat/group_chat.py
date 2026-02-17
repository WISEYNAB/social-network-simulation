"""
GroupChat - Manages multi-agent discussions
Handles turn-based rounds where all agents interact in a shared space
"""

import random
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from agents.social_agent import SocialAgent
    from utils.logger import ConversationLogger
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agents.social_agent import SocialAgent
    from utils.logger import ConversationLogger


class GroupChat:
    """
    Simulates a social media group discussion.

    Each round:
    1. All agents see the discussion topic
    2. Each agent decides whether to post (based on their persona)
    3. Agents see each other's posts and can comment
    4. All interactions are logged for similarity analysis
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

        # Conversation history - all posts from all rounds
        self.conversation_history: List[Dict[str, Any]] = []

        # Track current round
        self.current_round = 0

        # Store round summaries for analysis
        self.round_summaries: List[Dict[str, Any]] = []

        print(f"\n🏘️  GroupChat '{self.chat_name}' initialized")
        print(f"   Members: {', '.join([a.name for a in self.agents])}")

    def run_round(
        self,
        topic: str,
        topic_type: str = "general",
        enable_comments: bool = True
    ) -> Dict[str, Any]:
        """
        Run one full discussion round.

        Args:
            topic: The discussion prompt for this round
            topic_type: Category of topic (technical, creative, controversial etc.)
            enable_comments: Whether agents comment on each other's posts

        Returns:
            Dictionary with round results and statistics
        """

        self.current_round += 1
        round_number = self.current_round

        # Log round start
        self.logger.log_round_start(round_number, topic, topic_type)

        # ── PHASE 1: POSTING ──────────────────────────────────────────
        # Each agent sees the topic + recent conversation history
        # and decides whether to post

        print(f"\n📝 PHASE 1: Agents posting...")

        round_posts = []

        # Build context from recent conversation history (last 10 messages)
        context = self.conversation_history[-10:] if self.conversation_history else []

        # Shuffle agent order to avoid always having same agent go first
        posting_order = self.agents.copy()
        random.shuffle(posting_order)

        for agent in posting_order:
            print(f"   Waiting for {agent.name}...", end="", flush=True)

            post = agent.generate_post(
                prompt=topic,
                context=context
            )

            if post:
                # Record the post
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

                # Log it
                self.logger.log_post(
                    round_number,
                    agent.name,
                    agent.archetype,
                    post
                )
            else:
                self.logger.log_skip(round_number, agent.name)

        print(f"\n   ✓ {len(round_posts)} agents posted")

        # ── PHASE 2: COMMENTING ───────────────────────────────────────
        # Agents read each other's posts and can comment/react

        round_comments = []

        if enable_comments and len(round_posts) > 0:
            print(f"\n💬 PHASE 2: Agents commenting...")

            for agent in self.agents:
                # Each agent can comment on posts from other agents
                # Limit to 2 comments per agent per round to avoid spam
                comments_made = 0
                max_comments = 2

                # Shuffle posts so agents don't always comment on same ones
                posts_to_review = round_posts.copy()
                random.shuffle(posts_to_review)

                for post_record in posts_to_review:
                    # Don't comment on own post
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

        # ── ROUND SUMMARY ─────────────────────────────────────────────

        # Count active agents
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
            "stats": stats
        }

    def run_multiple_rounds(
        self,
        topics: List[Dict[str, str]],
        enable_comments: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run multiple discussion rounds with different topics.

        Args:
            topics: List of topic dicts with 'text' and 'type' keys
            enable_comments: Whether to enable agent commenting

        Returns:
            List of round results
        """

        print(f"\n🚀 Starting {len(topics)}-round discussion in '{self.chat_name}'")
        print(f"   Agents: {len(self.agents)}")
        print(f"   Topics: {len(topics)}")

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

        # Finalize logging
        self.logger.finalize()

        # Print final summary
        self._print_final_summary()

        return all_results

    def _print_final_summary(self):
        """Print a summary of all rounds"""
        print(f"\n\n{'='*80}")
        print(f"SIMULATION COMPLETE - {self.chat_name}")
        print(f"{'='*80}")
        print(f"Total rounds: {self.current_round}")
        print(f"Total messages: {len(self.conversation_history)}")

        print(f"\nPer-Agent Activity:")
        for agent in self.agents:
            stats = agent.get_stats()
            total_activity = stats['total_posts'] + stats['total_comments']
            bar = "█" * total_activity
            print(f"  {agent.name:<35} Posts: {stats['total_posts']:>2}  Comments: {stats['total_comments']:>2}  {bar}")

        print(f"\nRound-by-Round Summary:")
        for summary in self.round_summaries:
            print(f"  Round {summary['round']}: {summary['total_posts']} posts, "
                  f"{summary['total_comments']} comments, "
                  f"{summary['active_agents']} active agents")
            if summary['silent_agents']:
                print(f"    Silent: {', '.join(summary['silent_agents'])}")

    def get_all_interactions(self) -> List[Dict[str, Any]]:
        """Return full conversation history for similarity analysis"""
        return self.conversation_history

    def get_agent_posts(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get all posts by a specific agent"""
        return [
            msg for msg in self.conversation_history
            if msg.get("author") == agent_name or msg.get("commenter") == agent_name
        ]