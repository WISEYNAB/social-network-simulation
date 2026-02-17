"""
Conversation Logger
Saves all groupchat interactions to files for later analysis
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


class ConversationLogger:
    """
    Logs all agent interactions to JSON and text files.
    Creates a new log file per simulation run.
    """

    def __init__(self, log_dir: str = "data/conversations"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create unique session ID based on timestamp
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Storage for full session data
        self.session_data = {
            "session_id": self.session_id,
            "started_at": datetime.now().isoformat(),
            "rounds": []
        }

        print(f"📁 Logging session to: {self.session_dir}")

    def log_round_start(self, round_number: int, topic: str, topic_type: str):
        """Log the start of a new discussion round"""
        round_data = {
            "round_number": round_number,
            "topic": topic,
            "topic_type": topic_type,
            "started_at": datetime.now().isoformat(),
            "posts": [],
            "comments": []
        }
        self.session_data["rounds"].append(round_data)
        print(f"\n{'='*60}")
        print(f"📢 ROUND {round_number} | Topic: {topic}")
        print(f"{'='*60}")

    def log_post(
        self,
        round_number: int,
        agent_name: str,
        archetype: str,
        content: str
    ):
        """Log an agent's post"""
        post_data = {
            "agent": agent_name,
            "archetype": archetype,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        # Add to current round
        current_round = self._get_round(round_number)
        if current_round:
            current_round["posts"].append(post_data)

        # Print to console
        print(f"\n🗣️  {agent_name} ({archetype}):")
        print(f"   {content}")

    def log_skip(self, round_number: int, agent_name: str):
        """Log when an agent skips posting"""
        current_round = self._get_round(round_number)
        if current_round:
            current_round["posts"].append({
                "agent": agent_name,
                "content": None,
                "skipped": True,
                "timestamp": datetime.now().isoformat()
            })
        print(f"\n   [{agent_name} chose not to post]")

    def log_comment(
        self,
        round_number: int,
        commenter_name: str,
        target_name: str,
        comment: str
    ):
        """Log a comment/reply between agents"""
        comment_data = {
            "commenter": commenter_name,
            "target": target_name,
            "comment": comment,
            "timestamp": datetime.now().isoformat()
        }

        current_round = self._get_round(round_number)
        if current_round:
            current_round["comments"].append(comment_data)

        # Print to console
        print(f"\n   ↳ {commenter_name} → {target_name}:")
        print(f"     {comment}")

    def log_round_end(self, round_number: int, stats: Dict[str, Any]):
        """Log the end of a round with statistics"""
        current_round = self._get_round(round_number)
        if current_round:
            current_round["ended_at"] = datetime.now().isoformat()
            current_round["stats"] = stats

        print(f"\n📊 Round {round_number} Stats:")
        print(f"   Posts: {stats.get('total_posts', 0)}")
        print(f"   Comments: {stats.get('total_comments', 0)}")
        print(f"   Active agents: {stats.get('active_agents', 0)}")

        # Save after each round
        self._save_session()

    def _get_round(self, round_number: int) -> Dict:
        """Get round data by round number"""
        for round_data in self.session_data["rounds"]:
            if round_data["round_number"] == round_number:
                return round_data
        return None

    def _save_session(self):
        """Save full session data to JSON"""
        json_path = self.session_dir / "session.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.session_data, f, indent=2, ensure_ascii=False)

    def save_readable_transcript(self):
        """Save a human-readable text transcript"""
        transcript_path = self.session_dir / "transcript.txt"

        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SOCIAL NETWORK SIMULATION - CONVERSATION TRANSCRIPT\n")
            f.write(f"Session: {self.session_id}\n")
            f.write(f"Started: {self.session_data['started_at']}\n")
            f.write("=" * 80 + "\n\n")

            for round_data in self.session_data["rounds"]:
                f.write(f"\nROUND {round_data['round_number']}\n")
                f.write(f"Topic: {round_data['topic']}\n")
                f.write("-" * 40 + "\n\n")

                f.write("POSTS:\n")
                for post in round_data["posts"]:
                    if post.get("skipped"):
                        f.write(f"  [{post['agent']} - no post]\n")
                    else:
                        f.write(f"  {post['agent']}: {post['content']}\n\n")

                if round_data["comments"]:
                    f.write("\nCOMMENTS:\n")
                    for comment in round_data["comments"]:
                        f.write(f"  {comment['commenter']} → {comment['target']}:\n")
                        f.write(f"  {comment['comment']}\n\n")

                f.write("\n")

        print(f"\n💾 Transcript saved to: {transcript_path}")
        return transcript_path

    def finalize(self):
        """Finalize session - save everything"""
        self.session_data["ended_at"] = datetime.now().isoformat()
        self._save_session()
        transcript_path = self.save_readable_transcript()
        print(f"\n✅ Session complete! Files saved to: {self.session_dir}")
        return self.session_dir