"""
Social Network Simulation - Main Entry Point
Step 2: GroupChat with multi-round discussions
"""

import sys
import yaml
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
src_dir = Path(__file__).parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

from agents.persona_loader import create_agents, display_agent_summary
from groupchat.group_chat import GroupChat
from utils.logger import ConversationLogger


def load_topics(topics_path: str) -> list:
    """Load discussion topics from YAML file"""
    with open(topics_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Convert YAML format to list format
    topics = []
    for key, value in data['topics'].items():
        topics.append({
            "text": value["text"],
            "type": value["type"]
        })
    return topics


def main():
    """Main simulation runner"""

    print("\n" + "🚀 " * 20)
    print("SOCIAL NETWORK SIMULATION - STEP 2: GROUP DISCUSSIONS")
    print("🚀 " * 20 + "\n")

    # ── LOAD CONFIG ───────────────────────────────────────────────────
    config_path  = project_root / "config" / "agents.yaml"
    topics_path  = project_root / "config" / "topics.yaml"
    data_dir     = project_root / "data" / "conversations"

    # Verify files exist
    for path in [config_path, topics_path]:
        if not path.exists():
            print(f"❌ File not found: {path}")
            return

    # ── CREATE AGENTS ─────────────────────────────────────────────────
    agents = create_agents(
        config_path=str(config_path),
        model_name="llama3.1:latest",
        temperature=0.8
    )

    display_agent_summary(agents)

    # ── LOAD TOPICS ───────────────────────────────────────────────────
    topics = load_topics(str(topics_path))
    print(f"\n📋 Loaded {len(topics)} discussion topics")
    for i, t in enumerate(topics, 1):
        print(f"   Round {i}: [{t['type'].upper()}] {t['text'][:60]}...")

    # ── SETUP LOGGER ──────────────────────────────────────────────────
    logger = ConversationLogger(log_dir=str(data_dir))

    # ── CREATE GROUPCHAT ──────────────────────────────────────────────
    groupchat = GroupChat(
        agents=agents,
        chat_name="Initial Mixed Community",
        logger=logger
    )

    # ── RUN SIMULATION ────────────────────────────────────────────────
    print("\n" + "💬 " * 20)
    print("STARTING GROUP DISCUSSION SIMULATION")
    print("💬 " * 20)

    results = groupchat.run_multiple_rounds(
        topics=topics,
        enable_comments=True
    )

    # ── PRINT INTERACTION DATA ────────────────────────────────────────
    print("\n" + "📊 " * 20)
    print("FINAL AGENT STATISTICS")
    print("📊 " * 20 + "\n")

    for agent in agents:
        stats = agent.get_stats()
        print(f"\n{stats['name']}:")
        print(f"  Posts      : {stats['total_posts']}")
        print(f"  Comments   : {stats['total_comments']}")
        print(f"  Agreements : {stats['agreements']}")
        print(f"  Disagreements: {stats['disagreements']}")
        if stats['topics_mentioned']:
            print(f"  Topics     : {stats['topics_mentioned']}")

    print("\n✅ Step 2 Complete!")
    print("📁 Check data/conversations/ for saved logs")
    print("🔜 Next step: Similarity matrix calculation")


if __name__ == "__main__":
    main()