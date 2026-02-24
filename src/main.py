"""
Social Network Simulation - Main Entry Point
Step 4: Coalition Formation with Dynamic Regrouping (Option B)

Timeline:
- Rounds 1-2: All agents in main community
- After Round 2: CLUSTER → Form coalitions v1
- Round 3: Coalitions discuss separately
- Round 4: Coalitions discuss
- After Round 4: RE-CLUSTER → Form coalitions v2
- Round 5: Final round in new coalitions
"""

import sys
import yaml
from pathlib import Path

project_root = Path(__file__).parent.parent
src_dir = Path(__file__).parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

from agents.persona_loader import create_agents, display_agent_summary
from agents.social_agent import SocialAgent
from groupchat.group_chat import GroupChat
from utils.logger import ConversationLogger
from clustering.coalition_detector import CoalitionDetector
from typing import List


def load_topics(topics_path: str) -> list:
    """Load discussion topics from YAML file"""
    with open(topics_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    topics = []
    for key, value in data['topics'].items():
        topics.append({
            "text": value["text"],
            "type": value["type"]
        })
    return topics


def run_round_in_coalitions(
    coalitions: List[List[str]],
    all_agents: List[SocialAgent],
    topic: dict,
    round_number: int,
    logger: ConversationLogger
) -> List[dict]:
    """
    Run a single round across multiple coalitions simultaneously.
    Each coalition has its own discussion.
    """

    print(f"\n{'🔄'*30}")
    print(f"ROUND {round_number}: Multiple Coalition Discussions")
    print(f"{'🔄'*30}")
    print(f"Topic: {topic['text']}")
    print(f"Number of coalitions: {len(coalitions)}\n")

    agent_lookup = {agent.name: agent for agent in all_agents}
    all_results = []

    for coalition_idx, coalition_members in enumerate(coalitions, 1):
        print(f"\n{'─'*60}")
        print(f"Coalition {coalition_idx} Discussion ({len(coalition_members)} members)")
        print(f"Members: {', '.join(coalition_members)}")
        print(f"{'─'*60}")

        # Get agent objects for this coalition
        coalition_agents = [agent_lookup[name] for name in coalition_members]

        # Create a temporary groupchat for this coalition
        coalition_chat = GroupChat(
            agents=coalition_agents,
            chat_name=f"Coalition {coalition_idx}",
            logger=logger
        )

        # Run the round for this coalition
        result = coalition_chat.run_round(
            topic=topic["text"],
            topic_type=topic.get("type", "general"),
            enable_comments=True
        )

        result["coalition_id"] = coalition_idx
        result["coalition_members"] = coalition_members
        all_results.append(result)

    return all_results


def main():
    """Main simulation with Option B: Cluster after rounds 2 and 4"""

    print("\n" + "🚀 " * 20)
    print("SOCIAL NETWORK SIMULATION - OPTION B: DYNAMIC COALITIONS")
    print("🚀 " * 20 + "\n")

    # ── LOAD CONFIG ───────────────────────────────────────────────────
    config_path = project_root / "config" / "agents.yaml"
    topics_path = project_root / "config" / "topics.yaml"

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

    # ── SETUP LOGGER AND COALITION DETECTOR ───────────────────────────
    logger = ConversationLogger(log_dir=str(project_root / "data" / "conversations"))
    coalition_detector = CoalitionDetector(
        agent_names=[a.name for a in agents],
        min_coalition_size=2,
        max_coalitions=4
    )

    # ── PHASE 1: ROUNDS 1-2 IN MAIN COMMUNITY ─────────────────────────
    print("\n" + "="*80)
    print("PHASE 1: Initial Mixed Community (Rounds 1-2)")
    print("="*80)

    main_groupchat = GroupChat(
        agents=agents,
        chat_name="Initial Mixed Community",
        logger=logger
    )

    # Run rounds 1 and 2
    for round_num in [1, 2]:
        print(f"\n{'🔄'*30}")
        print(f"STARTING ROUND {round_num} of 5")
        print(f"{'🔄'*30}")

        main_groupchat.run_round(
            topic=topics[round_num - 1]["text"],
            topic_type=topics[round_num - 1].get("type", "general"),
            enable_comments=True
        )

    # ── CLUSTERING EVENT 1: After Round 2 ─────────────────────────────
    print("\n" + "🔍"*40)
    print("CLUSTERING EVENT 1: Forming Initial Coalitions")
    print("🔍"*40)

    similarity_matrix_v1 = main_groupchat.get_similarity_matrix().matrix

    coalitions_v1 = coalition_detector.detect_coalitions(
        similarity_matrix=similarity_matrix_v1,
        round_number=2,
        method="threshold",  # Try: "threshold", "louvain", "hierarchical"
        threshold=0.55,
        n_clusters=3
    )

    # ── PHASE 2: ROUND 3 IN SEPARATE COALITIONS ───────────────────────
    print("\n" + "="*80)
    print("PHASE 2: Coalition-Based Discussions (Round 3)")
    print("="*80)

    run_round_in_coalitions(
        coalitions=coalitions_v1,
        all_agents=agents,
        topic=topics[2],  # Round 3 topic
        round_number=3,
        logger=logger
    )

    # Update main groupchat's conversation history with coalition discussions
    # (This ensures similarity matrix has all data)
    # We'll merge all coalition conversations into the main history
    for agent in agents:
        main_groupchat.conversation_history.extend(agent.post_history)
        main_groupchat.conversation_history.extend(agent.interaction_history)

    # Recompute similarity after round 3
    main_groupchat.current_round = 3
    agent_stats = [agent.get_stats() for agent in agents]
    main_groupchat.similarity_matrix.update(
        round_number=3,
        conversation_history=main_groupchat.conversation_history,
        agent_stats=agent_stats
    )

    # ── PHASE 3: ROUND 4 IN COALITIONS ────────────────────────────────
    print("\n" + "="*80)
    print("PHASE 3: Continued Coalition Discussions (Round 4)")
    print("="*80)

    run_round_in_coalitions(
        coalitions=coalitions_v1,
        all_agents=agents,
        topic=topics[3],  # Round 4 topic
        round_number=4,
        logger=logger
    )

    # Update similarity after round 4
    main_groupchat.current_round = 4
    agent_stats = [agent.get_stats() for agent in agents]
    main_groupchat.similarity_matrix.update(
        round_number=4,
        conversation_history=main_groupchat.conversation_history,
        agent_stats=agent_stats
    )

    # ── CLUSTERING EVENT 2: After Round 4 ─────────────────────────────
    print("\n" + "🔍"*40)
    print("CLUSTERING EVENT 2: Re-forming Coalitions")
    print("🔍"*40)

    similarity_matrix_v2 = main_groupchat.get_similarity_matrix().matrix

    coalitions_v2 = coalition_detector.detect_coalitions(
        similarity_matrix=similarity_matrix_v2,
        round_number=4,
        method="threshold",
        threshold=0.55,
        n_clusters=3
    )

    # Analyze stability
    stability_metrics = coalition_detector.analyze_stability()
    print(f"\n📊 Coalition Stability Analysis:")
    print(f"   Churn rate: {stability_metrics['churn_rate']*100:.1f}%")
    print(f"   Agents migrated: {stability_metrics['agents_migrated']}/{stability_metrics['total_agents']}")
    if stability_metrics['migrations']:
        print(f"   Migrations:")
        for migration in stability_metrics['migrations']:
            print(f"      • {migration['agent']}: Coalition {migration['from_coalition']+1} → {migration['to_coalition']+1}")

    # ── PHASE 4: ROUND 5 IN NEW COALITIONS ────────────────────────────
    print("\n" + "="*80)
    print("PHASE 4: Final Round in Reformed Coalitions (Round 5)")
    print("="*80)

    run_round_in_coalitions(
        coalitions=coalitions_v2,
        all_agents=agents,
        topic=topics[4],  # Round 5 topic
        round_number=5,
        logger=logger
    )

    # Final similarity update
    main_groupchat.current_round = 5
    agent_stats = [agent.get_stats() for agent in agents]
    main_groupchat.similarity_matrix.update(
        round_number=5,
        conversation_history=main_groupchat.conversation_history,
        agent_stats=agent_stats
    )

    # ── FINAL SUMMARY ──────────────────────────────────────────────────
    logger.finalize()

    print("\n\n" + "="*80)
    print("SIMULATION COMPLETE - OPTION B: DYNAMIC COALITIONS")
    print("="*80)

    print(f"\n📊 Final Agent Statistics:")
    for agent in agents:
        stats = agent.get_stats()
        print(f"\n{stats['name']}:")
        print(f"  Posts: {stats['total_posts']}")
        print(f"  Comments: {stats['total_comments']}")
        print(f"  Agreements: {stats['agreements']}")
        print(f"  Disagreements: {stats['disagreements']}")

    print(f"\n🔗 Final Top 5 Most Similar Pairs:")
    for name_i, name_j, score in main_groupchat.similarity_matrix.get_top_pairs(5):
        print(f"   {name_i} ↔ {name_j}: {score:.3f}")

    print("\n✅ All phases complete!")
    print(f"📁 Data saved to:")
    print(f"   - Conversations: data/conversations/")
    print(f"   - Similarity matrices: data/similarity/")
    print(f"   - Coalitions: data/coalitions/")
    print("\n🔜 Next step: Visualization and analysis!")


if __name__ == "__main__":
    main()