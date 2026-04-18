"""
Main Simulation Script - 5 AGENTS
Smaller scale for faster iteration and testing
"""

import yaml
from pathlib import Path
from datetime import datetime

try:
    from agents.persona_loader import create_agents
    from groupchat.group_chat import GroupChat
    from utils.logger import ConversationLogger
    from clustering.coalition_detector import CoalitionDetector
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from agents.persona_loader import create_agents
    from groupchat.group_chat import GroupChat
    from utils.logger import ConversationLogger
    from clustering.coalition_detector import CoalitionDetector


def main():
    """Run 5-agent simulation with Option B coalition formation"""
    
    print("\n" + "🚀 "*20)
    print("5-AGENT SOCIAL NETWORK SIMULATION")
    print("🚀 "*20)
    
    # Set data directory for this scale
    data_dir = Path("data/run_5_agents")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # ── STEP 1: LOAD AGENTS ─────────────────────────────────────────
    print("\n📥 Loading 5 agent personas...")
    config_path = Path(__file__).parent.parent / "config" / "agents_5.yaml"
    agents = create_agents(config_path=str(config_path))
    print(f"   ✅ Loaded {len(agents)} agents")
    
    # ── STEP 2: LOAD TOPICS ─────────────────────────────────────────
    print("\n📥 Loading discussion topics...")
    topics_path = Path(__file__).parent.parent / "config" / "topics.yaml"
    with open(topics_path, 'r') as f:
        topics_data = yaml.safe_load(f)
        topics = topics_data['topics']
    print(f"   ✅ Loaded {len(topics)} topics")
    
    # ── STEP 3: CREATE INITIAL GROUPCHAT ────────────────────────────
    logger = ConversationLogger(base_dir=str(data_dir / "conversations"))
    main_chat = GroupChat(
        agents=agents,
        chat_name="5-Agent Main Community",
        logger=logger
    )
    
    # ── STEP 4: ROUNDS 1-2 (MIXED COMMUNITY) ────────────────────────
    print("\n" + "="*80)
    print("PHASE 1: MIXED COMMUNITY (Rounds 1-2)")
    print("="*80)
    
    mixed_results = []
    for i in range(2):
        result = main_chat.run_round(
            topic=topics[i]["text"],
            topic_type=topics[i].get("type", "general"),
            enable_comments=True
        )
        mixed_results.append(result)
        
        # Save similarity matrix
        sim_matrix = main_chat.get_similarity_matrix()
        sim_matrix.save_to_file(
            round_number=result['round_number'],
            output_dir=str(data_dir / "similarity")
        )
    
    # ── STEP 5: COALITION FORMATION (After Round 2) ─────────────────
    print("\n" + "="*80)
    print("🔍 CLUSTERING EVENT 1: FORMING COALITIONS")
    print("="*80)
    
    similarity_matrix = main_chat.get_similarity_matrix()
    detector = CoalitionDetector(
        similarity_matrix=similarity_matrix,
        agent_names=[a.name for a in agents]
    )
    
    coalitions_v1 = detector.detect_coalitions(
        method='threshold',
        threshold=0.50  # Lower threshold for smaller group
    )
    
    detector.save_coalitions(
        coalitions=coalitions_v1,
        round_number=2,
        output_dir=str(data_dir / "coalitions")
    )
    
    print(f"\n✅ Formed {len(coalitions_v1)} coalitions")
    for i, coalition in enumerate(coalitions_v1, 1):
        print(f"   Coalition {i}: {coalition}")
    
    # ── STEP 6: ROUNDS 3-4 (COALITION DISCUSSIONS) ──────────────────
    print("\n" + "="*80)
    print("PHASE 2: COALITION DISCUSSIONS (Rounds 3-4)")
    print("="*80)
    
    coalition_chats = []
    for i, coalition_members in enumerate(coalitions_v1, 1):
        coalition_agents = main_chat.get_agents_by_names(coalition_members)
        
        coalition_chat = GroupChat(
            agents=coalition_agents,
            chat_name=f"5-Agent Coalition {i}",
            logger=logger
        )
        coalition_chats.append(coalition_chat)
    
    # Rounds 3-4 in coalitions
    for round_idx in range(2, 4):
        print(f"\n{'─'*80}")
        print(f"ROUND {round_idx + 1}")
        print(f"{'─'*80}")
        
        for chat in coalition_chats:
            result = chat.run_round(
                topic=topics[round_idx]["text"],
                topic_type=topics[round_idx].get("type", "general"),
                enable_comments=True
            )
            
            # Save similarity matrix for this coalition
            sim_matrix = chat.get_similarity_matrix()
            sim_matrix.save_to_file(
                round_number=result['round_number'],
                output_dir=str(data_dir / "similarity")
            )
    
    # ── STEP 7: RE-CLUSTERING (After Round 4) ───────────────────────
    print("\n" + "="*80)
    print("🔍 CLUSTERING EVENT 2: RE-FORMING COALITIONS")
    print("="*80)
    
    # Merge similarity matrices from all coalitions
    merged_matrix = coalition_chats[0].get_similarity_matrix()
    
    detector = CoalitionDetector(
        similarity_matrix=merged_matrix,
        agent_names=[a.name for a in agents]
    )
    
    coalitions_v2 = detector.detect_coalitions(
        method='threshold',
        threshold=0.50
    )
    
    detector.save_coalitions(
        coalitions=coalitions_v2,
        round_number=4,
        output_dir=str(data_dir / "coalitions")
    )
    
    # Analyze stability
    stability = detector.analyze_stability(coalitions_v1, coalitions_v2)
    print(f"\n📊 Coalition Stability Analysis:")
    print(f"   Churn Rate: {stability['churn_rate']*100:.1f}%")
    print(f"   Agents Migrated: {stability['agents_migrated']}")
    
    # ── STEP 8: ROUND 5 (NEW COALITIONS) ────────────────────────────
    print("\n" + "="*80)
    print("PHASE 3: FINAL ROUND (Round 5)")
    print("="*80)
    
    new_coalition_chats = []
    for i, coalition_members in enumerate(coalitions_v2, 1):
        coalition_agents = main_chat.get_agents_by_names(coalition_members)
        
        coalition_chat = GroupChat(
            agents=coalition_agents,
            chat_name=f"5-Agent Coalition {i} (Reformed)",
            logger=logger
        )
        new_coalition_chats.append(coalition_chat)
    
    for chat in new_coalition_chats:
        result = chat.run_round(
            topic=topics[4]["text"],
            topic_type=topics[4].get("type", "general"),
            enable_comments=True
        )
        
        sim_matrix = chat.get_similarity_matrix()
        sim_matrix.save_to_file(
            round_number=5,
            output_dir=str(data_dir / "similarity")
        )
    
    # ── FINALIZE ─────────────────────────────────────────────────────
    logger.finalize()
    
    print("\n" + "="*80)
    print("✅ 5-AGENT SIMULATION COMPLETE")
    print("="*80)
    print(f"\n📁 Results saved to: {data_dir}/")
    print("\n")


if __name__ == "__main__":
    main()