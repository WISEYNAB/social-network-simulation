"""
Social Network Simulation - Main Entry Point
"""

import sys
from pathlib import Path

# Get the project root directory (parent of src)
project_root = Path(__file__).parent.parent
src_dir = Path(__file__).parent

# Add to Python path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

from agents.persona_loader import create_agents, display_agent_summary
from agents.social_agent import SocialAgent


def test_single_agent(agent: SocialAgent, prompt: str):
    """Test a single agent's posting capability"""
    print(f"\n{'='*80}")
    print(f"Testing: {agent.name}")
    print(f"{'='*80}")
    print(f"Prompt: {prompt}\n")
    
    post = agent.generate_post(prompt)
    
    if post:
        print(f"{agent.name}: {post}")
    else:
        print(f"{agent.name}: [chose not to post]")
    
    print(f"{'='*80}\n")


def test_interaction(agent1: SocialAgent, agent2: SocialAgent, prompt: str):
    """Test two agents interacting"""
    print(f"\n{'='*80}")
    print(f"Testing Interaction: {agent1.name} and {agent2.name}")
    print(f"{'='*80}")
    print(f"Prompt: {prompt}\n")
    
    # Agent 1 posts
    post1 = agent1.generate_post(prompt)
    if post1:
        print(f"{agent1.name}: {post1}\n")
        
        # Agent 2 comments
        comment = agent2.generate_comment(post1, agent1.name)
        if comment:
            print(f"  ↳ {agent2.name}: {comment}")
        else:
            print(f"  ↳ {agent2.name}: [chose not to engage]")
    else:
        print(f"{agent1.name}: [chose not to post]")
    
    print(f"{'='*80}\n")


def main():
    """Main function to initialize and test agents"""
    
    print("\n" + "🚀 " * 20)
    print("SOCIAL NETWORK SIMULATION - AGENT INITIALIZATION")
    print("🚀 " * 20 + "\n")
    
    # FIXED: Build absolute path to config file
    config_path = project_root / "config" / "agents.yaml"
    
    print(f"Looking for config at: {config_path}")
    print(f"Config exists: {config_path.exists()}\n")
    
    if not config_path.exists():
        print("❌ ERROR: Config file not found!")
        print(f"Expected location: {config_path}")
        print("\nPlease create config/agents.yaml in your project root.")
        return
    
    # Create all agents
    agents = create_agents(
        config_path=str(config_path),
        model_name="llama3.1:latest",
        temperature=0.8
    )
    
    # Display agent summary
    display_agent_summary(agents)
    
    # Test prompts
    test_prompts = [
        "What's your take on the future of AI?",
        "Share something you're passionate about",
        "What's the biggest challenge in your field right now?"
    ]
    
    # Test individual agents
    print("\n" + "🧪 " * 20)
    print("TESTING INDIVIDUAL AGENT POSTS")
    print("🧪 " * 20)
    
    # Test a few different personas
    test_agents = [agents[0], agents[1], agents[3], agents[5]]
    
    for agent in test_agents:
        test_single_agent(agent, test_prompts[0])
    
    # Test interactions
    print("\n" + "💬 " * 20)
    print("TESTING AGENT INTERACTIONS")
    print("💬 " * 20)
    
    # Test some interesting pairings
    test_interaction(agents[0], agents[5], test_prompts[1])
    test_interaction(agents[1], agents[4], test_prompts[2])
    
    # Display stats
    print("\n" + "📊 " * 20)
    print("AGENT STATISTICS")
    print("📊 " * 20 + "\n")
    
    for agent in agents:
        stats = agent.get_stats()
        print(f"{stats['name']}: {stats['total_posts']} posts, {stats['total_comments']} comments")


if __name__ == "__main__":
    main()