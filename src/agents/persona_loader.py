import yaml
from pathlib import Path
from typing import List, Dict, Any
from .social_agent import SocialAgent


def load_agent_personas(config_path: str = "config/agents.yaml") -> Dict[str, Any]:
    """
    Load agent persona definitions from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing agent configurations
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def create_agents(
    config_path: str = "config/agents.yaml",
    model_name: str = "llama3.1:latest",
    temperature: float = 0.8
) -> List[SocialAgent]:
    """
    Create SocialAgent instances from configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        model_name: Ollama model to use for all agents
        temperature: LLM temperature (higher = more creative/random)
        
    Returns:
        List of initialized SocialAgent objects
    """
    config = load_agent_personas(config_path)
    agents = []
    
    print(f"Creating agents using model: {model_name}")
    print("=" * 60)
    
    for agent_config in config['agents']:
        agent = SocialAgent(
            name=agent_config['name'],
            archetype=agent_config['archetype'],
            persona=agent_config['persona'],
            model_name=model_name,
            temperature=temperature
        )
        agents.append(agent)
        print(f"✓ Created: {agent.name} ({agent.archetype})")
    
    print("=" * 60)
    print(f"Total agents created: {len(agents)}\n")
    
    return agents


def display_agent_summary(agents: List[SocialAgent]):
    """Display a summary of all agents and their personas"""
    print("\n" + "=" * 80)
    print("AGENT ROSTER SUMMARY")
    print("=" * 80)
    
    for i, agent in enumerate(agents, 1):
        print(f"\n{i}. {agent.name}")
        print(f"   Archetype: {agent.archetype}")
        print(f"   Role: {agent.persona.role}")
        print(f"   Personality: {', '.join(agent.persona.personality_traits)}")
        print(f"   Posting Style: {agent.persona.posting_style}")
        print(f"   Key Topics: {', '.join(agent.persona.topics[:3])}")
        print(f"   Example: \"{agent.persona.example_posts[0][:80]}...\"")
    
    print("\n" + "=" * 80)