from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import random


class AgentPersona(BaseModel):
    """Structured persona data for an agent"""
    role: str
    personality_traits: List[str]
    posting_style: str
    tone: str
    topics: List[str]
    social_behavior: str
    engagement_pattern: str
    emoji_usage: str
    example_posts: List[str]


class SocialAgent:
    """
    An LLM-based social media agent with a distinct persona.
    
    This agent can:
    - Generate posts based on discussion prompts
    - Comment on other agents' posts
    - Track interaction history
    - Maintain consistent personality
    """
    
    def __init__(
        self,
        name: str,
        archetype: str,
        persona: Dict[str, Any],
        model_name: str = "llama3.1:latest",
        temperature: float = 0.8
    ):
        self.name = name
        self.archetype = archetype
        self.persona = AgentPersona(**persona)
        
        # Initialize LLM
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
        )
        
        # Tracking
        self.post_history: List[Dict[str, Any]] = []
        self.interaction_history: List[Dict[str, Any]] = []
        self.topics_mentioned: Dict[str, int] = {}
        self.sentiment_scores: List[float] = []
        self.current_community: Optional[int] = None
        
        # Behavioral stats
        self.total_posts = 0
        self.total_comments = 0
        self.agreements = 0
        self.disagreements = 0
        
    def _build_system_prompt(self) -> str:
        """Construct the system prompt that defines the agent's persona"""
        
        example_posts_text = "\n".join([f"- {post}" for post in self.persona.example_posts])
        
        system_prompt = f"""You are {self.name}, a {self.archetype}.

YOUR CORE IDENTITY:
Role: {self.persona.role}
Personality: {', '.join(self.persona.personality_traits)}

HOW YOU COMMUNICATE:
Posting Style: {self.persona.posting_style}
Tone: {self.persona.tone}
Emoji Usage: {self.persona.emoji_usage}

YOUR INTERESTS:
{', '.join(self.persona.topics)}

YOUR SOCIAL BEHAVIOR:
{self.persona.social_behavior}

EXAMPLES OF HOW YOU POST:
{example_posts_text}

CRITICAL INSTRUCTIONS:
- Stay completely in character as {self.name}
- Match the tone, style, and personality described above
- Your responses should feel authentic to this persona
- Keep posts natural and conversational, like real social media
- Don't break character or mention that you're an AI
- Use emojis as specified in your persona (not more, not less)
- Your posting frequency: {self.persona.engagement_pattern}
"""
        return system_prompt
    
    def generate_post(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 200
    ) -> str:
        """
        Generate a social media post in response to a discussion prompt.
        
        Args:
            prompt: The discussion topic or question
            context: Recent posts and comments from other agents for context
            max_tokens: Maximum length of the post
            
        Returns:
            The generated post as a string
        """
        
        # Decide if this agent should post (based on engagement pattern)
        if not self._should_post():
            return None
        
        # Build context string — handles both posts AND comments
        context_str = ""
        if context:
            context_str = "\n\nRECENT COMMUNITY POSTS:\n"
            for msg in context[-5:]:
                msg_type = msg.get("type", "post")

                if msg_type == "post":
                    author  = msg.get("author", "Someone")
                    content = msg.get("content", "")
                    if content:
                        context_str += f"- {author}: {content}\n"

                elif msg_type == "comment":
                    commenter = msg.get("commenter", "Someone")
                    target    = msg.get("target_author", "")
                    comment   = msg.get("comment", "")
                    if comment:
                        context_str += f"- {commenter} (replying to {target}): {comment}\n"
        
        system_prompt = self._build_system_prompt()
        
        user_prompt = f"""The community is discussing: "{prompt}"

{context_str}

Respond to this prompt as {self.name} would on social media. 
Keep it authentic, natural, and true to your character.
Post length should match your typical style.

Your post:"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            post_content = response.content.strip()
            
            # Track the post
            self._record_post(prompt, post_content)
            
            return post_content
            
        except Exception as e:
            print(f"Error generating post for {self.name}: {e}")
            return None
    
    def generate_comment(
        self,
        original_post: str,
        post_author: str,
        max_tokens: int = 150
    ) -> Optional[str]:
        """
        Generate a comment/reaction to another agent's post.
        
        Args:
            original_post: The post to comment on
            post_author: Who wrote the original post
            max_tokens: Maximum length of the comment
            
        Returns:
            The generated comment or None if agent chooses not to engage
        """
        
        # Decide if this agent should comment
        if not self._should_engage(original_post):
            return None
        
        system_prompt = self._build_system_prompt()
        
        user_prompt = f"""{post_author} posted: "{original_post}"

As {self.name}, respond to this post if it interests you or relates to your persona.
Your response should be authentic to your character and social behavior.

If you strongly relate: engage enthusiastically
If you disagree: feel free to challenge (if that fits your persona)
If you're supportive: validate and encourage (if that fits your persona)
If it doesn't interest you much: keep it brief or don't respond

Your comment (or say 'SKIP' if you wouldn't engage):"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            comment_content = response.content.strip()
            
            # Check if agent chose to skip
            if "SKIP" in comment_content.upper() or len(comment_content) < 5:
                return None
            
            # Track the interaction
            self._record_comment(post_author, original_post, comment_content)
            
            return comment_content
            
        except Exception as e:
            print(f"Error generating comment for {self.name}: {e}")
            return None
    
    def _should_post(self) -> bool:
        """Determine if agent should post based on engagement pattern"""
        
        pattern = self.persona.engagement_pattern.lower()

        if "very low" in pattern:
            return random.random() < 0.3
        elif "low" in pattern:
            return random.random() < 0.6
        elif "very high" in pattern:
            return random.random() < 0.95
        elif "high" in pattern:
            return random.random() < 0.85
        else:
            return random.random() < 0.8  # moderate default
    
    def _should_engage(self, post_content: str) -> bool:
        """Determine if agent should comment on a post"""
        
        # Check topic relevance
        post_lower = post_content.lower()
        relevant = any(topic.lower() in post_lower for topic in self.persona.topics)
        
        pattern = self.persona.engagement_pattern.lower()

        if "very high" in pattern:
            base_prob = 0.8
        elif "high" in pattern:
            base_prob = 0.6
        elif "very low" in pattern:
            base_prob = 0.15
        elif "low" in pattern:
            base_prob = 0.3
        else:
            base_prob = 0.4
        
        # Boost probability if topic is relevant
        if relevant:
            base_prob = min(base_prob + 0.2, 0.95)
        
        return random.random() < base_prob
    
    def _record_post(self, prompt: str, content: str):
        """Track a post in agent's history"""
        self.post_history.append({
            "type": "post",
            "prompt": prompt,
            "content": content,
            "iteration": len(self.post_history)
        })
        self.total_posts += 1
        
        # Extract topics (simple keyword matching)
        for topic in self.persona.topics:
            if topic.lower() in content.lower():
                self.topics_mentioned[topic] = self.topics_mentioned.get(topic, 0) + 1
    
    def _record_comment(self, target_author: str, target_post: str, comment: str):
        """Track a comment/interaction in agent's history"""
        self.interaction_history.append({
            "type": "comment",
            "target_author": target_author,
            "target_post": target_post,
            "comment": comment,
            "iteration": len(self.interaction_history)
        })
        self.total_comments += 1
        
        # Simple agreement/disagreement detection
        comment_lower = comment.lower()

        agreement_keywords = [
            "agree", "exactly", "yes", "great point", "love this",
            "you're right", "100%", "absolutely", "totally", "same",
            "well said", "couldn't agree more", "spot on", "true"
        ]
        disagreement_keywords = [
            "disagree", "actually", "but", "however", "not sure",
            "wrong", "counterpoint", "i don't think", "doubt",
            "not quite", "misleading", "oversimplified", "flawed"
        ]
        
        if any(kw in comment_lower for kw in agreement_keywords):
            self.agreements += 1
        elif any(kw in comment_lower for kw in disagreement_keywords):
            self.disagreements += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent's behavioral statistics"""
        return {
            "name": self.name,
            "archetype": self.archetype,
            "total_posts": self.total_posts,
            "total_comments": self.total_comments,
            "agreements": self.agreements,
            "disagreements": self.disagreements,
            "topics_mentioned": self.topics_mentioned,
            "current_community": self.current_community
        }
    
    def __repr__(self):
        return f"SocialAgent(name='{self.name}', archetype='{self.archetype}')"