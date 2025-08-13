from pydantic import BaseModel, ValidationError
from typing import List


class LiveContribution(BaseModel): 
    message: str # AI's contribution
    focus_topic: str 
    suggested_questions: List[str] # Optional follow-up questions for participants
    insight_type: str # Analogy, clarification etc
    insights: List[str] # Optional observations about the discussion so far
    urgency_score: int # A score that determines whether it should chime in immediately
    

class Summary(BaseModel):
    overall_summary: str 
    key_themes: List[str]
    insights: List[str]
    follow_up_plan: str
    highlighted_messages: List[str]