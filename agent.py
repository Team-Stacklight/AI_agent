from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from answer_model import LiveContribution, Summary
from vertexai import init
from dotenv import load_dotenv
from typing import List
import json
import os
import re

load_dotenv()
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

init(
    project="woven-nimbus-461919-j1",
    location="us-central1"
)

model_name = "gemini-2.5-flash"

# The type of answer the model will generate
model_answer = {
    "LiveContribution": LiveContribution,
    "Summary": Summary
}

# Initialize the model via langchain
init_llm = ChatVertexAI(
    model_name=model_name,
    temperature=0.2,
    max_output_tokens=2000,
)

"""
{conversation_history} would be all the messages in the chat. 
The output is structured JSON, I think that would be the easiest 
for the backend to save or display. 

There are two different Pydantic structures, for the two different kinds 
of answers we want. One is designed for the occasional call to the LLM, 
or when it wants to chime in. The other is for the end of chat summary. 

Below you'll find an XML prompt with the instructions to the LLM. XML 
prompts generally lead to better results. 
"""

prompt_template = ChatPromptTemplate.from_template("""
<Prompt>
  <Role>
    You are an AI facilitator for workplace learning groups. Engage dynamically in text-message style conversations, providing concise insights, follow-up questions, and occasional observations.
  </Role>
  <GroupInfo>
    <GroupName>{group_name}</GroupName>
    <LearningTopic>{learning_topic}</LearningTopic>
    <Goal>{goal}</Goal>
  </GroupInfo>
  <Instructions>
    <LiveParticipation>
      Respond in real time to messages when relevant. Keep responses short, actionable, and clear, suitable for text-message style chat. Include suggestions or insights if they add value.
    </LiveParticipation>
    <EndOfChatSummary>
      When the discussion ends, provide a structured summary of the conversation. Include:
      <HighLevelSummary>A concise overview of topics discussed.</HighLevelSummary>
      <KeyInsights>Important takeaways or patterns.</KeyInsights>
      <FollowUpQuestions>Optional suggested questions or next steps.</FollowUpQuestions>
      <HighlightedMessages>Optional standout messages from participants.</HighlightedMessages>
    </EndOfChatSummary>
    <OutputFormat>
      Respond ONLY with JSON. For live contributions:
      {{
        "message": "...",
        "focus_topic": "...",
        "suggested_questions": ["...", "..."],
        "insight_type": "...",
        "insights": ["...", "..."],
        "urgency_score": 0.0
      }}
      For end-of-chat summary:
      {{
        "overall_summary": "...",
        "key_themes": ["...", "..."],
        "insights": ["...", "..."],
        "follow_up_plan": ["...", "..."],
        "highlighted_messages": ["...", "..."]
      }}
    </OutputFormat>
    <BehaviorHints>
      For live contributions, keep outputs concise, text-message style, and do not approach the maximum token limit. 
      End-of-chat summaries can be longer and may use more tokens to capture the full discussion. Only generate end-of-chat summary when triggered.
    </BehaviorHints>
  </Instructions>
  <ConversationHistory>{conversation_history}</ConversationHistory>
  <NewMessage>{message}</NewMessage>
</Prompt>
""")

"""
The response type will preliminarily be decided by the caller, 
whether a message is calling generate_answers with response_type="live" or 
not. Not sure if this is the best way but I went with the first solution off 
the top of my head. 

A bit hard to say if this is really the best solution at the moment, but I 
figured we could at least work off of this scaffold. Also, I have not implemented
any error handling in the JSON parse, unsure if it's needed. 

I think the WebSockets/Chat backend will handle the call to the LLM 
whenever a user sends a message or if the chat is inactive for a while. 
The summary can also be invoked with the press of a physical button such as 
End Chat. 
"""

def generate_answers(
    conversation_history: List[dict],
    latest_message: str,
    group_name: str,
    learning_topic: str,
    goal: str,
    response_type: str
) -> str:

    conversation_history_str = "\n".join(
        f"{msg['sender']}: {msg['message']}" for msg in conversation_history
    )

    # Chain which will get a clean, generated LLM response
    chain = prompt_template | init_llm | StrOutputParser()

    prompt_input = {
        "conversation_history": conversation_history_str,
        "group_name": group_name,
        "learning_topic": learning_topic,
        "goal": goal,
        "message": latest_message
    }

    # Get JSON response as a string
    response_str = chain.invoke(prompt_input).strip()

    # Strip markdown fences if present (just simple cleanup)
    response_str = response_str.strip()
    response_str = re.sub(r"^```(?:json)?\s*", "", response_str)
    response_str = re.sub(r"\s*```$", "", response_str)

    # response_str should be valid JSON now, parse directly
    parsed_dict = json.loads(response_str)
    model_class = model_answer["LiveContribution"] if response_type == "live" else model_answer["Summary"]
    structured_output = model_class.model_validate(parsed_dict)

    return structured_output