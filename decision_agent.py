from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from vertexai import init
from agent import generate_answers
import os


load_dotenv()
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path


deciding_model = ChatVertexAI(
    model_name="gemini-2.5-flash-lite",
    temperature=0.0,
    max_output_tokens=5,
)

# Unsure if this is necessary
init(
    project="woven-nimbus-461919-j1",
    location="us-central1"
)

gate_prompt = ChatPromptTemplate.from_template("""
<Prompt>
  <Instructions>
    Based on the new message and recent history, decide if the heavyweight agent should respond.
    Respond ONLY with:
      YES  → if the heavyweight agent **should** respond
      NO   → if the heavyweight agent should **not** respond
  </Instructions>
  <History>{conversation_history}</History>
  <NewMessage>{message}</NewMessage>
</Prompt>
""")


"""A function that accepts a list of dicts, and the latest message. 
The first step turns the history into a string that's formatted. The 
second step chains the system message to the model, and generates a 
string response (YES or NO). 

The function will return a bool, and if the 2.5 Flash Lite generates 
a YES response, it will be true."""

def decision(conversation_history: list[dict], message: str) -> bool:
    conv_string = "\n".join(f"{m['sender']}: {m['message']}" for m in conversation_history)
    chain = gate_prompt | deciding_model | StrOutputParser()
    decision = chain.invoke({"conversation_history": conv_string, "message": message})
    
    # Error handling just in case the model generates something else 
    try:
        return decision.strip().upper() == "YES"
    except Exception:
        return False
    
