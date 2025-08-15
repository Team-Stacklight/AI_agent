from decision_agent import decision
from agent import generate_answers



def handle_new_message(conversation_history: list[dict], 
                       latest_message: str,
                       group_name:str,
                       learning_topic: str, 
                       goal: str):
    """
    Decide whether the heavyweight agent should respond.
    Returns structured output if the agent responds, else None.
    """
    # Gatekeeper decides
    should_respond = decision(conversation_history, latest_message)

    if not should_respond:
        return None  # skip calling heavyweight LLM

    # Call main agent only if needed
    return generate_answers(
        conversation_history=conversation_history,
        latest_message=latest_message,
        group_name=group_name,
        learning_topic=learning_topic,
        goal=goal,
        response_type="live"
    )