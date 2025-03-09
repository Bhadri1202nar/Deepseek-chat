import streamlit as st
from typing import List, Dict
from langchain_ollama.llms import OllamaLLM
from langgraph.graph import StateGraph, START, END

# Define state class
class State(Dict):
    messages: List[Dict[str, str]]

# Initialize StateGraph
graph_builder = StateGraph(State)
llm = OllamaLLM(model="deepseek-r1:1.5b")

# Define chatbot function
def chatbot(state: State):
    messages = state.get("messages", [])[:]  # Create a copy to avoid modifying the original
    response = llm.invoke(messages)
    messages.append({"role": "assistant", "content": response})  # Modify the copy
    return {"messages": messages}  # Return a new state

# Add nodes and edges
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# Streamlit UI
st.title("Langchain Ollama Chatbot")
st.write("A chatbot with text support.")

# Initialize session state if not exists
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input (Text)
txt = st.text_input("User:", "", key="user_input")

# Process input
if txt:
    state = {"messages": [{"role": "user", "content": txt}]}
    
    for event in graph.stream(state):
        for value in event.values():
            response = value["messages"][-1]["content"]
            st.session_state.chat_history.append((txt, response))

# Display chat history
for user_msg, bot_msg in st.session_state.chat_history:
    st.write(f"**User:** {user_msg}")
    st.write(f"**Assistant:** {bot_msg}")
