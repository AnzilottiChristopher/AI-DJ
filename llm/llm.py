# LLM File

from typing import Callable, Literal
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json, re, sys, time
import customtkinter as ctk

class LlamaLLM():
    def __init__(self):
        self.llm = ChatOllama(model="llama3.1", temperature=0)
        self.COMMANDS: dict[str, Callable[[], None]] = {
        "hello": self.cmd_hello,
        "start_dj": self.cmd_start_dj,
        "stop_dj": self.cmd_stop_dj,
        "help": self.cmd_help,
        "none": lambda: print("[NO ACTION] No matching command.")
        }
        
        self.prompt = ChatPromptTemplate.from_template(
            """You are an intent classifier for a DJ app.
        Return STRICT JSON with fields: intent (one of: "hello","start_dj","stop_dj","help","none") and reason.

        User input: "{user_input}"

        Rules:
        - If the user asks to start/queue/launch dj/music/mixing, intent="start_dj".
        - If they ask to stop/end/quit dj/music, intent="stop_dj".
        - If they greet (hello/hi/hey), intent="hello".
        - If they ask for help or commands, intent="help".
        - Otherwise, intent="none".
        Return ONLY JSON, no code block, no extra text."""
        )
    
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    # ---- commands ----
    def cmd_hello():
        print("[CMD] Hello! This is the hello command.")

    def cmd_start_dj():
        print("[CMD] Starting DJ pipeline... (placeholder)")

    def cmd_stop_dj():
        print("[CMD] Stopping DJ pipeline... (placeholder)")

    def cmd_help():
        print("[CMD] Available commands: hello, start_dj, stop_dj, help")
        
    def extract_json(s: str) -> dict:
        """
        Try strict json.loads; if it fails, try to extract the first {...} block.
        On failure, return {"intent": "none", "reason": "..."}.
        """
        if not s or not s.strip():
            return {"intent": "none", "reason": "Empty model output."}

        # First try direct JSON
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass

        # Try to pull a JSON object substring
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        # Give up safely
        return {"intent": "none", "reason": "Could not parse JSON."}
    

    def classify(self, text: str) -> dict:
        msg = self.chain.invoke({"user_input": text})
        # LangChain ChatOllama returns a BaseMessage; get the .content
        raw = getattr(msg, "content", msg)
        print(raw)
        data = self.extract_json(raw)
        print("Raw Data: ")
        print(raw)
        intent = data.get("intent", "none")
        if intent not in self.COMMANDS:
            intent = "none"
        # debug if parse failed
        if intent == "none" and data.get("reason","").startswith("Could not parse"):
            print("[DEBUG] Raw model output:\n", raw)
        return {"intent": intent, "reason": data.get("reason", "")}

    def dispatch(self, intent: str):
        self.COMMANDS[intent]()
    
    def process_response(self, input:str) -> str:
        try:
            response = input
            #start_time = time.time()
        except (EOFError, KeyboardInterrupt):
            return "Couldn't process response"
        res = self.classify(response)
        self.dispatch(res["intent"])
        return f"[INTENT] {res['intent']} — {res['reason']}"
        
        

# # ---- commands ----
# def cmd_hello():
#     print("[CMD] Hello! This is the hello command.")

# def cmd_start_dj():
#     print("[CMD] Starting DJ pipeline... (placeholder)")

# def cmd_stop_dj():
#     print("[CMD] Stopping DJ pipeline... (placeholder)")

# def cmd_help():
#     print("[CMD] Available commands: hello, start_dj, stop_dj, help")
    



# # map intents -> functions
# COMMANDS: dict[str, Callable[[], None]] = {
#     "hello": cmd_hello,
#     "start_dj": cmd_start_dj,
#     "stop_dj": cmd_stop_dj,
#     "help": cmd_help,
#     "none": lambda: print("[NO ACTION] No matching command.")
# }

# # ---- LLM setup ----
# llm = ChatOllama(model="llama3.1", temperature=0)

# prompt = ChatPromptTemplate.from_template(
#     """You are an intent classifier for a DJ app.
# Return STRICT JSON with fields: intent (one of: "hello","start_dj","stop_dj","help","none") and reason.

# User input: "{user_input}"

# Rules:
# - If the user asks to start/queue/launch dj/music/mixing, intent="start_dj".
# - If they ask to stop/end/quit dj/music, intent="stop_dj".
# - If they greet (hello/hi/hey), intent="hello".
# - If they ask for help or commands, intent="help".
# - Otherwise, intent="none".
# Return ONLY JSON, no code block, no extra text."""
# )

# # TO-DO
# #1. Recognize song names to pass into the queue.
# #2. Recognize parameters in the prompt, pass them into commands.
# #3. 

# chain = prompt | llm | StrOutputParser()

# def extract_json(s: str) -> dict:
#     """
#     Try strict json.loads; if it fails, try to extract the first {...} block.
#     On failure, return {"intent": "none", "reason": "..."}.
#     """
#     if not s or not s.strip():
#         return {"intent": "none", "reason": "Empty model output."}

#     # First try direct JSON
#     try:
#         return json.loads(s)
#     except json.JSONDecodeError:
#         pass

#     # Try to pull a JSON object substring
#     m = re.search(r"\{.*\}", s, re.S)
#     if m:
#         try:
#             return json.loads(m.group(0))
#         except json.JSONDecodeError:
#             pass

#     # Give up safely
#     return {"intent": "none", "reason": "Could not parse JSON."}

# def classify(text: str) -> dict:
#     msg = chain.invoke({"user_input": text})
#     # LangChain ChatOllama returns a BaseMessage; get the .content
#     raw = getattr(msg, "content", msg)
#     data = extract_json(raw)
#     print("Raw Data: ")
#     print(raw)
#     intent = data.get("intent", "none")
#     if intent not in COMMANDS:
#         intent = "none"
#     # debug if parse failed
#     if intent == "none" and data.get("reason","").startswith("Could not parse"):
#         print("[DEBUG] Raw model output:\n", raw)
#     return {"intent": intent, "reason": data.get("reason", "")}

# def dispatch(intent: str):
#     COMMANDS[intent]()

if __name__ == "__main__":
    llm = LlamaLLM()
    
#     print("Say something (e.g., 'spin up the dj', 'please stop the dj', 'hello'). Ctrl+C to exit.")
#     #start_time = None
#     while True:
#         try:
#             user = input("> ")
#             #start_time = time.time()
#         except (EOFError, KeyboardInterrupt):
#             print("\nbye!")
#             break
#         res = classify(user)
#         print(f"[INTENT] {res['intent']} — {res['reason']}")
#         dispatch(res["intent"])
#         #end_time = time.time()
#         #elapsed_time = end_time - start_time
#         #print(f"Total Time:{elapsed_time}")
        
        


