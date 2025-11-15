from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="llama3.2:3b", temperature=0.7)

prompt = ChatPromptTemplate.from_template(
    """Extract the song name from this input: {user_input}
    
    Return only JSON: {{"song": "SONG_NAME"}}"""
)

chain = prompt | llm | StrOutputParser()

# Test 1
result1 = chain.invoke({"user_input": "play stargazing"})
print(f"Test 1: {result1}")

# Test 2
result2 = chain.invoke({"user_input": "play levels by avicii"})
print(f"Test 2: {result2}")

# Test 3
result3 = chain.invoke({"user_input": "play wake me up"})
print(f"Test 3: {result3}")