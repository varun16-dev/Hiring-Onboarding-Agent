from langchain_ollama import ChatOllama

def create_llm():
    llm = ChatOllama(
        model="qwen3:0.6b",
        temperature=0.2 #more factual response/less random
    )
    return llm

if __name__== "__main__":
  user_input = input("Ask a question to the LLM:")
  llm = create_llm()
  response = llm.invoke(user_input)
  print("\n\n\n CLEAN RESPONSE \n\n\n",response.content)