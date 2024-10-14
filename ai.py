#from langchain_ollama import ChatOllama

from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_ollama.llms import OllamaLLM
from langchain import hub

llm = OllamaLLM(
    model = "llama3",
    temperature = 0.8,
    num_predict = 256,
    stream = True
)

wikipedia = WikipediaAPIWrapper()

# Define la herramienta de búsqueda
tools = [
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Busca información en Wikipedia"
    )
]

#messages = [
 #   ("system", "Siempre hablas en Español"),
  #  ("human", "Buscame la edad de Barack Obama."),
#]

question = "Busca información sobre Barack Obama y hazme un resumen corto"

template = PromptTemplate(
    input_variables=["question"],
    template="Responde la siguiente pregunta: {question}"
)



formatted_message = template.format(question=question)
print(formatted_message)
# Inicializa un agente con las herramientas
agent = create_react_agent(
    tools=tools,
    llm=llm,
    prompt = hub.pull("hwchase17/react"),
)

# Realiza una búsqueda
#response = agent.run({formatted_message})

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": formatted_message})
#print(response)

#llm.invoke(messages)
#for chunk in llm.stream(messages):
    # Acceder al texto generado a medida que se va creando
    #print(chunk.content, end='', flush=True)