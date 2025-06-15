from typing import Union, List

from dotenv import load_dotenv
from langchain.agents import tool
from langchain.tools import Tool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description
from langchain_openai import ChatOpenAI

load_dotenv()


@tool()
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip(("'\n")).strip(
        " " " "
    )  # stripping any non alphabetic characters just in case
    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name:str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found.")


if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    tools = [get_text_length]

    # this is a chain of thought prompt
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought:
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatOpenAI(
        temperature=0, stop=["\nObservation"], model="gpt-4o-mini"
    )

    agent = (
        {"input": lambda x: x["input"]} | prompt | llm | ReActSingleInputOutputParser()
            #ReActSingleInputOutputParser is the PARSING step within the ReAct steps framework picture. without the output parsar, we are not able to invoke the actual function because the previous output was one giant string but we need to separate the tool and output. that way here we can call the actual tool
    )
    # ^ right above is PARSING step

    agent_step: Union[AgentAction,AgentFinish] = agent.invoke({"input": "What is the length in characters of the text Nate"})

    print(agent_step)

    #this is the TOOL execution part
    if isinstance(agent_step,AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools,tool_name)
        tool_input = agent_step.tool_input

        obervation = tool_to_use.func(str(tool_input))

        print(f"{obervation}=")
