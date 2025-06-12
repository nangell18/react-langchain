from dotenv import load_dotenv
from langchain.agents import tool
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
    )  # we need the render_text_description because we cannot put tool objects in llm, we need strings of our tools

    llm = ChatOpenAI(
        temperature=0, stop=["\nObservation"]
    )  # the stop parameter is there to ensure the LLM stops once it finds the real result that we are looking for. if this is not supplied, then it will keep generating text and hallucinate