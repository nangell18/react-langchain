from dotenv import load_dotenv
load_dotenv()

def get_text_length(text:str) -> int:
    """Returns the length of a text by characters""" # this description is important because this is going to decide if the LLM is going to use this tool or not in its reasoning engine
    return len(text)
if __name__ == '__main__':
    print("Hello ReAct LangChain!")
    print(get_text_length(text="Nate"))


