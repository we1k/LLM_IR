from langchain.agents import AgentType, load_tools, initialize_agent
from langchain.agents import AgentExecutor, create_sql_agent
from langchain.llms import ChatGLM
from langchain.chains import QAWithSourcesChain, LLMChain

from langchain.chains import RetrievalQA
from src.llm.template_manager import template_manager

endpoint_url = ("http://127.0.0.1:29501")

llm = ChatGLM(
    endpoint_url=endpoint_url,
    max_token=512,
    history=[],
    temperature=0.7,
    # top_p=0.9,
    model_kwargs={"sample_model_args": False},
)

input_dict = {"question": "如何打开车门", "related_str": "在中央显示屏中点击-车辆设置-电动尾门，进入电动尾门设置界面。"}
print(template_manager.get_template(0).format_prompt(**input_dict))
chain = LLMChain(llm=llm, prompt=template_manager.get_template(0))
result = chain(input_dict)
print(result)
