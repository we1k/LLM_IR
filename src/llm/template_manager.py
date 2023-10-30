from langchain.prompts import PromptTemplate


class TemplateManager:
    def __init__(self):
        self.templates = {
            # 如何？
            0 : PromptTemplate(
                input_variables=["question", "related_str"],
                template="""
                你是一位智能汽车使用说明的问答助手，根据在说明书提取的已知信息，完整、准确并简要地回答问题。不要回答额外信息。\n问题是：{question}\n已知信息：{related_str}\n答案是："
                """
            ),
            # 总结
            1 : PromptTemplate(
                input_variables=["question",],
                template="请简要地总结上述回答，只保留与问题最相关的部分。问题是：{question}"
            ),
            
        }

    def get_template(self, template_name):
        return self.templates.get(template_name, None)
    
template_manager = TemplateManager()