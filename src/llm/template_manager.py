from langchain.prompts import PromptTemplate


class TemplateManager:
    def __init__(self):
        self.templates = {
            # 如何？
            0 : PromptTemplate(
                input_variables=["question", "related_str"],
                template="根据已知信息筛选出最相关信息，综合地简洁的来回答问题。\n问题是：{question}\n已知信息：{related_str}\n答案是："
            ),
            # 如何？
            1 : PromptTemplate(
                input_variables=["question",],
                template="根据已知信息中简洁和专业的来回答问题.\n问题是：{question}\n答案是："
            ),
            
        }

    def get_template(self, template_name):
        return self.templates.get(template_name, None)
    
template_manager = TemplateManager()