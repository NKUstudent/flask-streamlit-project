import modelchoice
from langchain_community.chat_models import ChatZhipuAI

modelchoice.setenv()

# 项目配置
class LangchainCFG(object):
    chat_llm = ChatZhipuAI(model="glm-4")
    embedding_model_name = 'C:\\Users\\墨池洗砚\\Desktop\\学习\\暑期实训\\practical-train-project\\m3e-base'
    vector_store_path = 'D:\\vector_store'
    doc_path = 'D:\\uploads'
    source_file_path = 'D:\\夸克网盘\\墨池\\《红楼梦》.txt'