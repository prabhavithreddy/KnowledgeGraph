import sys
sys.path.append(".")

import os
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm = AzureChatOpenAI(
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name=os.environ["AZURE_DEPLOYMENT_NAME"],
    openai_api_type=os.environ["OPENAI_API_TYPE"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    default_headers={
            "fds-message-id": "14485820-1e81-4f0b-a708-d386d4672a81",
            "fds-conversation-id": "9616f92c-070c-4f3b-8e51-5321b472b24c"
        }
)
llm_transformer = LLMGraphTransformer(llm=llm)

from langchain_community.graphs import Neo4jGraph
#graph = Neo4jGraph(username="neo4j", password="graphdatabase", url="neo4j://localhost:7687")

from langchain_core.documents import Document

text = """
Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris.
"""
documents = [Document(page_content=text)]
graph_documents = llm_transformer.convert_to_graph_documents(documents)
print(f"Nodes:{graph_documents[0].nodes}")
print(f"Relationships:{graph_documents[0].relationships}")