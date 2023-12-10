import os
import requests
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langfuse.client import Langfuse
from langfuse.model import CreateTrace
from langfuse.callback import CallbackHandler
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers import ArxivRetriever
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
load_dotenv()

'''
과정)
1. 질문하기
2. search_question_chain를 활용하여 질문에 대한 프롬프트 생성
3. 2의 프롬프트를 기반으로, ArxivRetriever를 이용하여 논문 검색
5. 3의 요약 정보 출력 후 
이때 url과, 요약정보를 join하여 collapse_list_of_lists에 전달한다.
이를 통해 참고 자료 url을 기입할 수 있다.
6. 5의 정보는 질문프롬프트 갯수 * url 만큼 생성되며 이를 collapse_list_of_lists로 이용햐여 합치고,
 WRITER_SYSTEM_PROMPT와, RESEARCH_REPORT_TEMPLATE로 1200자의 보고서 출력하도록 한다.
7. langServe와 fastApi를 이용한 배포 진행
- http://localhost:8000/research-assistant/playground/
'''

retriever = ArxivRetriever()

handler = CallbackHandler(
    public_key = os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key = os.environ["LANGFUSE_SECRET_KEY"],
    host="https://us.cloud.langfuse.com"
)

RESULTS_PER_QUESTION = 1

ddg_search = DuckDuckGoSearchAPIWrapper()

SUMMARY_TEMPLATE = """{text} 
-----------
Using the above text, answer in short the following question: 
> {question}
-----------
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""

SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)


def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
    print("web_search query : ", query)
    results = ddg_search.results(query, num_results)

    print("web_search results : ", [r["link"] for r in results])
    return [r["link"] for r in results]


def scrape_text(url: str):
    print('scrape_text 실행!!!!1')
    # Send a GET request to the webpage
    try:
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the content of the request with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract all text from the webpage
            page_text = soup.get_text(separator=" ", strip=True)

            # Print the extracted text
            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retrieve the webpage: {e}"

url = "https://blog.langchain.dev/announcing-langsmith/"

# scrape_and_summarize_chain = RunnablePassthrough.assign(
#     summary = RunnablePassthrough.assign(
#     text=lambda x: scrape_text(x["url"])[:10000]
# ) | SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
# ) | (lambda x: f"URL: {x['url']}\n\nSUMMARY: {x['summary']}")


# web_search_chain = RunnablePassthrough.assign(
#     urls = lambda x: web_search(x['question'])
# ) | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | scrape_and_summarize_chain.map()

SUMMARY_TEMPLATE = """{doc} 
-----------
Using the above text, answer in short the following question: 
> {question}
-----------
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""

SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)


scrape_and_summarize_chain = RunnablePassthrough.assign(
    summary =  SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
) | (lambda x: f"Title: {x['doc'].metadata['Title']}\n\nSUMMARY: {x['summary']}")


web_search_chain = RunnablePassthrough.assign(
    docs = lambda x: retriever.get_summaries_as_docs(x["question"])
)| (lambda x: [{"question": x["question"], "doc": u} for u in x["docs"]]) | scrape_and_summarize_chain.map()


# "user" 문자열은 해당 메시지가 사용자(user)에 의해 전달되었다는 것을 나타냅니다. 이는 대화의 컨텍스트를 설정하는 데 중요합니다. 대화형 시스템에서는 사용자와 시스템(또는 어시스턴트) 간의 상호작용을 모델링합니다. 각 메시지는 누가 말하고 있는지를 명확하게 하기 위해 발신자 식별자를 필요로 합니다.
SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 google search queries to search online that form an "
            "objective opinion from the following: {question}\n"
            "You must respond with a list of strings in the following format: "
            '["query 1", "query 2", "query 3"].',
        ),
    ]
)



search_question_chain = SEARCH_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser() | json.loads

full_research_chain = search_question_chain | ((lambda x: [{"question": q} for q in x])) | web_search_chain.map()

WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."  # noqa: E501

# Report prompts from https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/master/prompts.py
RESEARCH_REPORT_TEMPLATE = """Information:
--------
{research_summary}
--------
Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
The report should focus on the answer to the question, should be well structured, informative, \
in depth, with facts and numbers if available and a minimum of 1,200 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
Please do your best, this is very important to my career.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)

def collapse_list_of_lists(list_of_lists):
    print('list_of_lists : ',list_of_lists)
    content = []
    for l in list_of_lists:
        content.append("\n\n".join(l))
    
    return "\n\n".join(content)


chain = RunnablePassthrough.assign(
    research_summary= full_research_chain | collapse_list_of_lists
) | prompt | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()

handler.langfuse.flush()
#!/usr/bin/env python
from fastapi import FastAPI
from langserve import add_routes


app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    chain,
    path="/research-assistant",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)