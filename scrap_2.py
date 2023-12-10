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
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
load_dotenv()

'''
과정)
1. 질문하기
2. search_question_chain를 활용하여 질문에 대한 프롬프트 생성
3. 2의 프롬프트를 기반으로, web_search를 통해 질문에 대한 urls 출력
4. scrape_text를 통해 2의 url에 대해 스크래핑
5. 3.의 스크래핑 text를 이용하여 요약 정보 출력

'''

handler = CallbackHandler(
    public_key = os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key = os.environ["LANGFUSE_SECRET_KEY"],
    host="https://us.cloud.langfuse.com"
)

RESULTS_PER_QUESTION = 2

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

scrape_and_summarize_chain = RunnablePassthrough.assign(
    text=lambda x: scrape_text(x["url"])[:10000]
) | SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()


web_search_chain = RunnablePassthrough.assign(
    urls = lambda x: web_search(x['question'])
) | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | scrape_and_summarize_chain.map()


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

chain = search_question_chain | ((lambda x: [{"question": q} for q in x])) | web_search_chain.map()

result = chain.invoke(
    {
        "question": "what is the difference between langsmith and langchain",
    },
    config={"callbacks": [handler]}
)
handler.langfuse.flush()
print(result)