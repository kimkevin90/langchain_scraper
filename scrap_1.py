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
2. web_search를 통해 질문에 대한 urls 출력
3. scrape_text를 통해 2의 url에 대해 스크래핑
4. 3.의 스크래핑 text를 이용하여 요약 정보 출력

'''
handler = CallbackHandler(
    public_key = os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key = os.environ["LANGFUSE_SECRET_KEY"],
    host="https://us.cloud.langfuse.com"
)

RESULTS_PER_QUESTION = 2

ddg_search = DuckDuckGoSearchAPIWrapper()

def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
    results = ddg_search.results(query, num_results)
    # print("web_search results : ", [r["link"] for r in results])
    return [r["link"] for r in results]

SUMMARY_TEMPLATE = """{text} 
-----------
Using the above text, answer in short the following question: 
> {question}
-----------
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""

SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)

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

# RunnablePassthrough.assign 메소드는 Langchain 환경에서 외부 코드를 실행한 결과를 특정 변수에 할당하는 기능을 수행합니다. 이 메소드를 사용하여, Python 함수나 다른 언어의 코드를 실행한 후, 그 결과를 Langchain 내의 변수에 저장할 수 있습니다.
scrape_and_summarize_chain = RunnablePassthrough.assign(
    text=lambda x: scrape_text(x["url"])[:10000]
) | SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()

# question에 대한 webUrl 획득 후  scrape_text실행, 결과는 RESULTS_PER_QUESTION 만큼의 list로 출력
chain = RunnablePassthrough.assign(
    urls = lambda x: web_search(x['question'])
) | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | scrape_and_summarize_chain.map()

result = chain.invoke(
    {
        "question": "What is langsmith",
    },
    config={"callbacks": [handler]}
)
handler.langfuse.flush()
print(result)