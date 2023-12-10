# LangChain을 이용한 웹 스크래핑 및 응답 생성

LangChain을 활용하여 웹 스크래핑을 수행하고, 사용자의 질문에 대한 응답을 생성합니다. 각 단계는 아래와 같습니다.

## 과정 설명

### 1. 질문하기

- 사용자가 질문을 하면, 이후의 과정이 시작됩니다.

### 2. `search_question_chain` 사용

- 사용자의 질문에 대한 프롬프트를 생성합니다.
- `SEARCH_PROMPT`와 `ChatOpenAI` 모델을 사용하여 사용자의 질문을 처리하고, 적절한 검색 쿼리를 생성합니다.

### 3. `web_search_chain` 실행

- 생성된 프롬프트를 기반으로, 관련된 웹사이트의 URL을 검색합니다.
- `web_search` 함수를 이용하여 DuckDuckGo 검색 API를 통해 웹 검색을 수행하고, 검색 결과 URL을 반환합니다.

### 4. `scrape_and_summarize_chain`을 통한 스크래핑 및 요약

- 검색된 URL에서 텍스트를 스크래핑합니다.
- `scrape_text` 함수를 사용하여 각 URL에서 HTML 콘텐츠를 추출하고, BeautifulSoup를 이용해 텍스트를 파싱합니다.
- 파싱된 텍스트를 `SUMMARY_PROMPT`와 `ChatOpenAI` 모델을 사용하여 요약합니다.
- 각 URL에 대해 요약된 텍스트와 원본 URL을 조합하여 결과를 생성합니다.

### 5. 정보 합치기 및 보고서 생성

- 요약 정보와 질문 프롬프트의 조합을 `collapse_list_of_lists`를 통해 합칩니다.
- `WRITER_SYSTEM_PROMPT`와 `RESEARCH_REPORT_TEMPLATE`을 사용하여 1200자 이상의 보고서를 작성합니다.

### 6. LangServe와 FastAPI를 이용한 배포

- `http://localhost:8000/research-assistant/playground/`에서 배포를 진행합니다.

## Lanfuse를 활용한 Trace 설정

```python
handler = CallbackHandler(
    public_key = os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key = os.environ["LANGFUSE_SECRET_KEY"],
    host="https://us.cloud.langfuse.com"
)

handler.langfuse.flush()
```
