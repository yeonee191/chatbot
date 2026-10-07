"""DATA 폴더의 PDF를 검색해서 답변하는 간단한 RAG 챗봇입니다."""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader
from validation import run_validation


# .env 파일의 OPENAI_API_KEY를 환경 변수로 불러옵니다.
load_dotenv()

DATA_DIR = Path(__file__).parent / "DATA"


class DirectOpenAIEmbeddings(Embeddings):
    """토크나이저 파일을 다운로드하지 않고 OpenAI 임베딩 API를 호출합니다."""

    def __init__(self, model: str, batch_size: int = 100) -> None:
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAI()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """문서들을 여러 묶음으로 나누어 OpenAI 임베딩 API에 보냅니다."""
        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            ordered_data = sorted(response.data, key=lambda item: item.index)
            all_embeddings.extend(item.embedding for item in ordered_data)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """사용자 질문 한 개를 OpenAI 임베딩 API에 보냅니다."""
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
        )
        return response.data[0].embedding


def load_pdf_documents() -> list[Document]:
    """DATA 폴더의 모든 PDF를 페이지 단위 문서로 읽습니다."""
    documents: list[Document] = []

    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            # PDF 페이지에서 글자를 추출합니다. 글자가 없는 페이지는 건너뜁니다.
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue

            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                )
            )

    if not documents:
        raise FileNotFoundError("DATA 폴더에서 읽을 수 있는 PDF 문서를 찾지 못했습니다.")

    return documents


@st.cache_resource(show_spinner="문서를 읽고 검색 데이터베이스를 만드는 중입니다...")
def create_vector_store() -> InMemoryVectorStore:
    """문서를 잘게 나누고 OpenAI 임베딩을 만들어 메모리 벡터 DB에 저장합니다."""
    page_documents = load_pdf_documents()

    # 긴 페이지를 검색하기 좋은 크기의 조각으로 나눕니다.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1_000,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = splitter.split_documents(page_documents)

    # 토크나이저 파일을 별도로 받지 않고 지정한 OpenAI 임베딩 모델을 사용합니다.
    embeddings = DirectOpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(chunks)
    return vector_store


def build_answer_chain() -> ChatPromptTemplate | object:
    """문서 내용만 근거로 답변하도록 하는 최신 Runnable 체인을 만듭니다."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
너는 제공된 문서에 근거해서만 답변하는 공무원 여비 전문 안내 도우미야.
반드시 아래 문서 내용만 사용하고, 문서에 없는 내용은 추측하거나 일반 상식으로 보완하지 마.
답을 뒷받침하는 내용이 문서에 없으면 정확히 '제공된 문서에서 답을 확인할 수 없습니다.'라고 답해.
질문과 관련 없는 문서 내용은 사용하지 마.
답변은 초보자가 이해하기 쉬운 한국어로 작성해.

문서 내용:
{context}
""",
            ),
            ("human", "질문: {question}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return prompt | llm | StrOutputParser()


def format_sources(retrieved_documents: list[tuple[Document, float]]) -> None:
    """검색된 문서의 파일명과 근거 구간을 화면에 표시합니다."""
    st.markdown("#### 출처 및 근거 문장")

    shown_sources: set[tuple[str, int]] = set()
    for document, score in retrieved_documents:
        source = str(document.metadata.get("source", "알 수 없는 파일"))
        page = int(document.metadata.get("page", 0))
        source_key = (source, page)

        # 같은 PDF 페이지에서 여러 조각이 검색되면 한 번만 표시합니다.
        if source_key in shown_sources:
            continue
        shown_sources.add(source_key)

        st.markdown(f"**파일:** `{source}` (p. {page})")
        st.caption(document.page_content.replace("\n", " ").strip())


def main() -> None:
    st.set_page_config(page_title="공무원 여비 RAG 챗봇", page_icon="📚")
    st.title("📚 공무원 여비 RAG 챗봇")
    st.write("DATA 폴더의 문서만 근거로 답변합니다.")

    if not st.session_state.get("api_key_ready"):
        import os

        if not os.getenv("OPENAI_API_KEY"):
            st.error(".env 파일에 OPENAI_API_KEY를 입력한 뒤 앱을 다시 실행해 주세요.")
            st.stop()
        st.session_state.api_key_ready = True

    try:
        vector_store = create_vector_store()
        answer_chain = build_answer_chain()
    except Exception as error:
        st.error(f"문서나 OpenAI 설정을 준비하는 중 오류가 발생했습니다: {error}")
        st.stop()

    # 대표 질문으로 검색과 답변 품질을 점검하는 개발용 검증 버튼입니다.
    with st.sidebar:
        st.header("검증 도구")
        if st.button("RAG 검증 실행"):
            with st.spinner("검증 질문을 실행하는 중입니다..."):
                validation_results = run_validation(vector_store, answer_chain)
            st.session_state.validation_results = validation_results

        for result in st.session_state.get("validation_results", []):
            icon = "✅" if result["passed"] else "❌"
            st.markdown(f"{icon} {result['name']}")
            if not result["passed"]:
                st.caption(
                    "키워드: %s / 출처: %s / 거부: %s"
                    % (result["keywords_ok"], result["source_ok"], result["abstention_ok"])
                )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                format_sources(message["sources"])

    question = st.chat_input("문서에 대해 질문해 주세요.")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("문서에서 관련 내용을 찾는 중입니다..."):
            # 사례형 질문의 정답과 관련 규정이 서로 다른 페이지에 있을 수 있으므로 넉넉히 검색합니다.
            retrieved_documents = vector_store.similarity_search_with_score(question, k=8)
            context = "\n\n".join(
                f"[출처: {document.metadata.get('source')} p.{document.metadata.get('page')}]\n"
                f"{document.page_content}"
                for document, _score in retrieved_documents
            )
            answer = answer_chain.invoke({"context": context, "question": question})

        st.markdown(answer)
        format_sources(retrieved_documents)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": retrieved_documents,
        }
    )


if __name__ == "__main__":
    main()
