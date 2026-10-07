"""RAG 문서 로딩, 검색, 답변, 출처를 점검하는 간단한 검증 모듈입니다."""

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class ValidationCase:
    """한 개의 검증 질문과 기대 결과입니다."""

    name: str
    question: str
    answer_keywords: tuple[str, ...] = ()
    expected_source_pages: tuple[tuple[str, int], ...] = ()
    must_abstain: bool = False


VALIDATION_CASES = (
    ValidationCase(
        name="국내출장 여비 항목",
        question="근무지 외 국내출장 시 지급되는 여비 항목은 무엇인가요?",
        answer_keywords=("운임", "숙박비", "식비", "일비"),
        expected_source_pages=(("2024년 9장 공무원여비업무 처리기준.pdf", 43),),
    ),
    ValidationCase(
        name="제주 숙박비 계산",
        question="제주 2박 3일 출장에서 숙박비가 5만2천 원, 4만7천 원이면 얼마를 받을 수 있나요?",
        answer_keywords=("9만9천원",),
        expected_source_pages=(("공무원여비100문100답.pdf", 31),),
    ),
    ValidationCase(
        name="근무지 내 출장비",
        question="근무지 내 국내출장을 4시간 이상 하면 여비는 얼마인가요?",
        answer_keywords=("2만원",),
        expected_source_pages=(("2024년 9장 공무원여비업무 처리기준.pdf", 41),),
    ),
    ValidationCase(
        name="문서 밖 질문 거부",
        question="파이썬으로 웹사이트를 만드는 방법을 알려줘.",
        must_abstain=True,
    ),
)


def build_context(retrieved_documents: list[tuple[Document, float]]) -> str:
    """검색 결과를 답변 체인에 넣을 문맥으로 구성합니다."""
    return "\n\n".join(
        "[출처: %s p.%s]\n%s"
        % (document.metadata.get("source"), document.metadata.get("page"), document.page_content)
        for document, _score in retrieved_documents
    )


def run_validation(vector_store, answer_chain) -> list[dict[str, object]]:
    """대표 질문을 실행하고 검색·답변·출처 검증 결과를 반환합니다."""
    results: list[dict[str, object]] = []

    for case in VALIDATION_CASES:
        retrieved_documents = vector_store.similarity_search_with_score(case.question, k=8)
        answer = str(
            answer_chain.invoke(
                {
                    "context": build_context(retrieved_documents),
                    "question": case.question,
                }
            )
        )

        sources = {
            (str(document.metadata.get("source")), int(document.metadata.get("page", 0)))
            for document, _score in retrieved_documents
        }
        source_ok = not case.expected_source_pages or any(
            expected_page in sources
            for expected_page in case.expected_source_pages
        )
        answer_lower = answer.replace(" ", "")
        keywords_ok = all(keyword.replace(" ", "") in answer_lower for keyword in case.answer_keywords)
        abstention_ok = ("확인할 수 없습니다" in answer) if case.must_abstain else True
        passed = keywords_ok and source_ok and abstention_ok

        results.append(
            {
                "name": case.name,
                "question": case.question,
                "answer": answer,
                "sources": sorted(sources),
                "keywords_ok": keywords_ok,
                "source_ok": source_ok,
                "abstention_ok": abstention_ok,
                "passed": passed,
            }
        )

    return results
