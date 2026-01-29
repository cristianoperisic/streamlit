import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import config


# 1. 문서 업로드 및 학습 함수
def embed_documents(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:
        try:
            # Streamlit 업로드 파일을 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # 메타데이터(출처) 추가
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name

            documents.extend(docs)
            os.remove(tmp_path)

        except Exception as e:
            return False, f"파일 처리 중 오류 발생: {uploaded_file.name} - {e}"

    if documents:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        split_texts = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)

        # [수정 포인트 1] pinecone_api_key 인자 제거 (환경변수 자동 인식)
        PineconeVectorStore.from_documents(
            documents=split_texts,
            embedding=embeddings,
            index_name=config.PINECONE_INDEX_NAME,
        )
        return (
            True,
            f"총 {len(split_texts)}개의 데이터가 데이터베이스에 저장되었습니다.",
        )

    return False, "처리할 텍스트가 없습니다."


# 2. 챗봇 엔진 (RAG Chain)
def get_rag_chain():
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)

    # [수정 포인트 2] pinecone_api_key 인자 제거
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=config.PINECONE_INDEX_NAME, embedding=embeddings
    )

    # 전문적인 톤의 프롬프트
    template = """
    당신은 법률 문서 분석을 지원하는 AI 어시스턴트입니다.
    제공된 [데이터베이스]의 내용을 바탕으로 사용자의 질문에 답변하십시오.
    
    [지시사항]
    1. 사용자의 질문에 대해 '데이터베이스에 있는 법적 기준(법령, 지침)'과 '대상 약관'을 비교하여 분석하십시오.
    2. 감정적인 표현이나 불필요한 수식어를 배제하고, 사실 위주로 건조하게 서술하십시오.
    3. 답변의 근거가 되는 조항이나 파일명을 명시하여 신뢰도를 높이십시오.
    
    [답변 양식]
    - 검토 결과: (문제 없음 / 검토 필요 / 위반 소지 있음)
    - 상세 분석: (법적 기준과 약관 내용을 대조하여 설명)
    - 참고 문헌: (정보 출처 파일명 표기)

    Context: {context}
    Question: {question}
    Answer:
    """
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    llm = ChatOpenAI(
        model_name="gpt-4o", temperature=0, openai_api_key=config.OPENAI_API_KEY
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT},
    )

    return qa_chain
