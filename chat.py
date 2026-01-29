import streamlit as st
import llm  # llm.py 로직 가져오기

# 페이지 설정
st.set_page_config(page_title="Project RM", page_icon="⚖️", layout="wide")

st.title("⚖️ Project RM: 불공정 약관 심판관")
st.markdown(
    """
법적 기준(약관법, 분쟁해결기준)과 분석할 약관(넷플릭스, 카카오톡 등)을 업로드하세요.
RM이 위험한 조항을 찾아내어 법적 근거와 함께 판결해 드립니다.
"""
)

# ==========================================
# [사이드바] 관리자용: 문서 학습
# ==========================================
with st.sidebar:
    st.header("📚 지식 베이스 관리")
    st.info("궁금한 약관 PDF를 여기에 업로드하여 학습시키세요.")

    uploaded_files = st.file_uploader(
        "PDF 파일 업로드 (다중 선택 가능)", type=["pdf"], accept_multiple_files=True
    )

    # 업로드 버튼을 눌렀을 때만 동작하게 하려면 버튼 추가 가능 (여기선 자동 처리)
    if uploaded_files:
        if st.button("지식 베이스에 업로드 및 학습 시작"):
            with st.spinner("문서를 분석하고 Pinecone에 저장 중입니다..."):
                success, message = llm.embed_documents(uploaded_files)
                if success:
                    st.success(f"✅ 학습 완료! {message}")
                else:
                    st.error(f"❌ 학습 실패: {message}")

    st.divider()
    st.caption("Powered by LangChain & Pinecone")

# ==========================================
# [메인] 사용자용: 채팅 인터페이스
# ==========================================

# 1. 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "안녕하세요! RM입니다. 약관 파일들을 학습시키셨나요? 궁금한 점을 물어봐 주세요.",
        }
    ]

# 2. 이전 대화 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 3. 사용자 입력 처리
if user_input := st.chat_input("질문을 입력하세요 (예: 넷플릭스 환불 규정은 공정해?)"):
    # 사용자 메시지 표시 및 저장
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # 4. 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("법령과 약관을 대조하여 판결 중..."):
            try:
                # RAG 체인 가져오기
                qa_chain = llm.get_rag_chain()

                # 답변 요청
                response = qa_chain.invoke({"query": user_input})
                result_text = response["result"]

                st.write(result_text)

                # 답변 저장
                st.session_state.messages.append(
                    {"role": "assistant", "content": result_text}
                )

            except Exception as e:
                st.error(
                    f"죄송합니다. 답변을 생성하는 중 오류가 발생했습니다.\n\n오류 내용: {e}"
                )
