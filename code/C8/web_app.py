"""
宠物猫养护 RAG 系统 - Streamlit 网页版
运行方式：在 code/C8 目录下执行 streamlit run web_app.py
"""

import sys
from pathlib import Path

# 确保从 C8 目录运行，使相对路径正确
C8_DIR = Path(__file__).resolve().parent
if str(C8_DIR) not in sys.path:
    sys.path.insert(0, str(C8_DIR))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(C8_DIR / ".env" if (C8_DIR / ".env").exists() else None)


@st.cache_resource
def init_rag_system():
    """初始化 RAG 系统并构建知识库（仅执行一次）"""
    from main import CatCareRAGSystem

    with st.spinner("正在加载知识库，请稍候..."):
        rag = CatCareRAGSystem()
        rag.initialize_system()
        rag.build_knowledge_base()
    return rag


def main():
    st.set_page_config(
        page_title="宠物猫养护智能问答",
        page_icon="🐱",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # 侧边栏
    with st.sidebar:
        st.title("🐱 猫咪养护助手")
        st.markdown("---")
        st.markdown("**使用说明**")
        st.markdown("- 在下方输入框输入您关于猫咪养护的问题")
        st.markdown("- 支持流式输出，回答会逐字显示")
        st.markdown("- 本回答仅供参考，不能替代兽医诊断")
        st.markdown("---")
        use_stream = st.checkbox("使用流式输出", value=True, help="勾选后回答会逐字显示")

    # 主标题
    st.title("🐱 宠物猫养护智能问答")
    st.caption("基于 RAG 的猫咪养护知识问答，解答行为、健康、营养、护理与急救等问题")

    # 初始化 RAG 系统
    try:
        rag = init_rag_system()
    except Exception as e:
        st.error(f"系统初始化失败：{e}")
        st.info("请确保已设置 MOONSHOT_API_KEY 环境变量，且 data/C8/cat_care 数据目录存在。")
        return

    # 初始化对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=msg.get("avatar", None)):
            st.markdown(msg["content"])

    # 用户输入
    if prompt := st.chat_input("请输入您关于猫咪养护的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "🧑"})

        # 显示用户消息
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        # 生成回答
        with st.chat_message("assistant", avatar="🐱"):
            try:
                if use_stream:
                    response_container = st.empty()
                    full_response = ""
                    for chunk in rag.ask_question(prompt, stream=True):
                        full_response += chunk
                        response_container.markdown(full_response + "▌")
                    response_container.markdown(full_response)
                else:
                    full_response = rag.ask_question(prompt, stream=False)
                    st.markdown(full_response)
            except Exception as e:
                full_response = f"抱歉，处理您的问题时出错：{e}"
                st.error(full_response)

        # 添加助手消息到历史
        st.session_state.messages.append({"role": "assistant", "content": full_response, "avatar": "🐱"})


if __name__ == "__main__":
    main()
