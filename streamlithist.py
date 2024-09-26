import streamlit as st
import os
import tempfile
import uuid
import requests
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models.gigachat import GigaChat
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.retrievers import ArxivRetriever
from langchain_community.document_loaders import Docx2txtLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain


os.environ['CURL_CA_BUNDLE'] = ''
load_dotenv()

# Создаём модель сбер
# @st.cache_resource
# def get_llm():
#     return GigaChat(auth_url='https://sm-auth-sd.prom-88-89-apps.ocp-geo.ocp.sigma.sbrf.ru/api/v2/oauth',
#     credentials=os.getenv('credentials'),
#     verify_ssl_certs=False)

# функции в отдельный файл переносить не стал, т.к. кэшстримлита становится муторным.
# Почти во всех функциях грузятся эмбеддинги\ллм.
# Создаём модель не сбер


@st.cache_resource
def get_llm():
    return GigaChat(credentials=os.getenv('credentials'), verify_ssl_certs=False)


# Эмбеддинги 
@st.cache_resource
def get_embeddings():
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
        )
    return embedding


# история
def create_conversational_rag_chain(retriever):
    llm = get_llm()
    if not llm:
        return None

    contextualize_q_system_prompt = (
        "Учитывая историю чата и последний вопрос пользователя, "
        "который может ссылаться на контекст в истории чата,"
        "сформулируйте отдельный вопрос, который можно понять "
        "без истории чата. НЕ отвечайте на вопрос - просто переформулируйте его,"
        "если нужно, а в противном случае верните как есть. Всегда отвечай на русском языке."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # "Вы являетесь помощником при выполнении заданий по поиску ответов на вопросы."
    # "Используйте приведенные ниже фрагменты из извлеченного контекста для ответа"
    # "на вопрос. Если вы не знаете ответа, скажите, что вы"
    # "не знаете."# Используйте максимум три предложения и старайтесь,"
    # "чтобы ответ был кратким. "
    # "\n\n"
    # "Что тебе делать: При получении вопроса и соответствующего контекста тщательно изучи предоставленную информацию, выяви ключевые факты и данные,"
    # "затем используй их для формирования точного ответа на заданный вопрос. Всегда отвечай на русском языке. Отвечай как можно более структурировано и развёрнуто."
    # "Кроме того, применяй критическое мышление для оценки достоверности и релевантности информации в контексте, задавая вопросы и проверяя предположения, которые могут влиять на твой ответ."
    # "Дополнительные техники: Используй технику цепочки мыслей для объяснения твоего рассуждения и логического процесса, приводящего к ответу."
    system_prompt = (
        "Твоя роль: Аналитик специализирующийся на быстром поиске информации в предоставленном контексте."
        "Краткая инструкция: Анализировать предложенный контекст и отвечать на вопросы, опираясь исключительно на информацию из этого контекста. Если информации в документе нет или не хватает, то так и пиши: информации недостаточно. Всегда отвечай на русском языке."
        "Твоя цель: Обеспечить абсолютно точный ответ, полностью основанный на информации из предложенного контекста, без внесения внешних данных или предположений."
        "Результат: Ответ должен быть четким и точным, содержать только информацию из предложенного контекста."
        "Ожидается, что ответ будет логически обоснованным и последовательным. Если ответа на вопрос нет в контексте, скажи об этом пользователю"
        "Ограничения: Отвечать можно только по контексту."
        
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # история чата
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain, store


# pdf ЗАГРУЖЕННЫЙ в стримлит и бд по нему vectore_store_st 
def extract_text_from_pdf(pdf_files, session_id):
    data = []
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_file_path = tmp_file.name
                
        # if tmp_file_path.endswith('.pdf'):
            # loader = PDFPlumberLoader(tmp_file_path)
        # elif tmp_file_path.endswith('.docx'):
        #     loader = Docx2txtLoader(tmp_file_path)
        loader = PDFPlumberLoader(tmp_file_path)
        # loader = PyPDFLoader(temp_file, extract_images=True) #если PDF в виде скана мб нужно боковой тогл добавить
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=300)
        splitted_data = text_splitter.split_documents(documents)
        data.extend(splitted_data)
        os.remove(tmp_file_path)
        # os.unlink(tmp_file_path)
        
    embeddings = get_embeddings()
    
    vectorstore_st = FAISS.from_documents(data, embeddings)
    retriever_st = vectorstore_st.as_retriever(search_kwargs={"k": 3}, search_type="mmr")
    return retriever_st


# загрузка и подготовка текста с arxiv
def extract_text_from_arxiv(query):
    retriever = ArxivRetriever(
        load_max_docs=2,
        get_ful_documents=True,
        top_k_results=4)
    docs_arxiv = retriever.invoke(query)
    docs = []
    meta = []
    for doc in docs_arxiv:
        pdf_path = doc.metadata["Entry ID"].replace('abs','pdf')
        response_load = requests.get(pdf_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(response_load.content)
                tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        # loader = PyPDFLoader(temp_file, extract_images=True) #если PDF в виде скана мб нужно боковой тогл добавить
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=450)
        splitted_data = text_splitter.split_documents(documents)
        for i in splitted_data:
                i.metadata["source"] = pdf_path
        docs.extend(splitted_data)
        meta.extend((f" ответ будет производиться по следующей статье {pdf_path}",f"опубликованной {doc.metadata['Published'].strftime("%Y-%m-%d")}"))
        os.remove(tmp_file_path)
    # os.unlink(tmp_file_path)

    embeddings = get_embeddings()

    vectorstore_arxiv = FAISS.from_documents(docs, embeddings)
    retriever_arxiv = vectorstore_arxiv.as_retriever(search_kwargs={"k": 3}, search_type="mmr")
    return retriever_arxiv, meta


# предзагрузка бд, очевидно не будет работать без самой бд - нужно создать.
# Функционал ответа по файлам и архиву работает без этой функции
# @st.cache_resource
def prep_bd(session_id):
    file_path = 'C:\Work\Rag\DB_merged'
    vector_store_loaded = FAISS.load_local(folder_path=file_path, embeddings=get_embeddings(), allow_dangerous_deserialization= True)
    loaded_retriever = vector_store_loaded.as_retriever(search_kwargs={'k': 3}, search_type="mmr")
    return loaded_retriever


# помощь с перефразировкой вопросов
def rephrase(user_input):
    llm = get_llm()
    prompt_template = PromptTemplate(
        input_variables=user_input,
        template="Перефразируй вопрос '{user_input}'и дай 3 похожих варианта без потери смысла первоначального вопроса"
        )   
    chain = LLMChain(llm=llm, prompt=prompt_template)

    resp = chain.invoke(user_input)
    return resp['text']


# главный экран для зазгрузки файлов и переключений между функциональными окнами по сешн стейту.
def main_screen():
    st.title("RAG-bot")
    st.sidebar.title("Меню")
    with st.sidebar:
        uploaded_file = st.file_uploader("Загрузите файлы", type="pdf", accept_multiple_files= True)
        button1 = st.button("Начать отвечать по документам")
        buttonarx = st.button("Q&A по arxiv.org")
        buttonbd = st.button('Отвечать по редзагруженной БД')

    st.write("Выберите режим работы.")

    if uploaded_file is not None and button1:
        with st.spinner("Обрабатываю pdf..."):
            session_id = str(uuid.uuid4())
            retriever = extract_text_from_pdf(uploaded_file,session_id)
            if retriever:
                st.session_state.retriever = retriever
                st.session_state.conversational_rag_chain, st.session_state.history_store = create_conversational_rag_chain(retriever)
                if st.session_state.conversational_rag_chain:
                    st.success("PDF загруженна")
                    st.session_state.page = "chat"
                    st.session_state.session_id = session_id
                    st.rerun()
                else:
                    st.error("Failed to create conversation chain. Please try again.")
            else:
                st.error("Failed to process the PDF. Please try another file.")
    
    if buttonbd:
        with st.spinner("Загружаю БД..."):
            session_id = str(uuid.uuid4())
            retrieverbd = prep_bd(session_id)
            if retrieverbd:
                st.session_state.retriever = retrieverbd
                st.session_state.conversational_rag_chain, st.session_state.history_store = create_conversational_rag_chain(retrieverbd)
                if st.session_state.conversational_rag_chain:
                    st.success("БД загруженна")
                    st.session_state.page = "bd"
                    st.session_state.session_id = session_id
                    st.rerun()
                else:
                    st.error("Failed to create conversation chain. Please try again.")
            else:
                st.error("Failed to process the BD. Please try another file.")

    if buttonarx:
        st.session_state.page = "arxiv"
        st.session_state.retriever = None
        st.session_state.chat_history = []
        st.session_state.conversational_rag_chain = None
        st.session_state.history_store = None
        st.session_state.session_id = None
        st.rerun()


# экран чата
def chat_screen():
    st.title("Вопросы по документам")
    st.sidebar.title("Меню")
    # st.sidebar.info(f"PDF: {st.session_state.pdf_name}")
    with st.sidebar:
        button2 = st.button('Вернуться на главный экран')
        butpha = st.button('Помоги с вопросом')
    if button2:
        st.session_state.page = "main"
        st.session_state.retriever = None
        st.session_state.chat_history = []
        st.session_state.conversational_rag_chain = None
        st.session_state.history_store = None
        st.session_state.session_id = None
        st.rerun()

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Задайте вопрос по своим pdf.")
    
    if butpha:
        with st.sidebar.expander("Перефразированный ввод"):
            # st.write(st.session_state.chat_history[-2]['content'])
            st.write(rephrase(user_input=st.session_state.chat_history[-2]['content']))

    # логика чата в последующих блоках идентична, нужно будет засунуть её в доп функцию
    if user_input and st.session_state.conversational_rag_chain:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                for chunk in st.session_state.conversational_rag_chain.stream(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": st.session_state.session_id}
                    },
                ):
                    # st.write(chunk)
                    if isinstance(chunk, dict):
                        content = chunk.get('answer') or chunk.get('text') or chunk.get('content') or ''
                        if content:
                            full_response += content
                            response_placeholder.markdown(full_response + "▌")
                    elif isinstance(chunk, str):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")

                if full_response:
                    response_placeholder.markdown(full_response)
                else:
                    response_placeholder.markdown("Не могу сгенерировать ответ.")
            except Exception as e:
                st.error(f"Ошибка возникла во время генерации ответа: {str(e)}")
                full_response = "Попробуйте ещё раз."
                response_placeholder.markdown(full_response)

        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        # st.write(st.session_state.chat_history)

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        if st.session_state.session_id in st.session_state.history_store:
            del st.session_state.history_store[st.session_state.session_id]
        st.rerun()


# экран архив
def arxiv():
    st.title("Вопросы по https://arxiv.org")
    st.sidebar.title("Меню")
    with st.sidebar:
        button2 = st.button('Вернуться на главный экран')
        query = st.text_input('Тематика статей(обязательно на английском)')
        button3 = st.button('Начать отвечать по данной тематике')
        butpha1 = st.button('Помоги с вопросом')
        # st.write(st.session_state.meta)
    if button2:
        st.session_state.page = "main"
        st.session_state.retriever = None
        st.session_state.chat_history = []
        st.session_state.conversational_rag_chain = None
        st.session_state.history_store = None
        st.session_state.session_id = None
        st.rerun()

    if query is not None and button3:
        with st.spinner("Собираю статьи..."):
                retriever, meta = extract_text_from_arxiv(query)
                meta = [i for i in meta]
                st.session_state.meta = meta
                if retriever:
                    st.session_state.retriever = retriever
                    st.session_state.conversational_rag_chain, st.session_state.history_store = create_conversational_rag_chain(retriever)
                    if st.session_state.conversational_rag_chain:
                        st.success("Статьи загруженны в vectore_store!")
    st.sidebar.info(st.session_state.meta)

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Задайте вопрос по статьям с arxiv.")

    if butpha1:
        with st.sidebar.expander("Перефразированный ввод"):
            # st.write(st.session_state.chat_history[-2]['content'])
            st.write(rephrase(user_input=st.session_state.chat_history[-2]['content']))

    if user_input and st.session_state.conversational_rag_chain:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                for chunk in st.session_state.conversational_rag_chain.stream(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": st.session_state.session_id}
                    },
                ):
                    # st.write(chunk)
                    if isinstance(chunk, dict):
                        content = chunk.get('answer') or chunk.get('text') or chunk.get('content') or ''
                        if content:
                            full_response += content
                            response_placeholder.markdown(full_response + "▌")
                    elif isinstance(chunk, str):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                
                if full_response:
                    response_placeholder.markdown(full_response)
                else:
                    response_placeholder.markdown("Не могу сгенерировать ответ.")
            except Exception as e:
                st.error(f"Ошибка возникла во время генерации ответа: {str(e)}")
                full_response = "Попробуйте ещё раз."
                response_placeholder.markdown(full_response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        # st.write(st.session_state.chat_history)

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        if st.session_state.session_id in st.session_state.history_store:
            del st.session_state.history_store[st.session_state.session_id]
        st.rerun()


# чат бд
def bd():
    st.title("Вопросы по предзагруженной БД")
    
    st.sidebar.title("Меню")
    with st.sidebar:
        button2 = st.button('Вернуться на главный экран')
        butpha2 = st.button('Помоги с вопросом')
    if button2:
        st.session_state.page = "main"
        st.session_state.retriever = None
        st.session_state.chat_history = []
        st.session_state.conversational_rag_chain = None
        st.session_state.history_store = None
        st.session_state.session_id = None
        st.rerun()

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Задайте вопрос по статьям с arxiv.")

    if butpha2:
        with st.sidebar.expander("Перефразированный ввод"):
            # st.write(st.session_state.chat_history[-2]['content'])
            st.write(rephrase(user_input=st.session_state.chat_history[-2]['content']))
    
    if user_input and st.session_state.conversational_rag_chain:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                for chunk in st.session_state.conversational_rag_chain.stream(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": st.session_state.session_id}
                    },
                ):
                    # st.write(chunk)
                    if isinstance(chunk, dict):
                        content = chunk.get('answer') or chunk.get('text') or chunk.get('content') or ''
                        if content:
                            full_response += content
                            response_placeholder.markdown(full_response + "▌")
                    elif isinstance(chunk, str):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                
                if full_response:
                    response_placeholder.markdown(full_response)
                else:
                    response_placeholder.markdown("Не могу сгенерировать ответ.")
            except Exception as e:
                st.error(f"Ошибка возникла во время генерации ответа: {str(e)}")
                full_response = "Попробуйте ещё раз."
                response_placeholder.markdown(full_response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        # st.write(st.session_state.chat_history)

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        if st.session_state.session_id in st.session_state.history_store:
            del st.session_state.history_store[st.session_state.session_id]
        st.rerun()


def main():
    st.set_page_config(page_title="RAG", page_icon="📚", layout="wide")

    if "page" not in st.session_state:
        st.session_state.page = "main"
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = ""
    if "conversational_rag_chain" not in st.session_state:
        st.session_state.conversational_rag_chain = None
    if "history_store" not in st.session_state:
        st.session_state.history_store = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "meta" not in st.session_state:
        st.session_state.meta = None

    if st.session_state.page == "main":
        main_screen()
    elif st.session_state.page == "chat":
        chat_screen()
    elif st.session_state.page == "arxiv":
        arxiv()
    elif st.session_state.page == 'bd':
        bd()


if __name__ == "__main__":
    main()
