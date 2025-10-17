import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os


@st.cache_resource
def create_rag_chain():
    """Create and cache the RAG chain"""
    model = ChatOllama(model="llama3.1:8b")
    
    prompt = PromptTemplate.from_template(
        """<s> [Instructions] You are a friendly assistant. Answer the question based only on the following context. 
        If the question cannot be answered from the context, mention what is there in the context.
        [Instructions] Question: {input} 
        Context: {context} 
        Answer: [/Instructions]"""
    )
    
    embedding = FastEmbedEmbeddings()
    
    if not os.path.exists("./sql_chroma_db"):
        st.error("Vector store not found! Please run ingest() first.")
        return None
    
    vector_store = Chroma(persist_directory="./sql_chroma_db", embedding_function=embedding)
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.2}
    )
    
    document_chain = create_stuff_documents_chain(model, prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    
    return chain


def main():
    st.set_page_config(page_title="RAG Q&A System", page_icon="🤖", layout="wide")
    st.title("🤖 RAG Q&A System")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Load RAG chain
    chain = create_rag_chain()
    
    if chain is None:
        st.stop()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources"):
                    st.text(message["sources"])
    
    # Chat input
    if query := st.chat_input("Ask a question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = chain.invoke({"input": query})
                    answer = result["answer"]
                    
                    # Format sources
                    sources = "\n".join([
                        f"{i+1}. Source: {doc.metadata.get('source', 'Unknown')}, "
                        f"Chunk: {doc.metadata.get('chunk_id', i)}"
                        for i, doc in enumerate(result["context"])
                    ])
                    
                    st.markdown(answer)
                    with st.expander("📚 View Sources"):
                        st.text(sources)
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Clear chat button in sidebar
    with st.sidebar:
        st.header("Controls")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        st.caption(f"Total messages: {len(st.session_state.messages)}")


if __name__ == "__main__":
    main()