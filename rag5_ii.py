from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os


def create_rag_chain():
    """Create the RAG chain once and return it"""
    model = ChatOllama(model="llama3.1:8b")
    
    prompt = PromptTemplate.from_template(
        """
        <s> [Instructions] You are a friendly assistant. Answer the question based only on the following context. 
        If the question cannot be answered from the context, mention what is there in the context.
        [Instructions] Question: {input} 
        Context: {context} 
        Answer: [/Instructions]
        """
        #If you don't know the answer, then reply "No context available for this question: {input}". [/Instructions] </s> 
    )
    
    # Load vector store
    embedding = FastEmbedEmbeddings()
    
    # Check if vector store exists
    if not os.path.exists("./sql_chroma_db"):
        print("Error: Vector store not found! Please run ingest() first.")
        return None
    
    vector_store = Chroma(persist_directory="./sql_chroma_db", embedding_function=embedding)
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.1,
        },
    )
    
    # Create chains
    document_chain = create_stuff_documents_chain(model, prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    
    return chain

def ask(query: str, chain):
    """Ask a question using the provided chain"""
    if chain is None:
        print("Error: Chain not initialized!")
        return
    
    try:
        # Invoke chain
        result = chain.invoke({"input": query})
        
        # Print results
        print("\nAnswer:", result["answer"])
        
        # Print sources
        print("\nSources:")
        for i, doc in enumerate(result["context"]):
            source = doc.metadata.get("source", "Unknown")
            chunk_id = doc.metadata.get("chunk_id", i)
            print(f"  {i+1}. Source: {source}, Chunk: {chunk_id}")
            
    except Exception as e:
        print(f"Error processing query: {e}")

# Main execution
if __name__ == "__main__":
    
    print("Creating RAG chain...")
    chain = create_rag_chain()
    
    if chain:
        # Interactive Q&A loop
        print("\n=== RAG System Ready ===")
        print("Type 'quit' to exit")
        
        while True:
            question = input("\nQ?: ")
            if question.lower() in ['quit', 'exit', 'q']:
                break
            ask(question, chain)
    
        
        
        
