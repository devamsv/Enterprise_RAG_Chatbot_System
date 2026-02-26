# -*- coding: utf-8 -*-
"""
RAG Chatbot with Streamlit UI
A production-ready Retrieval-Augmented Generation chatbot.

Supports PDF, TXT, DOCX, and other document formats.
Uses OpenAI's GPT model for intelligent responses.
"""

import logging
import os
import tempfile
from typing import Optional, List

import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredFileLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fake import FakeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import config
from utils import (
    setup_logging,
    clean_temp_files,
    truncate_text,
    get_file_extension
)
# Setup logging
logger = setup_logging(config.LOG_LEVEL)

# Validate configuration early and fail fast in the UI
if not config.OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables")
    st.error("OPENAI_API_KEY not configured. Please set it in your .env file.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT
)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []


def load_document(file_path: str, file_type: str) -> Optional[List]:
    """
    Load document based on file type
    
    Args:
        file_path: Path to the document file
        file_type: File type (pdf, txt, docx, doc)
        
    Returns:
        List of documents or None if error
    """
    try:
        logger.info(f"Loading {file_type} document from {file_path}")
        
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
        elif file_type == "txt":
            # Try with UTF-8 encoding first, then fallback to latin-1
            try:
                loader = TextLoader(file_path, encoding="utf-8")
            except (UnicodeDecodeError, LookupError):
                logger.warning(f"UTF-8 failed for {file_path}, trying latin-1")
                loader = TextLoader(file_path, encoding="latin-1")
        elif file_type in ["docx", "doc"]:
            loader = Docx2txtLoader(file_path)
        else:
            loader = UnstructuredFileLoader(file_path)
        
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
        
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        st.error(f"Error loading file: {str(e)}")
        return None


def process_documents(uploaded_files: List) -> Optional[FAISS]:
    """
    Process uploaded documents and create vector store
    
    Args:
        uploaded_files: List of uploaded file objects
        
    Returns:
        FAISS vector store or None if error
    """
    all_docs = []
    temp_files = []
    
    try:
        with st.spinner("Processing documents..."):
            logger.info(f"Starting to process {len(uploaded_files)} files")
            
            for uploaded_file in uploaded_files:
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(
                        delete=False, 
                        suffix=f".{get_file_extension(uploaded_file.name)}"
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file.flush()
                        tmp_file_path = tmp_file.name
                        temp_files.append(tmp_file_path)
                    
                    # Get file type
                    file_type = get_file_extension(uploaded_file.name)
                    
                    # Load document
                    docs = load_document(tmp_file_path, file_type)
                    
                    if docs:
                        all_docs.extend(docs)
                        st.session_state.processed_files.append(uploaded_file.name)
                        logger.info(f"Added {len(docs)} documents from {uploaded_file.name}")
                
                except Exception as e:
                    logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    st.warning(f"Failed to process {uploaded_file.name}: {str(e)}")
                    continue
            
            if not all_docs:
                logger.error("No documents were successfully loaded")
                st.error("No documents were successfully loaded.")
                return None
            
            logger.info(f"Total documents loaded: {len(all_docs)}")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                length_function=len
            )
            chunks = text_splitter.split_documents(all_docs)
            logger.info(f"Split documents into {len(chunks)} chunks")
            
            # Initialize embedding model
            embedding_model = FakeEmbeddings(size=config.EMBEDDING_DIMENSION)
            
            # Create vector store using FAISS
            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embedding_model
            )
            logger.info("Vector store created successfully")
            
            return vectorstore
            
    except Exception as e:
        logger.error(f"Error in process_documents: {str(e)}")
        st.error(f"Error processing documents: {str(e)}")
        return None
    finally:
        # Clean up temp files
        clean_temp_files(temp_files)


def get_qa_chain(vectorstore: FAISS):
    """
    Create QA chain with retriever using LCEL
    
    Args:
        vectorstore: FAISS vector store
        
    Returns:
        QA chain with sources
    """
    try:
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type=config.RETRIEVAL_SEARCH_TYPE,
            search_kwargs={"k": config.RETRIEVAL_K}
        )
        
        # Initialize LLM (uses OPENAI_API_KEY from environment)
        llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=config.LLM_TEMPERATURE,
            timeout=config.LLM_TIMEOUT,
        )
        
        # Create prompt template
        prompt_template = """You are a helpful AI assistant that answers questions based only on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer the question using only the information from the context above
- If the answer is not in the context, say "I don't have enough information in the provided documents to answer this question."
- Be concise and accurate
- If relevant, quote specific parts from the context

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain using LCEL (LangChain Expression Language)
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        qa_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | PROMPT
            | llm
            | StrOutputParser()
        )
        
        # Return a wrapper that includes source documents
        class QAChainWithSources:
            """Wrapper to include source documents in response"""
            
            def __init__(self, chain, retriever):
                self.chain = chain
                self.retriever = retriever
            
            def invoke(self, inputs):
                """Execute the chain and get sources"""
                query = inputs.get("query") or inputs
                result = self.chain.invoke(query)
                source_documents = self.retriever.invoke(query)
                return {
                    "result": result,
                    "source_documents": source_documents
                }
        
        return QAChainWithSources(qa_chain, retriever)
        
    except Exception as e:
        logger.error(f"Error creating QA chain: {str(e)}")
        st.error(f"Error creating QA chain: {str(e)}")
        return None
# Streamlit UI
st.title(f"{config.PAGE_ICON} {config.PAGE_TITLE}")
st.markdown("Upload your documents and chat with them!")

# Sidebar for configuration
with st.sidebar:
    #st.header("⚙️ Configuration")
    
    st.markdown("---")

    # File upload
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=config.ALLOWED_FILE_TYPES,
        accept_multiple_files=True,
        help=f"Upload {', '.join(config.ALLOWED_FILE_TYPES).upper()} files"
    )
    
    # Process button
    if st.button("Process Documents", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            st.session_state.processed_files = []
            st.session_state.vectorstore = process_documents(uploaded_files)
            if st.session_state.vectorstore:
                st.success(f"✅ Processed {len(uploaded_files)} file(s) successfully!")
    
    # Display processed files
    if st.session_state.processed_files:
        st.markdown("---")
        st.header("📄 Processed Files")
        for i, file_name in enumerate(st.session_state.processed_files, 1):
            st.text(f"{i}. ✓ {file_name}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main chat interface
if st.session_state.vectorstore is None:
    st.info("👈 Please upload documents and click 'Process Documents' to start chatting.")
else:
    st.markdown("---")
    st.subheader("💬 Chat with your documents")
    st.caption("Type your question in the chat box at the bottom of the page.")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if question := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    logger.info(f"Processing query: {question[:50]}...")
                    
                    qa_chain = get_qa_chain(st.session_state.vectorstore)
                    if qa_chain is None:
                        raise Exception("Failed to create QA chain")
                    
                    response = qa_chain.invoke({"query": question})
                    answer = response['result']
                    
                    st.markdown(answer)
                    
                    # Show sources (optional)
                    if response['source_documents']:
                        with st.expander("📚 View Sources"):
                            for i, doc in enumerate(response['source_documents'], 1):
                                st.markdown(f"**Source {i}:**")
                                preview = truncate_text(
                                    doc.page_content,
                                    config.SOURCE_PREVIEW_LENGTH
                                )
                                st.text(preview)
                                st.markdown("---")
                    
                    # Add assistant response to chat
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })
                    logger.info("Query processed successfully")
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    logger.error(f"Query processing error: {str(e)}")
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p style='color: #888; font-size: 0.9em;'>
            Built with LangChain, Streamlit, and OpenAI
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
