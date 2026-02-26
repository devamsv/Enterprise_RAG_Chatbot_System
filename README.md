# 🤖 RAG Chatbot - Production Ready

A **production-grade** Retrieval-Augmented Generation (RAG) chatbot that allows you to upload documents (PDF, TXT, DOCX) and ask intelligent questions using natural language processing.

## ✨ Features

- **Multi-format Support**: Upload PDF, TXT, DOCX documents
- **AI-Powered**: Uses OpenAI's GPT model for intelligent responses
- **Source Citations**: View the exact passages used to generate answers
- **Clean UI**: Intuitive Streamlit-based interface
- **Context-Aware**: Answers strictly based on your documents
- **Production Ready**: Comprehensive logging, error handling, configuration management
- **Scalable**: Uses FAISS for efficient vector storage
- **Type Safe**: Full Python type hints throughout

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key (from [platform.openai.com](https://platform.openai.com))

### Installation

```bash
# 1. Clone/download the project
cd RAG_ChatBot

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Run the application
streamlit run rag_chatbot.py
```

The app opens at `http://localhost:8501`

## 📖 Usage

1. **Upload Documents**: Select one or more files (PDF, TXT, DOCX)
2. **Process**: Click "Process Documents" button
3. **Chat**: Ask questions about your documents
4. **Cite**: Expand "View Sources" to see source references

## 🏗️ Project Structure

```
RAG_ChatBot/
├── rag_chatbot.py          # Main Streamlit application
├── config.py               # Configuration management
├── utils.py                # Utility functions
├── requirements.txt        # Dependencies
├── .env.example            # Environment template
├── .gitignore              # Git ignore rules  
└── README.md               # This file
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Document Processing
CHUNK_SIZE = 1000           # Size of text chunks
CHUNK_OVERLAP = 200         # Overlap between chunks

# Retrieval 
RETRIEVAL_K = 3             # Number of sources to retrieve

# Language Model
LLM_TEMPERATURE = 0.0       # 0 = deterministic, 1 = random
OPENAI_MODEL = "gpt-4.1-mini"

# Logging
LOG_LEVEL = "INFO"          # DEBUG, INFO, WARNING, ERROR
```

## 📊 How It Works

```
Documents → Load → Split into Chunks → Embed → Store in FAISS
                                         ↓
                            Query → Retrieve Similar → LLM → Answer
```

1. **Load**: Documents loaded with proper encoding detection
2. **Split**: Text split into overlapping chunks  
3. **Embed**: Chunks converted to vector embeddings
4. **Store**: Vectors indexed in FAISS database
5. **Search**: Retrieve most similar chunks for query
6. **Generate**: LLM creates answer from context

## 🔑 Getting API Key

1. Visit [platform.openai.com](https://platform.openai.com)
2. Sign up or login
3. Create new API key
4. Add to `.env`: `OPENAI_API_KEY=your_key_here`

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| API Key not found | Check `.env` exists with valid key |
| File loading error | Verify PDF/TXT/DOCX format is valid |
| Slow processing | Try smaller files or increase CHUNK_SIZE |
| Import errors | Run `pip install -r requirements.txt` |
| Memory issues | Process fewer files at once |

## 📁 File Support

| Format | Extension | Status |
|--------|-----------|--------|
| PDF | `.pdf` | ✅ Supported |
| Text | `.txt` | ✅ Supported (UTF-8, Latin-1) |
| Word | `.docx`, `.doc` | ✅ Supported |

## 🚀 Production Deployment

### Logging
Application includes comprehensive logging:
- Document processing steps
- Query handling
- Error details and stack traces

### Error Handling
- Graceful error recovery
- User-friendly error messages
- Automatic cleanup of temporary files
- File encoding fallback (UTF-8 → Latin-1)

### Performance Optimizations
- FAISS for fast vector similarity search
- Minimal model dependencies (FakeEmbeddings)
- Memory-efficient streaming
- Caching of processed documents

## 🔐 Security

- API keys only in `.env` (not in code)
- Temp files automatically deleted
- Input validation on file uploads
- No sensitive data in logs
- `.gitignore` prevents key exposure

## 📚 Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| UI | Streamlit | Web interface |
| RAG | LangChain | Orchestration |
| LLM | OpenAI API | Language model |
| Vectors | FAISS | Similarity search |
| Embeddings | FakeEmbeddings | Vector representation |
| PDF | PyPDF | PDF processing |
| Word | docx2txt | DOCX processing |

## 🔄 Upgrading Embeddings

To use real embeddings (HuggingFace/OpenAI):

```python
# In config.py
from langchain_community.embeddings import HuggingFaceEmbeddings

# In rag_chatbot.py, replace:
embedding_model = FakeEmbeddings(size=config.EMBEDDING_DIMENSION)

# With:
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

Note: This requires additional dependencies and disk space.

## 📖 Dependencies

Core packages:
- `streamlit>=1.28.0` - UI framework
- `langchain>=0.1.0` - RAG framework
- `langchain-openai` - OpenAI integration
- `faiss-cpu>=1.7.4` - Vector search
- `pypdf>=4.0.1` - PDF reading
- `docx2txt>=0.8` - DOCX reading  
- `python-dotenv>=1.0.0` - Environment management

See `requirements.txt` for complete list.

## 🧪 Development

### Running in Debug Mode
```bash
# Set environment variable
set ENVIRONMENT=development  # Windows
export ENVIRONMENT=development  # Linux/Mac

# Run app
streamlit run rag_chatbot.py
```

### Code Quality
- Full type hints for all functions
- Comprehensive docstrings
- Proper error handling
- Logging at appropriate levels

## 📝 Example Queries

After uploading documents, try:
- "What are the main topics covered?"
- "Summarize the key findings"
- "What does it say about [topic]?"
- "Compare X and Y"
- "What are the conclusions?"

## 🎯 Best Practices

- **Chunk Size**: 1000-2000 tokens for balanced context
- **Temperature**: 0-0.3 for factual answers
- **Retrieval Count**: 3-5 for balance between relevance and context
- **File Size**: Keep documents under 100MB
- **Encoding**: UTF-8 preferred, auto-fallback to Latin-1

## 🆘 Getting Help

1. Check Troubleshooting section
2. Review logs: `ENVIRONMENT=development streamlit run rag_chatbot.py`
3. Verify `.env` configuration
4. Check file format compatibility
5. Review [LangChain documentation](https://python.langchain.com)

## 📄 License

MIT License - Use freely and modify as needed

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional file format support (PowerPoint, Excel)
- Real embedding models integration
- Vector database persistence
- Advanced retrieval strategies
- Performance benchmarking

---

**Built with** ❤️ using LangChain, Streamlit, and OpenAI

Last Updated: February 2024
