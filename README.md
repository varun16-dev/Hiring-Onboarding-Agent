# Hiring & Onboarding Agent

## About the Project
The **Hiring & Onboarding Agent** is an AI-powered recruitment assistant designed to streamline the hiring process and assist with employee onboarding. It acts as an intelligent assistant capable of answering questions, screening candidates, and managing interview schedules. 

The agent operates through a Gradio-based user interface and is driven by an underlying Large Language Model (LLM). To maintain high accuracy and relevance, the system is designed with strict data separation across its core functionalities.

The agent features three entirely distinct workflows:
1. **Candidate FAQ**: Answers candidate or employee questions regarding company policies, employee benefits, leave, holidays, working hours, insurance, etc., by referencing a company handbook.
2. **Resume Screening**: Analyzes a candidate's resume against a job description to determine matching skills, missing skills, and overall candidate qualifications/experience.
3. **Interview Scheduling**: Manages available interview slots, allowing users to query dates and times and book interviews via a mock calendar system.

## System Architecture
The system is built using modern AI and framework technologies, including **LangChain**, **Ollama**, **FAISS**, and **Gradio**. 

The architecture is broken down into the following key components:
- **LLM Engine (`demo.py`)**: Uses `ChatOllama` to instantiate the core language model (e.g., `qwen3:0.6b`).
- **Retrieval-Augmented Generation (RAG) (`rag.py`)**: Responsible for document processing. It ingests PDF documents (`company_handbook.pdf`, `job_description.pdf`, `resume.pdf`), chunks the text using `RecursiveCharacterTextSplitter`, creates embeddings using `OllamaEmbeddings`, and stores them locally in a `FAISS` vector store. It also defines custom retrievers for specific document types (policy, resume, jd).
- **Custom Tools (`tool_calls.py`)**: Implements the logic for specific actions such as `resume_to_jd_match` (screening), `interview_slot_lookup` (calendar lookup via `interview_slots.json`), and `book_interview_slot`.
- **Agent Orchestrator (`agent.py`)**: Connects the LLM, the tools, and the RAG pipelines using LangChain's agent framework. It uses a strict system prompt to ensure that the agent uses the correct tool and data source for the requested workflow (e.g., policy queries only search the handbook, screening only uses the resume and JD).
- **User Interface**: Provided by `Gradio`, allowing users to interact with the agent through a conversational markdown-supported chat interface.

### System Architecture Diagram

```mermaid
graph TB
    subgraph "User Interface Layer"
        UI[Gradio Web Interface]
    end
    
    subgraph "Agent Layer"
        AGENT[LangChain Agent Orchestrator]
        PROMPT[System Prompt]
    end
    
    subgraph "Tools Layer"
        TOOL1[candidate_policy_faq]
        TOOL2[resume_to_jd_match]
        TOOL3[interview_slot_lookup]
        TOOL4[book_interview_slot]
    end
    
    subgraph "RAG Layer"
        RAG[RAG Pipeline]
        SPLITTER[RecursiveCharacterTextSplitter]
        EMBED[OllamaEmbeddings]
        FAISS[FAISS Vector Store]
        RETRIEVER1[Policy Retriever]
        RETRIEVER2[Resume Retriever]
        RETRIEVER3[JD Retriever]
    end
    
    subgraph "LLM Layer"
        LLM[ChatOllama qwen3:0.6b]
    end
    
    subgraph "Data Layer"
        PDF1[company_handbook.pdf]
        PDF2[job_description.pdf]
        PDF3[resume.pdf]
        JSON[interview_slots.json]
    end
    
    UI --> AGENT
    AGENT --> PROMPT
    AGENT --> TOOL1
    AGENT --> TOOL2
    AGENT --> TOOL3
    AGENT --> TOOL4
    
    TOOL1 --> RAG
    TOOL2 --> RAG
    RAG --> SPLITTER
    SPLITTER --> EMBED
    EMBED --> FAISS
    FAISS --> RETRIEVER1
    FAISS --> RETRIEVER2
    FAISS --> RETRIEVER3
    
    RETRIEVER1 --> TOOL1
    RETRIEVER2 --> TOOL2
    RETRIEVER3 --> TOOL2
    
    TOOL1 --> LLM
    TOOL2 --> LLM
    AGENT --> LLM
    
    TOOL3 --> JSON
    TOOL4 --> JSON
    
    PDF1 --> RAG
    PDF2 --> RAG
    PDF3 --> RAG
    
    style UI fill:#e1f5ff
    style AGENT fill:#fff4e1
    style LLM fill:#ffe1f5
    style FAISS fill:#e1ffe1
    style JSON fill:#f5e1ff
```

### Workflow Diagram

```mermaid
graph LR
    subgraph "User Input"
        USER[User Query]
    end
    
    subgraph "Agent Decision"
        AGENT[LangChain Agent]
        DECISION{Query Type}
    end
    
    subgraph "Workflow 1: Candidate FAQ"
        W1[Policy Question]
        TOOL1[candidate_policy_faq]
        RAG1[RAG - Policy Retriever]
        PDF1[company_handbook.pdf]
        LLM1[LLM Processing]
        RESP1[Policy Answer]
    end
    
    subgraph "Workflow 2: Resume Screening"
        W2[Screening Request]
        TOOL2[resume_to_jd_match]
        RAG2[RAG - Resume + JD Retrievers]
        PDF2[job_description.pdf]
        PDF3[resume.pdf]
        LLM2[LLM Comparison]
        RESP2[Screening Results]
    end
    
    subgraph "Workflow 3: Interview Scheduling"
        W3[Scheduling Request]
        TOOL3[interview_slot_lookup]
        TOOL4[book_interview_slot]
        JSON[interview_slots.json]
        RESP3[Schedule Response]
    end
    
    subgraph "Response"
        OUTPUT[Agent Response]
    end
    
    USER --> AGENT
    AGENT --> DECISION
    
    DECISION -->|Policy/Benefits| W1
    DECISION -->|Resume Screening| W2
    DECISION -->|Interview Scheduling| W3
    
    W1 --> TOOL1
    TOOL1 --> RAG1
    RAG1 --> PDF1
    RAG1 --> LLM1
    LLM1 --> RESP1
    
    W2 --> TOOL2
    TOOL2 --> RAG2
    RAG2 --> PDF2
    RAG2 --> PDF3
    RAG2 --> LLM2
    LLM2 --> RESP2
    
    W3 --> TOOL3
    W3 --> TOOL4
    TOOL3 --> JSON
    TOOL4 --> JSON
    JSON --> RESP3
    
    RESP1 --> OUTPUT
    RESP2 --> OUTPUT
    RESP3 --> OUTPUT
    
    style USER fill:#e1f5ff
    style AGENT fill:#fff4e1
    style W1 fill:#ffe1f5
    style W2 fill:#e1ffe1
    style W3 fill:#f5e1ff
    style OUTPUT fill:#e1f5ff
```

## Conclusion & Usefulness```

## Conclusion & Usefulness
This project is highly useful for HR departments, recruiters, and hiring managers as it automates several time-consuming administrative tasks. 

By employing an agentic AI approach with RAG capabilities, the system provides several key benefits:
* **Efficiency**: It can instantly extract relevant policy information without requiring an HR representative to search through a lengthy handbook.
* **Objective Screening**: The agent provides rapid, unbiased comparisons between a candidate's resume and a job description to quickly surface matched or missing skills.
* **Automated Scheduling**: Eliminates the back-and-forth emails usually required for scheduling interviews by handling slot lookups and bookings natively.
* **Data Privacy/Separation**: The architecture's strict separation of concerns ensures that candidate screening data isn't accidentally mixed with company policy data, leading to a highly reliable assistant.
