Here is a markdown description of every process in your notebook:

---

## Advanced RAG Pipeline with Google's Gemini 1.5 Pro, Llama Index, and Chroma DB

### Overview
This notebook demonstrates an advanced Retrieval-Augmented Generation (RAG) pipeline using Google's Gemini 1.5 Pro, Llama Index, and Chroma DB. The pipeline aims to identify research gaps within multiple research papers.

### 1. Setup and Initialization
**Purpose:** Import necessary libraries and set up the environment.
```python
import json
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from llama_index import StorageContext, load_index_from_storage
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.llm_predictor import LangchainLLMPredictor
from langchain.chat_models import ChatOpenAI
```

### 2. Load LLM and Set Up Predictors
**Purpose:** Initialize the language models for the pipeline.
```python
llm_predictor_chatgpt = LangchainLLMPredictor(llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0))
llm_predictor_gemini = LangchainLLMPredictor(llm=ChatOpenAI(model="gemini-1.5-pro", temperature=0))
```

### 3. Load Index from Storage
**Purpose:** Load the pre-built index from the storage for querying.
```python
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
retriever = VectorIndexRetriever(index=index)
query_engine = RetrieverQueryEngine(retriever=retriever)
```

### 4. Define Pipeline Run Function
**Purpose:** Define a function to run the pipeline and retrieve relevant information.
```python
def run_pipeline(query):
    response = query_engine.query(query)
    return response
```

### 5. Query the Research Gap
**Purpose:** Use the pipeline to identify research gaps from the provided query.
```python
query_str = "Identify research gaps in the provided papers."
output = run_pipeline(query_str)
print(output)
```

### 6. Process and Display Results
**Purpose:** Process the results from the query and display them in a readable format.
```python
response_data = json.loads(str(output))
for item in response_data:
    print(item['text'])
```

### 7. Example of Synthesizing Research Context
**Purpose:** Demonstrate how to synthesize and display research context using the pipeline.
```python
output = pipeline.run(topic="What is methodology of this research ?")
display(Markdown(str(output)))
```

---
