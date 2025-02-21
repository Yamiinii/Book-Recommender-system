# Book Recommender System

![Uploading image.png…]()

## Project Setup

### 1. Initialize the Project
Set up a Python environment for the project.

### 2. Create a Virtual Environment
Use `venv` or `conda` to manage dependencies:

#### Using `venv`:
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate    # On Windows
```

#### Using `conda`:
```bash
conda create --name book-recommender python=3.9
conda activate book-recommender
```

### 3. Install Required Packages
Run the following command to install dependencies:
```bash
pip install kagglehub pandas matplotlib seaborn python-dotenv \
    langchain-community langchain-openai langchain-chroma gradio \
    transformers jupyter ipywidgets
```

### 4. Package Descriptions
| Package              | Description |
|----------------------|-------------|
| **kagglehub**        | Access and download datasets from Kaggle easily. |
| **pandas**           | Data manipulation and analysis library, useful for handling structured data. |
| **matplotlib**       | Visualization library for creating static, animated, and interactive plots. |
| **seaborn**          | Statistical data visualization library built on top of matplotlib. |
| **python-dotenv**    | Load environment variables from a `.env` file to manage secrets securely. |
| **langchain-community** | Community-supported extensions for working with LangChain. |
| **langchain-openai** | OpenAI API integration for LangChain applications. |
| **langchain-chroma** | ChromaDB integration for vector database storage and retrieval. |
| **gradio**           | Create interactive web interfaces for machine learning models easily. |
| **transformers**     | Hugging Face library for working with pre-trained transformer models. |
| **jupyter notebook** | Interactive computing environment for writing and running Python code. |
| **ipywidgets**       | Interactive widgets for Jupyter notebooks to enhance user experience. |

### 5. Running Jupyter Notebook
To start the Jupyter Notebook, run:
```bash
jupyter notebook
```

### 6. Data Processing and Vector Search
After cleaning the data, we will perform vector search and word embeddings to find similarities and dissimilarities between words. The process involves:
- Creating distance between words that are dissimilar.
- Relying on word embedding models by analyzing word usage in context.
- **Word2Vec**: Learning which words immediately precede and follow a given word.
- **Transforming words into embeddings** and adding positional embeddings to determine their position.
- Feeding these embeddings into a self-attention mechanism to understand word relationships within a sentence.
- Generating self-attention vectors for each word and averaging them over multiple iterations.
- This process of generating and normalizing self-attention vectors is called the **encoder block**.

  ![image](https://github.com/user-attachments/assets/63746ddf-203d-4c4c-932c-6bb05876adb8)
  ![image](https://github.com/user-attachments/assets/68e52662-ca2b-4b68-b47b-21b7e244cb0c)

### 7. Transformer Models and Language Translation
- **Encoder Block**: Learns all relationships between words in the source language.
- Sends output to the **Decoder**, which relates words in the target language and utilizes encoder output to predict the most likely translation.
- **Encoder-Only Models (e.g., RoBERTa)**: Trained to predict a masked word in text.
  - Tokenizes the text and adds special `[CLS]` and `[SEP]` tokens to mark the beginning and end.
  - Applies **word embeddings** and **self-attention in encoder blocks**.
  - Learns internal representations of language structure to improve accuracy.
  - 
![image](https://github.com/user-attachments/assets/97d3c29b-5402-4e76-a032-22f783e7c548)

### 8. Document Embedding and Search Optimization
- **Document Embedding**: Identifies whether documents are similar or dissimilar based on embeddings.
- We match embeddings to generate book recommendations.
- Currently using a **linear search** approach.
- Exploring **vector indexing databases** for grouping similar vectors efficiently.
- Tradeoff exists between **speed** and **accuracy** in search optimization.

### 9. LangChain for Advanced AI Pipelines
- **LangChain** is a Python framework offering various LLM functionalities.
- Used for **creating Retrieval-Augmented Generation (RAG) pipelines** and chatbots.
- **State-of-the-art AI capabilities** without being limited to a single LLM provider.

### 10. Zero-Shot Text Classification for Book Categorization
- Text classification is a branch of NLP that assigns text to categories.
- Zero-shot classification can categorize books into different groups without labeled training 
  data.
- Using Hugging Face’s transformers library, we apply zero-shot learning to classify books by 
  genre, topic, or audience.This step helps refine recommendations by filtering books based on   user preferences.
  
  ![image](https://github.com/user-attachments/assets/2b50e6f9-6cb3-41d4-bac0-23f4a8540e85)

### 11. Sentiment Analysis for Enhanced Control
- To provide users with an additional degree of control, we fine-tune our LLM to classify emotion.
- We consider the **RoBERTa** model with its encoder layer.
- Instead of predicting masked words, we replace the last layer with an emotion classification layer.
- This helps categorize books based on emotional tone, improving recommendations.
![image](https://github.com/user-attachments/assets/c242088c-ddd9-43a5-bec9-aa4c67cfa5ff)

### 12. Summary Vector Database and Gradio Dashboard
- We implement a **summary vector database** that allows us to retrieve the most similar texts based on queries.
- Text classification is used to determine if a book is **fiction** or **non-fiction**.
- After classification, we analyze the **emotional tone** of the book.
- We create an interactive **Gradio dashboard**, an open-source Python package, to visualize and explore recommendations dynamically.

Wrapped Gradio in FastAPI (required for Vercel).

---

## Contributing
Feel free to fork this repository and submit pull requests. Contributions are welcome!




