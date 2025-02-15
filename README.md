# Book Recommender System

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

### 6. Running the Application
To launch the Gradio web interface, execute:
```bash
python app.py  # Replace with the actual script name
```

---

## Contributing
Feel free to fork this repository and submit pull requests. Contributions are welcome!

## License
This project is licensed under the MIT License.


