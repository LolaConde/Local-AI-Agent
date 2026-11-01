# Necessary imports
from io import StringIO
from time import sleep
import pandas as pd

from bs4 import BeautifulSoup
import requests
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.tools import FunctionTool
import json
from ddgs import DDGS
# Other implemented classes imports
from models.models_factory import LlamaIndexModelFactory

# Constants
CHUNK_OVERLAP = 300
CHUNK_SIZE = 750
TOP_K = 3

def _get_text_from_url(url: str) -> str:
    """
    Obtain the text content from a given URL.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return f"Error: Unable to fetch the URL. Status code: {response.status_code}"
    soup = BeautifulSoup(response.text, 'html.parser')
    # Delete script and style elements
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
    
    texto = soup.get_text()
    # Delete empty lines
    texto = '\n'.join(line.strip() for line in texto.splitlines() if line.strip())
    return texto

async def _divide_text_in_chunks(texto: str) -> list:
    """This function divides the text into chunks using a sentence splitter and then it generates embeddings for each chunk using the specified model."""
    pipeline = IngestionPipeline(
        transformations=[
                SentenceSplitter(chunk_overlap=CHUNK_OVERLAP, chunk_size=CHUNK_SIZE),
                LlamaIndexModelFactory().obtain_embeddings_model()
            ]
        )
    documento = Document(text=texto)
    nodes = await pipeline.arun(
        documents=[documento]
    )
    return list(nodes)

def _search_web(query: str) -> str:
    "Function that searches the web given the string query and returns the results as a string. If the same query is repeated, it will return the same results."
    # Wait 30s between searches to avoid being blocked by DuckDuckGo
    sleep(30)

    resultados = DDGS().text(
        query,
        backend="duckduckgo",
        max_results=5,
    )
    return json.dumps(resultados, ensure_ascii=False)

web_search_tool = FunctionTool.from_defaults(
    fn=_search_web,
    name="search_web",
    description=(
    "Searches the web and returns titles, URLs, and snippets. "
    "Use this tool to discover relevant sources."
    "After finding relevant URLs, extract relevant text fragments or/and tables from them before answering the question."
)
)

async def _get_texts_from_url(url:str, query:str)-> list[str]:
    "Function that gets the parts more relevant to the query from the url and returns them as a list of strings."
    # Acceder a la url y obtener el contenido de la página
    texto = _get_text_from_url(url)
    if texto.startswith("Error:"):
        return [texto]
    # Dividir el texto
    fragmentos = await _divide_text_in_chunks(texto)
    indice = VectorStoreIndex(
        nodes=fragmentos,
        embed_model=LlamaIndexModelFactory().obtain_embeddings_model(),
    )
    retriever = indice.as_retriever(
        similarity_top_k=TOP_K,
    )
    resultados = await retriever.aretrieve(query)
    return [
        resultado.node.get_content()
        for resultado in resultados
    ]

web_rag_tool = FunctionTool.from_defaults(
    async_fn=_get_texts_from_url,
    name="get_texts_from_url",
    description=(
        "Reads a web page and retrieves relevant text fragments. "
        "Takes the full URL and a query describing the information "
        "to find. Returns fragments of text from the page that are relevant to the query."
        "This tool does not scroll or advance through the page."
        "Repeated calls may return the same passages."
    ),
)

def _read_tables_from_url(url: str) -> str:
    "Function that reads tables from a given URL and returns them as a string."
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()

    tablas = pd.read_html(StringIO(response.text))

    resultados = []

    for numero, tabla in enumerate(tablas, start=1):
        if isinstance(tabla.columns, pd.MultiIndex):
            tabla.columns = [
                " | ".join(str(parte) for parte in columna)
                for columna in tabla.columns
            ]

        resultados.append(
            f"Fuente: {url}\n"
            f"Tabla {numero}\n"
            f"Filas: {len(tabla)}\n"
            f"{tabla.to_csv(index=False)}"
        )

    return "\n\n".join(resultados)

web_tables_tool = FunctionTool.from_defaults(
    fn=_read_tables_from_url,
    name="read_tables_from_url",
    description=(
        "Reads tables from a web page and returns them as a string. "
        "Takes the full URL of the page. Returns the tables found on the page."
    ),
)