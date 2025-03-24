from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import requests
from typing import List

app = FastAPI()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

templates = Jinja2Templates(directory="templates")

def search_arxiv(title: str):
    url = f"http://export.arxiv.org/api/query?search_query=all:{title}&start=0&max_results=5"
    response = requests.get(url)
    if response.status_code != 200:
        return []

    import xml.etree.ElementTree as ET
    root = ET.fromstring(response.text)
    
    results = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        arxiv_title = entry.find("{http://www.w3.org/2005/Atom}title").text
        arxiv_abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        results.append({"title": arxiv_title, "abstract": arxiv_abstract})
    
    return results

class QueryModel(BaseModel):
    title: str
    abstract: str

@app.post("/search")
def get_similar_papers(query: QueryModel):
    papers = search_arxiv(query.title)

    input_embedding = model.encode(query.abstract, convert_to_tensor=True)
    for paper in papers:
        paper_embedding = model.encode(paper["abstract"], convert_to_tensor=True)
        paper["similarity"] = util.pytorch_cos_sim(input_embedding, paper_embedding).item()

    sorted_papers = sorted(papers, key=lambda x: x["similarity"], reverse=True)
    
    return {"results": sorted_papers}

@app.get("/")
def form(request: Request):
    return templates.TemplateResponse("index.html.j2", {"request": request})
