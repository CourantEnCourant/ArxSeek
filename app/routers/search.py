# routers/search.py
from fastapi import APIRouter
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from accelerate import Accelerator
import requests
import xml.etree.ElementTree as ET

router = APIRouter()

accelerator = Accelerator()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
model.to(accelerator.device)

def search_arxiv(title: str):
    url = f"http://export.arxiv.org/api/query?search_query=all:{title}&start=0&max_results=30"
    response = requests.get(url)
    if response.status_code != 200:
        return []

    root = ET.fromstring(response.text)
    results = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        arxiv_title = entry.find("{http://www.w3.org/2005/Atom}title").text
        arxiv_abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        arxiv_id = entry.find("{http://www.w3.org/2005/Atom}id").text
        results.append({
            "title": arxiv_title,
            "abstract": arxiv_abstract,
            "url": arxiv_id
        })
    
    return results

class QueryModel(BaseModel):
    title: str
    abstract: str

@router.post("/search")
def get_similar_papers(query: QueryModel):
    papers = search_arxiv(query.title)
    input_embedding = model.encode(query.abstract, convert_to_tensor=True)
    for paper in papers:
        paper_embedding = model.encode(paper["abstract"], convert_to_tensor=True)
        paper["similarity"] = util.pytorch_cos_sim(input_embedding, paper_embedding).item()

    sorted_papers = sorted(papers, key=lambda x: x["similarity"], reverse=True)
    return {"results": sorted_papers}