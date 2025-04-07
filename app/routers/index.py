# routers/home.py
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/")
def form(request: Request):
    return templates.TemplateResponse("index.html.j2", {"request": request})