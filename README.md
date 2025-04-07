# ArxSeek
An app to help scholars find related articles about their current article using AI. Includes a monthly subscription to use newest and coolest models

### By Cmd
```bash
uvicorn main:app --reload
```

### By Docker
```bash
docker build -t fastapi-arxivseek .
docker run -d -p 8000:8000 fastapi-arxivseek
```