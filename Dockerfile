FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# code and the small offline fixture; real data and the LLM cache are mounted by docker-compose
COPY src ./src
COPY tests/fixtures ./tests/fixtures
COPY data/needs ./data/needs

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "src/trademesh/app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
