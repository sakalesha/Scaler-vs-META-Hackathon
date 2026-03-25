FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    openenv-core \
    fastapi \
    uvicorn \
    pydantic

EXPOSE 8000

CMD ["uvicorn", "src.envs.bug_fixer.server.server:app", \
     "--host", "0.0.0.0", "--port", "8000"]
