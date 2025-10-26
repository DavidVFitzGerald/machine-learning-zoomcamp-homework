FROM python:3.13.5-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY ".python-version" "pyproject.toml" "uv.lock" "./"

RUN uv sync --locked

COPY "05-deployment/predict.py" "05-deployment/model.bin" "./"

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]

