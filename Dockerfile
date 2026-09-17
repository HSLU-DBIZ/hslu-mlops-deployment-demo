FROM python:3.12-slim

# uv installs exactly what uv.lock pins
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_NO_CACHE=1 PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home app
WORKDIR /app
RUN chown app:app /app
USER app

# dependencies first: this layer is rebuilt only when the lockfile changes
COPY --chown=app pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=app README.md ./
COPY --chown=app src ./src
COPY --chown=app data ./data
COPY --chown=app registry ./registry
RUN uv sync --frozen --no-dev

# no model binary in git: build the champion from the committed snapshot
RUN python -m lucerne_temp.training.train

# Cloud Run sets $PORT; 8080 is the default
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn lucerne_temp.inference.api:app --host 0.0.0.0 --port $PORT"]
