FROM python:3.12-slim-bookworm AS builder

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        swig \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

COPY requirements.txt pyproject.toml ./

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        --no-cache-dir \
        -r requirements.txt

COPY src ./src

RUN python -m pip install \
    --no-cache-dir \
    --no-deps \
    .


FROM builder AS test

COPY requirements-dev.txt ./

RUN python -m pip install \
    --no-cache-dir \
    -r requirements-dev.txt

COPY tests ./tests

CMD ["python", "-m", "pytest", "-v"]


FROM python:3.12-slim-bookworm AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

RUN useradd \
        --create-home \
        --shell /usr/sbin/nologin \
        appuser \
    && mkdir -p /app/reports \
    && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["python", "-m", "rocket_landing"]

CMD ["--help"]