# inlumen-bench in a container. It drives the host's Docker engine through the
# mounted socket (ServiceCraft runs every submission in its own container), so
# the data and runs folders must be mounted at the same absolute paths as on the
# host. scripts/bench-docker does that.
FROM docker:27.3.1-cli AS docker-cli
FROM docker/buildx-bin:0.17.1 AS buildx

FROM python:3.12.7-slim-bookworm
COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=buildx /buildx /usr/local/libexec/docker/cli-plugins/docker-buildx
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    # Mounted datasets are owned by the host user, not by the container's root.
    && git config --system --add safe.directory '*'

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[evaluator]'

ENV INLUMEN_BENCH_IN_DOCKER=1 \
    INLUMEN_BENCH_TOOL_PYTHON=/usr/local/bin/python \
    INLUMEN_BENCH_CONFIG_DIR=/config \
    XDG_CACHE_HOME=/cache \
    PYTHONUNBUFFERED=1
ENTRYPOINT ["bench"]
CMD ["--help"]
