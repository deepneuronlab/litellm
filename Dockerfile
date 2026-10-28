# Base image for building
ARG LITELLM_BUILD_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:31da6565f35af6401031c1d7aa91dc84ac76c5c48edd17fb90f0ed9e3173c7a9

# Runtime image
ARG LITELLM_RUNTIME_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:31da6565f35af6401031c1d7aa91dc84ac76c5c48edd17fb90f0ed9e3173c7a9
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a

FROM $UV_IMAGE AS uvbin

# Current Wolfi Node packages require a newer glibc than the pinned base.
FROM $LITELLM_BUILD_IMAGE AS nodebin
RUN apk add --no-cache curl xz
RUN set -eu; \
    case "$(uname -m)" in \
        x86_64) arch=x64; sha=6170b2d474e1c626f43cff3adc118df9fe3b573258a4fc3d850c8dbc6b079bbd ;; \
        aarch64) arch=arm64; sha=42aa04922591ee5d4dbc7740d452621acf28508239daa512f5ee8c47ee224fdb ;; \
        *) echo "Unsupported Node architecture" >&2; exit 1 ;; \
    esac; \
    archive="node-v26.12.1-linux-${arch}.tar.xz"; \
    curl --fail --show-error --location --proto '=https' --retry 3 \
        -o "/tmp/${archive}" "https://nodejs.org/dist/v26.12.1/${archive}"; \
    printf '%s  %s\n' "$sha" "/tmp/${archive}" | sha256sum -c -; \
    mkdir -p /opt/node; \
    tar -xJf "/tmp/${archive}" -C /opt/node --strip-components=1; \
    rm "/tmp/${archive}"

# Builder stage
FROM $LITELLM_BUILD_IMAGE AS builder

WORKDIR /app
USER root

COPY --from=uvbin /uv /usr/local/bin/uv
COPY --from=uvbin /uvx /usr/local/bin/uvx
COPY --from=nodebin /opt/node /opt/node

RUN apk add --no-cache \
    bash \
    gcc \
    libatomic \
    openssl \
    openssl-dev \
    libsndfile

ARG UV_PYTHON_VERSION=3.13.13
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:/opt/node/bin:${PATH}"

RUN uv python install --no-bin "$UV_PYTHON_VERSION"

# Copy dependency metadata first for layer caching
COPY pyproject.toml uv.lock ./
COPY enterprise/pyproject.toml enterprise/
COPY litellm-proxy-extras/pyproject.toml litellm-proxy-extras/

# Install third-party dependencies (cached unless pyproject.toml/uv.lock change)
RUN uv sync --frozen --no-install-project --no-install-workspace --no-default-groups --no-editable \
    --extra proxy \
    --extra proxy-runtime \
    --extra extra_proxy \
    --extra semantic-router \
    --python "$(uv python find "$UV_PYTHON_VERSION")"

# Copy full source tree
COPY . .

# Build Admin UI before final sync
RUN sed -i 's/\r$//' docker/build_admin_ui.sh && chmod +x docker/build_admin_ui.sh && ./docker/build_admin_ui.sh

# Install project and workspace packages (fast - deps already cached)
RUN uv sync --frozen --no-default-groups --no-editable \
    --extra proxy \
    --extra proxy-runtime \
    --extra extra_proxy \
    --extra semantic-router \
    --python "$(uv python find "$UV_PYTHON_VERSION")"

RUN prisma generate --schema=./schema.prisma

RUN sed -i 's/\r$//' docker/entrypoint.sh && chmod +x docker/entrypoint.sh && \
    sed -i 's/\r$//' docker/prod_entrypoint.sh && chmod +x docker/prod_entrypoint.sh

# Runtime stage
FROM $LITELLM_RUNTIME_IMAGE AS runtime

USER root
COPY --from=nodebin /opt/node /opt/node
ENV PATH="/opt/node/bin:${PATH}"

RUN apk add --no-cache bash openssl tzdata libsndfile libatomic libstdc++ && \
    npm install -g npm@11.14.0 tar@7.5.11 glob@13.0.6 @isaacs/brace-expansion@5.0.1 brace-expansion@5.0.5 minimatch@10.2.4 diff@8.0.3 picomatch@4.0.4 && \
    GLOBAL="$(npm root -g)" && \
    for pkg in tar glob @isaacs/brace-expansion brace-expansion minimatch diff picomatch; do \
        name="${pkg##*/}"; \
        find "$GLOBAL/npm" -type d -name "$name" -path "*/node_modules/$pkg" | while read d; do \
            rm -rf "$d" && cp -rL "$GLOBAL/$pkg" "$d"; \
        done; \
    done && \
    npm cache clean --force && \
    test "$(npm --version)" = 11.14.0 && \
    rm -rf /opt/node/lib/node_modules/npm /opt/node/bin/npm /opt/node/bin/npx

WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}"

COPY --from=builder /app /app
COPY --from=builder /opt/uv-python /opt/uv-python
RUN /app/.venv/bin/python -c \
    "import math, subprocess, sys; from pathlib import Path; assert sys.version_info[:2] == (3, 13), sys.version; assert sys.prefix == '/app/.venv', sys.prefix; assert Path(sys.base_prefix).is_relative_to('/opt/uv-python'), sys.base_prefix; assert math.sqrt(4) == 2; subprocess.run([sys.executable, '-c', 'import math'], check=True)"
# Prisma binaries live in $HOME/.cache (default prisma-python location),
# which is /root/.cache here. Copy only the Prisma subdirs — copying the
# whole /root/.cache drags in the uv build cache (~660 MB, includes a
# setuptools wheel that surfaces as a CVE finding even though it's not
# on the runtime sys.path).
COPY --from=builder /root/.cache/prisma /root/.cache/prisma
COPY --from=builder /root/.cache/prisma-python /root/.cache/prisma-python

RUN test "$(node --version)" = v26.12.1 && \
    prisma --version

RUN find /app/.venv -type f -path "*/tornado/test/*" -delete && \
    find /app/.venv -type d -path "*/tornado/test" -delete

EXPOSE 4000/tcp

ENTRYPOINT ["docker/prod_entrypoint.sh"]
CMD ["--port", "4000"]
