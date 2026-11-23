# The CI runner image: GitHub's actions-runner, pinned by digest, plus the C toolchain go test -race
# needs (cgo) and the few tools the jobs call that the base image lacks. Build it, push it to the
# registry the cluster pulls from, and put the pushed digest in both values files (README.md).
FROM ghcr.io/actions/actions-runner:2.338.1@sha256:e275409999c37162649aa050ca573f03eab2eabcb3a1e7fcc47548de6b4f9690
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev git jq procps xz-utils zstd unzip ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
USER runner
