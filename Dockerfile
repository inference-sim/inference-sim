# syntax=docker/dockerfile:1

# ── Build stage ────────────────────────────────────────────────────────────────
# Keep base image Go version in sync with the 'go' directive in go.mod.
FROM golang:1.21-alpine AS builder

# TARGETOS/TARGETARCH are injected by Buildx for each platform in a multi-arch
# build (e.g. linux/amd64, linux/arm64). Declaring them as ARGs makes them
# available to the RUN command below so Go cross-compiles for the correct arch.
ARG TARGETOS=linux
ARG TARGETARCH=amd64

WORKDIR /src

# Cache dependencies separately from source so layer is reused on code-only changes.
COPY go.mod go.sum ./
RUN go mod download

COPY . .
# CGO_ENABLED=0: required for a fully static binary compatible with the musl-based
# Alpine runtime image. Without this, the binary links against glibc and crashes
# at startup inside the Alpine container.
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -trimpath -ldflags="-s -w" -o /blis main.go

# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM alpine:3.19

# ca-certificates: enables TLS verification for HTTPS calls made by blis at runtime.
# tzdata: enables timezone-aware time parsing (e.g. for workload scheduling windows).
RUN apk add --no-cache ca-certificates tzdata

# Run as a non-root user for container security best practice.
RUN adduser -D -u 10001 blis
USER blis

COPY --from=builder /blis /usr/local/bin/blis

ENTRYPOINT ["/usr/local/bin/blis"]
