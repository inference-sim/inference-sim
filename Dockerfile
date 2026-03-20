# syntax=docker/dockerfile:1

# ── Build stage ────────────────────────────────────────────────────────────────
FROM golang:1.21-alpine AS builder

WORKDIR /src

# Cache dependencies separately from source so layer is reused on code-only changes.
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /blis main.go

# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM alpine:3.19

RUN apk add --no-cache ca-certificates tzdata

COPY --from=builder /blis /usr/local/bin/blis

ENTRYPOINT ["/usr/local/bin/blis"]
