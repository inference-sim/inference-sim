package engine

// RenderServing is implemented by an engine whose model server answers the router's render
// endpoints (/v1/completions/render and /v1/chat/completions/render), so that a simulator
// serving the router's HTTP calls answers as the real server does (inference-sim#2129).
//
// vLLM serves them only when the server runs with --enable-scale-out or in a dedicated
// token-only mode; otherwise every render call returns HTTP 403 PermissionDeniedError, and the
// router's token producer counts a render failure and goes on without token IDs. The fact
// belongs to the engine, so the simulator asks for it and does not decide it. An engine that
// does not implement RenderServing is taken to serve the endpoints, as before this interface
// existed.
type RenderServing interface {
	ServesRender() bool
}
