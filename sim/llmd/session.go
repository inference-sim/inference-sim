package llmd

import (
	"slices"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/embed"
)

// The simulated client of a multi-turn session carries its session the way each session plugin reads
// it (embed.Requirements.SessionIdentities):
//
//   - session_id strategy, or the session-id producer, from a header: the session's identifier in that
//     header (x-session-id is Request.SessionID; any other name goes in Request.Headers);
//   - the producer from a cookie: the identifier as that cookie's value, in the cookie header;
//   - encoded_endpoint_header strategy: the router writes base64(pod) into the configured response
//     header (the plugins' util/sessionaffinity WriteResponseHeader, default x-session-token) and
//     the client sends it back on the session's next request (DecodePodName in the scorer). The client
//     keeps the latest value per session and header, as a client does that echoes a header or cookie
//     the server set; a request of no session keeps nothing and sends nothing.

// sessionHeaders are the further request headers the session's next request carries.
func (g *Gateway) sessionHeaders(session string) map[string]string {
	if session == "" {
		return nil
	}
	h := map[string]string{}
	var cookies []string
	for _, id := range g.req.SessionIdentities {
		switch {
		case id.Strategy == embed.SessionStrategyEncodedEndpoint:
			if v := g.echoed[session][id.Header]; v != "" {
				h[id.Header] = v
			}
		case id.Cookie != "":
			cookies = append(cookies, id.Cookie+"="+session)
		case id.Header != "":
			h[id.Header] = session
		}
	}
	if len(cookies) > 0 {
		slices.Sort(cookies)
		h["cookie"] = strings.Join(slices.Compact(cookies), "; ")
	}
	if len(h) == 0 {
		return nil
	}
	return h
}

// keepEchoed remembers, for the session, the token headers the router wrote on a response, to send them
// back on the session's next request.
func (g *Gateway) keepEchoed(session string, response map[string]string) {
	if session == "" {
		return
	}
	for _, id := range g.req.SessionIdentities {
		if id.Strategy != embed.SessionStrategyEncodedEndpoint {
			continue
		}
		if v, ok := response[id.Header]; ok {
			if g.echoed[session] == nil {
				g.echoed[session] = map[string]string{}
			}
			g.echoed[session][id.Header] = v
		}
	}
}
