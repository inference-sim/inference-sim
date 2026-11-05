// Package bot names the release bot's git identity, which the release bot commits and tags
// with and the commit-policy check recognises (X20: release-bot commits are signed and carry
// no sign-off).
package bot

const (
	Name  = "BLIS release bot"
	Email = "blis-release-bot@users.noreply.github.com"
)
