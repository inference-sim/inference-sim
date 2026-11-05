package vocab

// The licence allow-list of X12 and D6: the SPDX identifiers whose terms let BLIS commit and
// redistribute an artefact, given its notice. It is defined once, here, so that the catalog's
// config.json check, the data ledger and the measurement validator apply one list. Adding a
// licence is a review question for the ledger owners.
var redistributableLicences = map[string]bool{
	"Apache-2.0":   true,
	"MIT":          true,
	"BSD-2-Clause": true,
	"BSD-3-Clause": true,
	"CC-BY-4.0":    true,
	"CC0-1.0":      true,
}

// RedistributableLicences returns the allow-list, sorted.
func RedistributableLicences() []string { return Enumerate(redistributableLicences) }

// LicenceRedistributable reports whether spdx is on the allow-list. The comparison is exact:
// "apache-2.0" or "Apache 2.0" is not an SPDX identifier and is refused.
func LicenceRedistributable(spdx string) bool { return redistributableLicences[spdx] }
