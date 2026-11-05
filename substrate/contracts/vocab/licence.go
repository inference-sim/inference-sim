package vocab

// This file is a stand-in for task C's contracts/vocab/licence.go on branch m1/ledger, which
// defines the one licence allow-list of X12. It holds only the two functions the catalog's
// licence check calls, with the list C defines. When the branches merge, C's file replaces
// this one.

var redistributable = []string{"Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "CC-BY-4.0", "CC0-1.0", "MIT"}

// RedistributableLicences returns the SPDX identifiers of the licences on the X12 allow-list.
func RedistributableLicences() []string { return append([]string(nil), redistributable...) }

// LicenceRedistributable reports whether a licence, by SPDX identifier, is on the allow-list.
func LicenceRedistributable(spdx string) bool {
	for _, l := range redistributable {
		if l == spdx {
			return true
		}
	}
	return false
}
