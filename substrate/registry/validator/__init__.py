"""Derivation / value-preservation tests for the BLIS coefficient registry.

Schema validation is owned by blis-schemas (the Go validator every consumer loads
these files through); the registry runs it in CI via the schema-validate job rather
than keeping a validator of its own. The tests here check that each committed value is
re-derivable from public data, not that a document has the right shape. See the README.
"""
