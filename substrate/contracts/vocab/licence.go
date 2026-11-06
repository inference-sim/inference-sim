package vocab

// The licence allow-list of X12 and D6: the licence identifiers whose terms let BLIS commit
// and redistribute an artefact, given its notice. It is defined once, here, so that the
// catalog's config.json check, the data ledger and the measurement validator apply one list.
// Adding a licence is a review question for the ledger owners.
//
// An entry is an SPDX identifier, or LicenseRef-<name> for one licence text SPDX does not
// list. A LicenseRef entry names a text a person has read, at a pinned revision, and found to
// permit redistribution; its comment gives that text's URL, its SHA-256 and the condition the
// text sets on redistribution. A generic name such as LicenseRef-other or a placeholder such
// as LicenseRef-none-stated is never an entry: it names no text.
//
// A licence that explicitly forbids redistribution is never added. A model under such a
// licence commits only its config.json's hash (model.yaml config.sha256) and revision, and
// the nightly fetch job derives its graph: the derived-only path. The catalog meets each
// entry's notice condition in catalog/licences/ and catalog/NOTICE, which validate-catalog
// checks.
var redistributableLicences = map[string]bool{
	"Apache-2.0":   true,
	"MIT":          true,
	"BSD-2-Clause": true,
	"BSD-3-Clause": true,
	"CC-BY-4.0":    true,
	"CC0-1.0":      true,

	// DeepSeek License Agreement v1.0,
	// https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/raw/604d5664dddd88a0433dbae533b7fe9472482de0/LICENSE,
	// sha256 ccfee4895df06bcab524151c278e8dde88bbe76165a24ecbcbcf9fafd71fd2b3 (DeepSeek-V3's
	// LICENSE-MODEL is the same bytes). Redistribution: give a copy of the licence, keep the
	// notices, pass on the use-based restrictions of paragraph 5.
	"LicenseRef-deepseek": true,
	// GLM-5.3 License,
	// https://huggingface.co/zai-org/GLM-5.3/raw/aca966e4e02791568aa6a4ced368624b3d897f42/LICENSE,
	// sha256 96e1622099fc9d6b70c9760f007d99e66d7497eec636b63c60fe208401e9170c. Redistribution:
	// keep the copyright notice and the licence.
	"LicenseRef-glm-5.3": true,
	// Kimi K3 License,
	// https://huggingface.co/moonshotai/Kimi-K3/raw/f831ab66814297da540d832a5235f8e904f29d06/LICENSE,
	// sha256 20c797ce19af0c17de52c6afb144644768a591c521655f5ebf5712c9850f2887. Redistribution:
	// keep the copyright notice and the licence.
	"LicenseRef-kimi-k3": true,
	// Llama 2 Community License Agreement,
	// https://raw.githubusercontent.com/meta-llama/llama/7c0a08ec24ce91880184ba9002227624262904a2/LICENSE,
	// sha256 f28d05612fee8ef09283d4e755c468080b13d1cd92c9ae1aa52c84b075799e23, with its
	// Acceptable Use Policy (USE_POLICY.md at the same commit). The Hugging Face repositories
	// are gated, so the text is read from Meta's own repository. Redistribution: give a copy
	// of the Agreement, and keep its attribution sentence in a "Notice" file.
	"LicenseRef-llama2": true,
	// Llama 3.1 Community License Agreement,
	// https://raw.githubusercontent.com/meta-llama/llama-models/1f0feb795a4130697ced243fb53051670d591653/models/llama3_1/LICENSE,
	// sha256 bf2eac60b81e5c5f779fc3b4849d3bfacfd63f97f3eb68b44e9c6868c5801712, with its
	// Acceptable Use Policy. Redistribution: give a copy of the Agreement, keep its
	// attribution sentence in a "Notice" file, display "Built with Llama" for a product.
	"LicenseRef-llama3.1": true,
	// Llama 4 Community License Agreement,
	// https://raw.githubusercontent.com/meta-llama/llama-models/5fdf83110cc9daa7435dfba6eb304892cc0041b8/models/llama4/LICENSE,
	// sha256 6b346865267ef58a4592287a99e9417178223c9300e9da361ce7801be6dea61f, with its
	// Acceptable Use Policy. Redistribution: as for Llama 3.1, with the Llama 4 sentence.
	"LicenseRef-llama4": true,
	// MiniMax Community License,
	// https://huggingface.co/MiniMaxAI/MiniMax-M3/raw/f0e1c1e04d40177e4673a22097036854f536e9c0/LICENSE,
	// sha256 b53f2fdda3049b0e9013207be51efc2d372cda1fcfdd8bb4bb8b22658ca5db9c. The grant is for
	// non-commercial purposes; commercial use needs a "Built with MiniMax M3" display and a
	// notice to, or authorisation from, MiniMax. Redistribution: keep the copyright notice
	// and the licence.
	"LicenseRef-minimax-community": true,
	// Two texts carry this name. Kimi K2.5 and K2.6's "Modified MIT License",
	// https://huggingface.co/moonshotai/Kimi-K2.5/raw/4d01dfe0332d63057c186e0b262165819efb6611/LICENSE,
	// sha256 c524b41658421b8b143c5b50637ef008909da56013ad12ccaba0536139ee35b0 (K2.6's differs
	// only in the model name, sha256 1d4ab2d8404287de44429024a78b58d3aabc4df9d0c490e73fec9ff1fd842529).
	// And MiniMax-M2.5, whose card names modified-mit but whose repository carries the
	// MINIMAX MODEL LICENSE,
	// https://huggingface.co/MiniMaxAI/MiniMax-M2.5/raw/f710177d938eff80b684d42c5aa84b382612f21f/LICENSE-MODEL,
	// sha256 278b29a14cf6b0cc60519617909af61fa4a6aca4ee928d6178cb9799785c5cc0. Redistribution:
	// keep the copyright notice and the licence; for MiniMax-M2.5, also its attribution
	// sentence in a "Notice" file.
	"LicenseRef-modified-mit": true,
	// MUSE MODEL LICENSE, version Muse Spark Open, which includes its Usage Policy as
	// Exhibit A,
	// https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B/raw/49cade7b0c6ed297f0434229ac67eea5f9ae1220/LICENSE.md,
	// sha256 458a5c57d8e622e395eb0e59799ba5bba1ea7524ffd120ee6e3018688d7d6ace. Redistribution:
	// include the copyright notice, the licence and the Usage Policy in all copies.
	"LicenseRef-muse-model-license": true,
	// The same MUSE MODEL LICENSE, under the name Inferact's card gives it,
	// https://huggingface.co/Inferact/Muse-Spark-Open-DSpark/raw/ffc3eb50c8ae7a5a391ddedc43085d2e5249bc35/LICENSE.md,
	// sha256 458a5c57d8e622e395eb0e59799ba5bba1ea7524ffd120ee6e3018688d7d6ace. Redistribution:
	// as LicenseRef-muse-model-license.
	"LicenseRef-muse-spark-open": true,
	// OpenMDW License Agreement 1.1,
	// https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/raw/a9904d24bcc1d289a1950fa9d2b978c47cf903b9/LICENSE,
	// sha256 d7c8a9e5d1896d0a9588319cc7b1433e64645ad6d9e55632c30b78d8c038c23b; the same text,
	// without NVIDIA's copyright line, is
	// https://raw.githubusercontent.com/OpenMDW/OpenMDW/9067ba54b2f84bc9e15a05734f570ec88e9f445d/1.1/LICENSE.OpenMDW-1.1,
	// sha256 2ab44b68365473c112f5092211a38f231cb23e50de68b75a13369adbd76a74df. SPDX lists only
	// OpenMDW-1.0. Redistribution: keep a copy of the agreement and the copyright notices.
	"LicenseRef-openmdw-1.1": true,
}

// RedistributableLicences returns the allow-list, sorted.
func RedistributableLicences() []string { return Enumerate(redistributableLicences) }

// LicenceRedistributable reports whether id is on the allow-list. The comparison is exact:
// "apache-2.0" or "Apache 2.0" is not an SPDX identifier and is refused.
func LicenceRedistributable(id string) bool { return redistributableLicences[id] }
