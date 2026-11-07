// template_probe — checks the afm_cider_v2 prompt template against Apple's own token counter.
//
// Build:  swiftc -O -o template_probe template_probe.swift      (macOS 27, Apple Intelligence enabled)
// Run:    ./template_probe
//
// SystemLanguageModel.tokenCount(for: [Transcript.Entry]) counts the fully rendered prompt (it equals the prompt length
// the inference service logs as "asked to run N tokens"). Compare each line with the length of the same conversation
// rendered by the GGUF's chat template, e.g. `llama-server … --jinja` + POST /apply-template then /tokenize.
//   expected with the converter's template:  I,P = 71   P only (no instructions entry) = 16   I(''),P = 62   I,P,R,P2 = 86
import Foundation
import FoundationModels

let m = SystemLanguageModel.default
func T(_ s: String) -> Transcript.Segment { .text(Transcript.TextSegment(content: s)) }
func I(_ s: String) -> Transcript.Entry { .instructions(Transcript.Instructions(segments: s.isEmpty ? [] : [T(s)], toolDefinitions: [])) }
func P(_ s: String) -> Transcript.Entry { .prompt(Transcript.Prompt(segments: s.isEmpty ? [] : [T(s)])) }
func R(_ s: String) -> Transcript.Entry { .response(Transcript.Response(assetIDs: [], segments: s.isEmpty ? [] : [T(s)])) }
let DEV = "You are a helpful, concise assistant."
let U = "What is the capital of France?"
func show(_ name: String, _ f: () async throws -> Int) async {
    do { print(name, try await f()) } catch { print(name, "error:", error) }
}
await show("raw text U (BOS + tokens)") { try await m.tokenCount(for: U) }
await show("I,P") { try await m.tokenCount(for: [I(DEV), P(U)]) }
await show("P only") { try await m.tokenCount(for: [P(U)]) }
await show("I(''),P") { try await m.tokenCount(for: [I(""), P(U)]) }
await show("I only") { try await m.tokenCount(for: [I(DEV)]) }
await show("I,P,R,P2") { try await m.tokenCount(for: [I(DEV), P(U), R("Paris"), P("And of Spain?")]) }
