// afm — CLI + OpenAI-compatible server over Apple's on-device Foundation Model.
//
// Build:  swiftc -O -o afm afm.swift
// Use:
//   ./afm "What is the capital of Japan?"          # one-shot
//   echo "prompt" | ./afm                          # piped
//   ./afm -s "You are a pirate." -t 0.8 "Hi"       # system prompt + temperature
//   ./afm --stream "Write a haiku about the sea"   # stream tokens
//   ./afm                                          # interactive REPL
//   ./afm --server 8080                            # OpenAI-compatible API on :8080
//        -> POST /v1/chat/completions  (works with any OpenAI client, e.g. `openai`, curl)
//
// This wraps Apple's real model (ANE routing and all). It is NOT a weight export;
// FoundationModels never exposes weights, so there is no GGUF path from here.
import FoundationModels
import Foundation
import Network

// ---- args ----
var args = Array(CommandLine.arguments.dropFirst())
func takeFlag(_ names: [String]) -> String? {
    for n in names { if let i = args.firstIndex(of: n), i+1 < args.count { let v = args[i+1]; args.removeSubrange(i...i+1); return v } }
    return nil
}
func hasFlag(_ names: [String]) -> Bool {
    for n in names { if let i = args.firstIndex(of: n) { args.remove(at: i); return true } }
    return false
}
let system = takeFlag(["-s","--system"]) ?? "You are a helpful, concise assistant."
let temp = Double(takeFlag(["-t","--temp"]) ?? "")
let maxTok = Int(takeFlag(["-m","--max"]) ?? "")
let stream = hasFlag(["--stream"])
let serverPort = takeFlag(["--server"])

let model = SystemLanguageModel.default
guard case .available = model.availability else {
    FileHandle.standardError.write("Model unavailable: \(model.availability)\n".data(using: .utf8)!)
    exit(1)
}

func makeOptions() -> GenerationOptions {
    var o = GenerationOptions()
    if let t = temp { o = GenerationOptions(temperature: t) }
    if let m = maxTok { o = GenerationOptions(temperature: temp, maximumResponseTokens: m) }
    return o
}

func respond(_ prompt: String, instructions: String, streaming: Bool) async -> String {
    let session = LanguageModelSession(instructions: instructions)
    do {
        if streaming {
            var full = ""
            for try await partial in session.streamResponse(to: prompt, options: makeOptions()) {
                let delta = String(partial.content.dropFirst(full.count))
                FileHandle.standardOutput.write(delta.data(using: .utf8)!)
                full = partial.content
            }
            print("")
            return full
        } else {
            return try await session.respond(to: prompt, options: makeOptions()).content
        }
    } catch { return "ERR: \(error)" }
}

// ---- OpenAI-compatible server ----
func startServer(_ port: UInt16) {
    let listener = try! NWListener(using: .tcp, on: NWEndpoint.Port(rawValue: port)!)
    print("afm server (Apple FoundationModels) on http://127.0.0.1:\(port)  ->  POST /v1/chat/completions")
    listener.newConnectionHandler = { conn in
        conn.start(queue: .global())
        conn.receive(minimumIncompleteLength: 1, maximumLength: 1<<20) { data, _, _, _ in
            guard let data = data, let req = String(data: data, encoding: .utf8) else { conn.cancel(); return }
            let body = req.components(separatedBy: "\r\n\r\n").dropFirst().joined(separator: "\r\n\r\n")
            var sys = system, userMsg = ""
            if let bd = body.data(using: .utf8),
               let j = try? JSONSerialization.jsonObject(with: bd) as? [String:Any],
               let msgs = j["messages"] as? [[String:Any]] {
                for m in msgs {
                    let role = m["role"] as? String ?? ""; let c = m["content"] as? String ?? ""
                    if role == "system" { sys = c } else if role == "user" { userMsg = c }
                }
            }
            Task {
                let reply = await respond(userMsg, instructions: sys, streaming: false)
                let resp: [String:Any] = [
                    "id":"afm-\(UUID().uuidString)","object":"chat.completion","model":"afmplus-v11.0-ifp",
                    "choices":[["index":0,"message":["role":"assistant","content":reply],"finish_reason":"stop"]]]
                let jd = try! JSONSerialization.data(withJSONObject: resp)
                let head = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: \(jd.count)\r\nConnection: close\r\n\r\n"
                conn.send(content: head.data(using:.utf8)! + jd, completion: .contentProcessed { _ in conn.cancel() })
            }
        }
    }
    listener.start(queue: .main)
    dispatchMain()
}

// ---- dispatch ----
if let p = serverPort, let port = UInt16(p) { startServer(port) }

let sem = DispatchSemaphore(value: 0)
if !args.isEmpty {                                   // one-shot from args
    let prompt = args.joined(separator: " ")
    Task { _ = await respond(prompt, instructions: system, streaming: stream || true); sem.signal() }
    sem.wait()
} else if isatty(fileno(stdin)) == 0 {               // piped
    let d = FileHandle.standardInput.readDataToEndOfFile()
    if let p = String(data: d, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines), !p.isEmpty {
        Task { _ = await respond(p, instructions: system, streaming: stream || true); sem.signal() }; sem.wait()
    }
} else {                                             // interactive
    print("afm REPL (Apple FoundationModels). Ctrl-D to quit.")
    while true {
        print("\n> ", terminator: ""); guard let line = readLine(), !line.isEmpty else { break }
        let s2 = DispatchSemaphore(value: 0)
        Task { _ = await respond(line, instructions: system, streaming: true); s2.signal() }; s2.wait()
    }
}
