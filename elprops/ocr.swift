// Αναγνώριση κειμένου με το Vision του macOS. Χρήση: ocr <εικόνα>
// Τυπώνει JSON: [{"text", "x", "y", "w", "h", "conf"}], συντεταγμένες 0–1 με αρχή πάνω αριστερά.
import Foundation
import Vision
import AppKit

guard CommandLine.arguments.count > 1 else {
    FileHandle.standardError.write("usage: ocr <image>\n".data(using: .utf8)!)
    exit(2)
}
let url = URL(fileURLWithPath: CommandLine.arguments[1])
guard let image = NSImage(contentsOf: url),
      let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    FileHandle.standardError.write("cannot read image\n".data(using: .utf8)!)
    exit(1)
}
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = false
request.recognitionLanguages = ["en-US"]
let handler = VNImageRequestHandler(cgImage: cg, options: [:])
do {
    try handler.perform([request])
} catch {
    FileHandle.standardError.write("ocr failed: \(error)\n".data(using: .utf8)!)
    exit(1)
}
var out: [[String: Any]] = []
for obs in (request.results ?? []) {
    guard let cand = obs.topCandidates(1).first else { continue }
    let b = obs.boundingBox
    out.append(["text": cand.string, "x": Double(b.minX), "y": Double(1 - b.maxY),
                "w": Double(b.width), "h": Double(b.height), "conf": Double(cand.confidence)])
}
let data = try! JSONSerialization.data(withJSONObject: out, options: [])
print(String(data: data, encoding: .utf8)!)
