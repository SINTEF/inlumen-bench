"""A tiny OpenAI-compatible streaming server for offline end-to-end runs.

    python tests/fake_server.py 8765

If the prompt contains a key from RESPONSES, the matching fixture file is
streamed back; otherwise the reply contains no code. Usage and cost are reported
like OpenRouter does, so the telemetry path is exercised.
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
RESPONSES = {"Player-style clustering": "case016_bundle.md"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.endswith("/models"):
            self._json({"data": [{"id": "fake-coder"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = "\n".join(m["content"] for m in body["messages"])
        text = "I cannot help with that."
        for key, fixture in RESPONSES.items():
            if key in prompt:
                text = (FIXTURES / fixture).read_text()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(0, len(text), 200):
            self._event({"model": body["model"], "choices": [{"delta": {"content": text[i:i + 200]}}]})
            time.sleep(0.05)
        self._event({"choices": [], "usage": {"prompt_tokens": len(prompt) // 4,
                                              "completion_tokens": len(text) // 4,
                                              "total_tokens": (len(prompt) + len(text)) // 4,
                                              "cost": 0.001}})
        self.wfile.write(b"data: [DONE]\n\n")

    def _event(self, data):
        self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()

    def _json(self, data):
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
