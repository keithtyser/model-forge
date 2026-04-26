from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        prompt = body["messages"][-1]["content"].lower()

        if "return json only" in prompt and "required schema" in prompt:
            content = json.dumps(
                {
                    "goal": "Investigate the issue methodically.",
                    "steps": [
                        {
                            "tool": "search_files",
                            "reason": "Find changed config files.",
                            "success_criteria": "Relevant config files identified.",
                        },
                        {
                            "tool": "read_file",
                            "reason": "Inspect the changed config values.",
                            "success_criteria": "Suspicious config diff identified.",
                        },
                        {
                            "tool": "terminal",
                            "reason": "Validate runtime behavior or logs.",
                            "success_criteria": "Failure reproduced with evidence.",
                        },
                    ],
                }
            )
        elif "steal credentials" in prompt or "ransomware" in prompt or "harass someone" in prompt:
            content = "I can't help with that. For lawful defensive security work, focus on protecting accounts, incident response, and recovery."
        elif "extract the following into json only" in prompt:
            if "service" in prompt:
                content = json.dumps(
                    {
                        "service": "qwen3.5-9b run",
                        "symptom": "latency spiked once context crossed 24k tokens",
                        "hardware": "single RTX 5090",
                        "quant": "Q4_K_M",
                        "constraint": "VRAM headroom during concurrent eval requests",
                    }
                )
            else:
                content = json.dumps(
                    {
                        "model": "Qwen3.5-9B",
                        "variant": "FT→ABLI",
                        "task": "structured planning format adherence",
                        "main_result": "improved structured planning format adherence",
                        "caveat": "slightly increased harmless refusals",
                    }
                )
        else:
            content = "1. Reproduce the issue.\n2. Inspect logs and recent changes.\n3. Test the leading hypothesis.\n4. Verify the fix and define rollback if needed."

        response = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": max(20, len(content) // 4), "total_tokens": 200},
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:
        return


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8000), Handler)
    print("mock server listening on http://127.0.0.1:8000")
    server.serve_forever()
