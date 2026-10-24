"""Small HTTP client for the gateway, shared by the full-screen app and print mode."""
import json
import socket
import threading
from typing import Iterator

import requests


class GatewayClient:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self._sock: socket.socket | None = None

    def status(self) -> dict | None:
        try:
            r = requests.get(f"{self.url}/v1/status", timeout=5)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            return None

    def set_mode(self, mode: str) -> dict:
        r = requests.post(f"{self.url}/v1/mode", json={"mode": mode}, timeout=5)
        r.raise_for_status()
        return r.json()

    def chat(self, body: dict, cancel: threading.Event | None = None) -> Iterator[tuple[str, object]]:
        """Stream one request. Yields ("step", dict), ("token", str) and ("routing", dict)."""
        with requests.post(f"{self.url}/v1/chat/completions", json={**body, "stream": True},
                           stream=True, timeout=600) as resp:
            resp.raise_for_status()
            self._sock = getattr(getattr(resp.raw, "_connection", None), "sock", None)
            for raw in resp.iter_lines():  # bytes: split on \n only, see serving/llm_client.py
                line = raw.decode("utf-8")
                if cancel is not None and cancel.is_set():
                    return
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    return
                chunk = json.loads(data)
                if "x_step" in chunk:
                    yield "step", chunk["x_step"]
                for choice in chunk.get("choices", []):
                    if text := (choice.get("delta") or {}).get("content"):
                        yield "token", text
                if "x_routing" in chunk:
                    yield "routing", chunk["x_routing"]

    def abort(self):
        """Stop the in-flight stream from another thread, without blocking.

        Shutting the socket down wakes the reading thread at once and the gateway sees the
        disconnect. Closing the response here instead would wait on the reader's buffer lock,
        freezing the caller (the UI) until the model finished.
        """
        sock = self._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
