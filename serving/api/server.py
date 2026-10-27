"""
Standard library HTTP API server.

Assigned to: Mohid
Status: Clean — Red Herring (Uses Python standard library http.server instead of Flask/FastAPI;
        intentional zero-dependency design).
"""

import http.server
import json
import threading
from typing import Optional, Any
from .request_handler import RequestHandler
from .routes import RouteHandler
from config.server_config import ServerConfig
from utils.logging_utils import get_logger

logger = get_logger("server")


def create_handler_class(router: RouteHandler):
    """Factory to create a BaseHTTPRequestHandler bound to a specific RouteHandler."""

    class TransformerHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
        def _send_json_response(self, status_code: int, data: dict):
            payload = json.dumps(data).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(payload)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            status, response = router.dispatch("GET", self.path)
            self._send_json_response(status, response)

        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                raw_body = self.rfile.read(content_length).decode("utf-8")
                try:
                    body = json.loads(raw_body)
                except Exception:
                    self._send_json_response(400, {"error": "Invalid JSON payload"})
                    return
            else:
                body = {}

            status, response = router.dispatch("POST", self.path, body)
            self._send_json_response(status, response)

        def log_message(self, format, *args):
            logger.info("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), format % args))

    return TransformerHTTPRequestHandler


class TransformerServer:
    """
    Pure stdlib multi-threaded HTTP server for transformer inference deployment.
    """

    def __init__(
        self,
        config: Optional[ServerConfig] = None,
        engine: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        self.config = config or ServerConfig()
        self.request_handler = RequestHandler(engine=engine, tokenizer=tokenizer)
        self.route_handler = RouteHandler(self.request_handler)
        self._server: Optional[http.server.ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self, background: bool = False) -> None:
        """Start the HTTP server on configured host and port."""
        handler_cls = create_handler_class(self.route_handler)
        self._server = http.server.ThreadingHTTPServer((self.config.host, self.config.port), handler_cls)
        self._server.daemon_threads = True
        logger.info(f"Starting stdlib HTTP server at http://{self.config.host}:{self.config.port}")

        if background:
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
        else:
            try:
                self._server.serve_forever()
            except KeyboardInterrupt:
                logger.info("Stopping HTTP server...")
                self.stop()

    def stop(self) -> None:
        """Stop and shutdown the HTTP server."""
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            logger.info("HTTP server stopped.")


def start_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    engine: Optional[Any] = None,
    tokenizer: Optional[Any] = None
) -> TransformerServer:
    """Convenience helper to launch the serving server."""
    cfg = ServerConfig(host=host, port=port)
    server = TransformerServer(config=cfg, engine=engine, tokenizer=tokenizer)
    server.start(background=True)
    return server
