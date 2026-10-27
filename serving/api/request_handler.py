"""
HTTP API request parsing, validation, and execution handler.
"""

import time
import uuid
from typing import Dict, Any, Optional, List
from utils.logging_utils import get_logger

logger = get_logger("request_handler")


class RequestHandler:
    """
    Handles endpoint request validation, inference invocation, and response formatting.
    """

    def __init__(self, engine: Optional[Any] = None, tokenizer: Optional[Any] = None):
        self.engine = engine
        self.tokenizer = tokenizer

    def handle_completion(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process OpenAI-compatible /v1/completions request.

        Args:
            request_body: Dictionary containing 'prompt', optional 'max_tokens', 'temperature', etc.

        Returns:
            Formatted response dictionary.
        """
        prompt = request_body.get("prompt", "")
        max_tokens = request_body.get("max_tokens", 16)
        temperature = request_body.get("temperature", 0.8)

        # Tokenize prompt
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            prompt_tokens = [ord(c) for c in prompt]

        # Generate continuation
        if self.engine is not None and hasattr(self.engine, "generate_tokens"):
            generated_tokens = self.engine.generate_tokens(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            completion_tokens = generated_tokens[len(prompt_tokens):]
            response_text = (
                self.tokenizer.decode(completion_tokens)
                if self.tokenizer is not None and hasattr(self.tokenizer, "decode")
                else "".join(chr(token) for token in completion_tokens if 0 <= token <= 0x10FFFF)
            )
        elif self.engine is not None and hasattr(self.engine, "generate"):
            generated_text = self.engine.generate(prompt, tokenizer=self.tokenizer, max_new_tokens=max_tokens)
            response_text = generated_text[len(prompt):] if generated_text.startswith(prompt) else generated_text
            completion_tokens = (
                self.tokenizer.encode(response_text, add_bos=False)
                if self.tokenizer is not None and hasattr(self.tokenizer, "encode")
                else [ord(c) for c in response_text]
            )
        else:
            generated_tokens = prompt_tokens + [prompt_tokens[-1] + 1 if prompt_tokens else 42]
            completion_tokens = generated_tokens[len(prompt_tokens):]
            response_text = (
                self.tokenizer.decode(completion_tokens)
                if self.tokenizer is not None and hasattr(self.tokenizer, "decode")
                else "".join(chr(token) for token in completion_tokens if 0 <= token <= 0x10FFFF)
            )

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "broken-transformer",
            "text": response_text,
            "choices": [
                {
                    "text": response_text,
                    "index": 0,
                    "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(completion_tokens),
                "total_tokens": len(prompt_tokens) + len(completion_tokens)
            }
        }

    def handle_health(self) -> Dict[str, Any]:
        """Process /health endpoint check."""
        return {
            "status": "ok",
            "service": "broken-transformer-server",
            "timestamp": time.time()
        }

    def handle_models(self) -> Dict[str, Any]:
        """Process /v1/models endpoint check."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "broken-transformer",
                    "object": "model",
                    "owned_by": "openai-dsg-iitr",
                    "ready": True
                }
            ]
        }
