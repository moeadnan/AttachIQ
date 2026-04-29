"""CLI entry point for the inference pipeline."""

from __future__ import annotations

import argparse
import json
import sys

from attachiq.inference.pipeline import predict
from attachiq.logging import get_logger
from attachiq.schemas import InferenceRequest

log = get_logger("cli")


def main() -> None:
    parser = argparse.ArgumentParser(description="AttachIQ inference CLI.")
    parser.add_argument("--text", type=str, default=None, help="Prompt text.")
    parser.add_argument("--image", type=str, default=None, help="Path to a document image.")
    parser.add_argument("--mode", type=str, default=None, help="Optional input_mode override.")
    args = parser.parse_args()

    req = InferenceRequest(prompt_text=args.text, image_path=args.image, input_mode=args.mode)
    resp = predict(req)
    log.info(f"Decision: {resp.decision} ({resp.compatibility_label}) conf={resp.confidence:.3f}")
    payload = json.dumps(resp.model_dump(), indent=2)
    sys.stdout.write(payload + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
