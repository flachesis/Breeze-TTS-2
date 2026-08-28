#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

HELPER = '''
def _flash_attn_2_available() -> bool:
    if not torch.cuda.is_available():
        return False
    return importlib.util.find_spec("flash_attn") is not None


'''

FALLBACK = '''            if (
                text_encoder_attn_implementation == "flash_attention_2"
                and not _flash_attn_2_available()
            ):
                logger.warning(
                    "flash_attn is unavailable; the text encoder falls back to "
                    "the eager attention kernel."
                )
                text_encoder_attn_implementation = "eager"
'''

NEEDLE = """            if text_encoder_attn_implementation is None:
                text_encoder_attn_implementation = "flash_attention_2"
            config.text_encoder_config._attn_implementation = (
"""


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <breeze-tts-dir>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1]) / "models" / "breeze.py"
    text = path.read_text(encoding="utf-8")
    if "_flash_attn_2_available" in text and "falls back to" in text:
        print(f"already patched: {path}")
        return 0
    if "import importlib.util" not in text:
        text = text.replace(
            "from __future__ import annotations\n\n",
            "from __future__ import annotations\n\nimport importlib.util\n",
            1,
        )
    if "_flash_attn_2_available" not in text:
        marker = "logger = logging.get_logger(__name__)\n"
        if marker not in text:
            print("could not find logger assignment", file=sys.stderr)
            return 1
        text = text.replace(marker, marker + "\n" + HELPER, 1)
    if NEEDLE not in text:
        print("could not find text-encoder attn assignment", file=sys.stderr)
        return 1
    text = text.replace(
        NEEDLE,
        "            if text_encoder_attn_implementation is None:\n"
        "                text_encoder_attn_implementation = \"flash_attention_2\"\n"
        + FALLBACK
        + "            config.text_encoder_config._attn_implementation = (\n",
        1,
    )
    path.write_text(text, encoding="utf-8")
    print(f"patched: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
