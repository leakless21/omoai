import json
from pathlib import Path

from omoai.logging_system.logger import get_logger, setup_logging


def main():
    setup_logging()
    logger = get_logger(__name__)
    segments = [
        {"start": "00:00:00:000", "end": "00:00:03:200", "text": "Hello world."},
        {
            "start": "0:05.500",
            "end": "0:07.000",
            "text_raw": "Second line",
            "text_punct": "Second line!",
        },
        {"start": 1.25, "end": 2.75, "text": "Third"},
    ]
    # cfg = get_config()  # Unused variable removed
    out_dir = Path("./tmp/omoai_ts_fix")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"bullets": ["x"]}
    final_json = {
        "segments": segments,
        "transcript_raw": "Hello world. Second line Third",
        "transcript_punct": "Hello world. Second line! Third.",
        "summary": summary,
        "metadata": {},
    }

    final_path = out_dir / "final.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

    logger.info(
        "Dev writer produced final.json",
        extra={
            "final_json": str(final_path),
            "exists": final_path.exists(),
            "size": (final_path.stat().st_size if final_path.exists() else -1),
        },
    )


if __name__ == "__main__":
    main()
