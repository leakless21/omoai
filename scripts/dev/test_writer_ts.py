from pathlib import Path
from omoai.config import get_config
from omoai.output.writer import write_outputs


def main():
    segments = [
        {"start": "00:00:00:000", "end": "00:00:03:200", "text": "Hello world."},
        {"start": "0:05.500", "end": "0:07.000", "text_raw": "Second line", "text_punct": "Second line!"},
        {"start": 1.25, "end": 2.75, "text": "Third"},
    ]
    cfg = get_config()
    cfg.output.formats = ["json", "srt", "vtt", "text", "md"]
    cfg.output.transcript.include_segments = True
    cfg.output.transcript.include_punct = True
    cfg.output.transcript.include_raw = True
    out_dir = Path("./tmp/omoai_ts_fix")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = write_outputs(
        out_dir,
        segments,
        "Hello world. Second line Third",
        "Hello world. Second line! Third.",
        {"bullets": ["x"]},
        {},
        cfg.output,
    )
    print({k: str(v) for k, v in paths.items()})
    for k, v in paths.items():
        print(k, v.exists(), v.stat().st_size if v.exists() else -1)


if __name__ == "__main__":
    main()