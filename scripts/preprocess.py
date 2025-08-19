import argparse
import subprocess
from pathlib import Path


def preprocess_to_wav(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess audio to 16kHz mono PCM16 WAV using ffmpeg")
    parser.add_argument("--input", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--output", type=str, required=True, help="Path to output WAV file")
    args = parser.parse_args()

    preprocess_to_wav(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()




