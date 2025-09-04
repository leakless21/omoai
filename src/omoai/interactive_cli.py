#!/usr/bin/env python3
"""
Interactive CLI for OMOAI - Audio Transcription and Summarization Pipeline

This module provides an interactive command-line interface for the OMOAI project,
allowing users to run the full pipeline or individual stages through guided prompts.
Uses the questionary library for user-friendly interactive prompts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import questionary
from questionary import Style


# Custom styling for the CLI
OMOAI_STYLE = Style([
    ('qmark', 'fg:#673ab7 bold'),           # question mark
    ('question', 'bold'),                   # question text
    ('answer', 'fg:#f44336 bold'),         # submitted answer text
    ('pointer', 'fg:#673ab7 bold'),         # pointer used in select prompts
    ('highlighted', 'fg:#673ab7 bold'),     # pointed-at choice in select prompts
    ('selected', 'fg:#cc5454'),             # style for selected items
    ('separator', 'fg:#cc5454'),            # separator in lists
    ('instruction', ''),                    # user instructions
    ('text', ''),                           # plain text
    ('disabled', 'fg:#858585 italic')       # disabled choices
])


def _repo_root() -> Path:
    """Get the repository root directory.

    Walk up from this file's location to find a directory that looks like the
    project root (contains either a top-level config.yaml or a .git folder).
    Fallback to the immediate parent if not found.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config.yaml").exists() or (parent / ".git").exists():
            return parent
    return here.parent


def _default_config_path() -> Path:
    """Get the default configuration file path."""
    env_cfg = os.environ.get("OMOAI_CONFIG")
    return Path(env_cfg) if env_cfg else (_repo_root() / "config.yaml")


def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[WARNING] Failed to load config from {config_path}: {e}")
    return {}


def _get_default_output_dir(audio_path: Path, config: Dict[str, Any]) -> Path:
    """Generate default output directory based on audio filename and timestamp."""
    paths = config.get("paths", {}) if isinstance(config, dict) else {}
    base_out_root = paths.get("out_dir") if isinstance(paths, dict) else None
    
    if base_out_root:
        default_root = Path(str(base_out_root))
    else:
        default_root = _repo_root() / "data/output"
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return default_root / f"{audio_path.stem}-{timestamp}"


def _get_default_model_dir(config: Dict[str, Any]) -> str:
    """Get default model directory from config."""
    paths = config.get("paths", {}) if isinstance(config, dict) else {}
    model_dir = paths.get("chunkformer_checkpoint") if isinstance(paths, dict) else None
    return str(model_dir) if model_dir else "models/chunkformer/chunkformer-large-vie"


def _validate_file_exists(path: str) -> bool | str:
    """Validate that a file exists."""
    if not path.strip():
        return "Please enter a file path"
    if not Path(path).exists():
        return f"File does not exist: {path}"
    return True


def _validate_directory_exists(path: str) -> bool | str:
    """Validate that a directory exists."""
    if not path.strip():
        return True  # Allow empty for optional fields
    if not Path(path).exists():
        return f"Directory does not exist: {path}"
    if not Path(path).is_dir():
        return f"Path is not a directory: {path}"
    return True


def _validate_wav_file(path: str) -> bool | str:
    """Validate that a file exists and is a WAV file."""
    basic_validation = _validate_file_exists(path)
    if basic_validation is not True:
        return basic_validation
    if not path.lower().endswith('.wav'):
        return "File must be a WAV file"
    return True


def _execute_command(cmd: list[str], description: str) -> bool:
    """Execute a command and return True if successful."""
    try:
        print(f"[INFO] {description}...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed:")
        print(f"Command: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"[ERROR] {description} failed with exception: {e}")
        return False


def _run_full_pipeline() -> None:
    """Run the complete audio processing pipeline."""
    print("\n=== Run Full Pipeline ===")
    
    # 1. Get audio file path
    audio_path_str = questionary.text(
        "Please enter the path to the audio file:",
        validate=_validate_file_exists,
        style=OMOAI_STYLE
    ).ask()
    
    if not audio_path_str:
        print("Operation cancelled.")
        return
    
    audio_path = Path(audio_path_str)
    
    # Load config for defaults
    config_file_path = _default_config_path()
    config = _load_config(config_file_path)
    
    # 2. Get output directory
    default_output = str(_get_default_output_dir(audio_path, config))
    output_dir_str = questionary.text(
        "Please enter the output directory:",
        default=default_output,
        style=OMOAI_STYLE
    ).ask()
    
    if not output_dir_str:
        print("Operation cancelled.")
        return
    
    output_dir = Path(output_dir_str)
    
    # 3. Get model directory (optional)
    default_model_dir = _get_default_model_dir(config)
    model_dir_str = questionary.text(
        "Please enter the path to the model directory (optional):",
        default=default_model_dir,
        validate=_validate_directory_exists,
        style=OMOAI_STYLE
    ).ask()
    
    if model_dir_str is None:
        print("Operation cancelled.")
        return
    
    # 4. Get config file (optional)
    config_path_str = questionary.text(
        "Please enter the path to the config file (optional):",
        default=str(config_file_path),
        style=OMOAI_STYLE
    ).ask()
    
    if config_path_str is None:
        print("Operation cancelled.")
        return
    
    # 5. Confirmation
    print("\n--- Pipeline Configuration ---")
    print(f"Audio file: {audio_path}")
    print(f"Output directory: {output_dir}")
    print(f"Model directory: {model_dir_str or 'Default from config'}")
    print(f"Config file: {config_path_str or 'Default'}")
    print("---")
    
    confirm = questionary.confirm(
        "Do you want to start the pipeline with these settings?",
        default=True,
        style=OMOAI_STYLE
    ).ask()
    
    if not confirm:
        print("Pipeline cancelled.")
        return
    
    # 6. Execute pipeline
    print("\n[INFO] Starting pipeline...")
    
    # Import and use the existing pipeline function
    try:
        from omoai.main import run_pipeline
        
        model_dir_path = Path(model_dir_str) if model_dir_str else None
        config_path = Path(config_path_str) if config_path_str else None
        
        rc = run_pipeline(audio_path, output_dir, model_dir_path, config_path)
        
        if rc == 0:
            print(f"\n[SUCCESS] Pipeline completed successfully!")
            print(f"[INFO] All outputs saved to: {output_dir}")
        else:
            print(f"\n[ERROR] Pipeline failed with return code: {rc}")
    
    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed: {e}")


def _run_preprocess_audio() -> None:
    """Run the audio preprocessing stage."""
    print("\n=== Preprocess Audio ===")
    
    # Get input file
    input_path_str = questionary.text(
        "Please enter the path to the input audio file:",
        validate=_validate_file_exists,
        style=OMOAI_STYLE
    ).ask()
    
    if not input_path_str:
        print("Operation cancelled.")
        return
    
    input_path = Path(input_path_str)
    
    # Get output file
    default_output = f"data/output/{input_path.stem}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}/preprocessed.wav"
    output_path_str = questionary.text(
        "Please enter the path for the output WAV file:",
        default=default_output,
        style=OMOAI_STYLE
    ).ask()
    
    if not output_path_str:
        print("Operation cancelled.")
        return
    
    output_path = Path(output_path_str)
    
    # Execute preprocessing
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ac", "1", "-ar", "16000", "-vn", "-c:a", "pcm_s16le",
        str(output_path)
    ]
    
    if _execute_command(cmd, "Preprocessing audio"):
        print(f"[SUCCESS] Preprocessing complete. Output: {output_path}")
    else:
        print("[ERROR] Preprocessing failed.")


def _run_asr() -> None:
    """Run the ASR (Automatic Speech Recognition) stage."""
    print("\n=== Run ASR ===")
    
    # Get preprocessed audio file
    audio_path_str = questionary.text(
        "Please enter the path to the preprocessed audio file:",
        validate=_validate_wav_file,
        style=OMOAI_STYLE
    ).ask()
    
    if not audio_path_str:
        print("Operation cancelled.")
        return
    
    audio_path = Path(audio_path_str)
    
    # Get output ASR JSON file
    default_output = str(audio_path.parent / "asr.json")
    output_path_str = questionary.text(
        "Please enter the path for the output ASR JSON file:",
        default=default_output,
        style=OMOAI_STYLE
    ).ask()
    
    if not output_path_str:
        print("Operation cancelled.")
        return
    
    output_path = Path(output_path_str)
    
    # Load config for defaults
    config_file_path = _default_config_path()
    config = _load_config(config_file_path)
    
    # Get model directory (optional)
    default_model_dir = _get_default_model_dir(config)
    model_dir_str = questionary.text(
        "Please enter the path to the model directory (optional):",
        default=default_model_dir,
        validate=_validate_directory_exists,
        style=OMOAI_STYLE
    ).ask()
    
    if model_dir_str is None:
        print("Operation cancelled.")
        return
    
    # Get config file (optional)
    config_path_str = questionary.text(
        "Please enter the path to the config file (optional):",
        default=str(config_file_path),
        style=OMOAI_STYLE
    ).ask()
    
    if config_path_str is None:
        print("Operation cancelled.")
        return
    
    # Execute ASR
    cmd = [
        sys.executable, "-m", "scripts.asr",
        "--config", config_path_str or str(config_file_path),
        "--audio", str(audio_path),
        "--out", str(output_path)
    ]
    
    if model_dir_str:
        cmd += ["--model-dir", model_dir_str]
    
    if _execute_command(cmd, "Running ASR"):
        print(f"[SUCCESS] ASR complete. Output: {output_path}")
    else:
        print("[ERROR] ASR failed.")


def _run_post_process() -> None:
    """Run the post-processing stage (punctuation and summarization)."""
    print("\n=== Post-process ASR Output ===")
    
    # Get ASR JSON file
    asr_json_str = questionary.text(
        "Please enter the path to the ASR JSON file:",
        validate=_validate_file_exists,
        style=OMOAI_STYLE
    ).ask()
    
    if not asr_json_str:
        print("Operation cancelled.")
        return
    
    asr_json_path = Path(asr_json_str)
    
    # Get output final JSON file
    default_output = str(asr_json_path.parent / "final.json")
    output_path_str = questionary.text(
        "Please enter the path for the output final JSON file:",
        default=default_output,
        style=OMOAI_STYLE
    ).ask()
    
    if not output_path_str:
        print("Operation cancelled.")
        return
    
    output_path = Path(output_path_str)
    
    # Get config file (optional)
    config_file_path = _default_config_path()
    config_path_str = questionary.text(
        "Please enter the path to the config file (optional):",
        default=str(config_file_path),
        style=OMOAI_STYLE
    ).ask()
    
    if config_path_str is None:
        print("Operation cancelled.")
        return
    
    # Execute post-processing
    cmd = [
        sys.executable, "-m", "scripts.post",
        "--config", config_path_str or str(config_file_path),
        "--asr-json", str(asr_json_path),
        "--out", str(output_path),
        "--auto-outdir"  # Always write separate transcript/summary files
    ]
    
    if _execute_command(cmd, "Post-processing ASR output"):
        print(f"[SUCCESS] Post-processing complete. Output: {output_path}")
    else:
        print("[ERROR] Post-processing failed.")


def _show_individual_stages_menu() -> None:
    """Show the individual stages submenu."""
    while True:
        stage_choice = questionary.select(
            "Select a stage:",
            choices=[
                "Preprocess Audio",
                "Run ASR",
                "Post-process ASR Output",
                "← Back to Main Menu"
            ],
            style=OMOAI_STYLE
        ).ask()
        
        if not stage_choice or stage_choice == "← Back to Main Menu":
            break
        elif stage_choice == "Preprocess Audio":
            _run_preprocess_audio()
        elif stage_choice == "Run ASR":
            _run_asr()
        elif stage_choice == "Post-process ASR Output":
            _run_post_process()


def _show_configuration() -> None:
    """Display current configuration settings."""
    print("\n=== Configuration ===")
    
    config_path = _default_config_path()
    config = _load_config(config_path)
    
    print(f"--- Current Configuration ---")
    print(f"Source: {config_path}")
    
    if config:
        # Display key configuration sections
        sections_to_show = ["paths", "asr", "llm", "output"]
        
        for section in sections_to_show:
            if section in config:
                print(f"{section}:")
                section_data = config[section]
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  {section_data}")
        
        # Show other top-level keys
        other_keys = [k for k in config.keys() if k not in sections_to_show]
        if other_keys:
            print("other settings:")
            for key in other_keys:
                print(f"  {key}: {config[key]}")
    else:
        print("No configuration loaded or configuration is empty.")
    
    print("---")
    
    # Wait for user to continue
    questionary.press_any_key_to_continue(
        "Press any key to return to the main menu...",
        style=OMOAI_STYLE
    ).ask()


def _show_quality_analysis() -> None:
    """Show quality metrics and diffs from processed files."""
    print("\n=== Quality Metrics & Diffs Analysis ===")
    
    # Get final JSON file
    final_json_str = questionary.text(
        "Please enter the path to the final.json file:",
        validate=_validate_file_exists,
        style=OMOAI_STYLE
    ).ask()
    
    if not final_json_str:
        print("Operation cancelled.")
        return
    
    final_json_path = Path(final_json_str)
    
    try:
        with open(final_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("\n--- Quality Analysis Results ---")
        
        # Display quality metrics if available
        if 'quality_metrics' in data and data['quality_metrics']:
            metrics = data['quality_metrics']
            print("\n📊 Quality Metrics:")
            print(f"  WER (Word Error Rate): {metrics.get('wer', 'N/A')}")
            print(f"  CER (Character Error Rate): {metrics.get('cer', 'N/A')}")
            print(f"  PER (Punctuation Error Rate): {metrics.get('per', 'N/A')}")
            print(f"  U-WER (Unpunctuated WER): {metrics.get('uwer', 'N/A')}")
            print(f"  F-WER (Formatted WER): {metrics.get('fwer', 'N/A')}")
            print(f"  Alignment Confidence: {metrics.get('alignment_confidence', 'N/A')}")
            
            # Provide interpretation
            wer = metrics.get('wer')
            if wer is not None:
                if wer < 0.05:
                    print("  🟢 WER indicates excellent performance")
                elif wer < 0.10:
                    print("  🟡 WER indicates good performance")
                elif wer < 0.15:
                    print("  🟠 WER indicates acceptable performance")
                else:
                    print("  🔴 WER indicates poor performance - may need investigation")
        else:
            print("\n⚠️  No quality metrics found in the file.")
            print("   Make sure to run processing with quality metrics enabled:")
            print("   - Set 'alignment.compute_quality_metrics: true' in config.yaml")
            print("   - Or use API with 'include_quality_metrics=true' parameter")
        
        # Display diffs if available
        if 'diffs' in data and data['diffs']:
            diffs = data['diffs']
            print("\n📝 Human-Readable Diff:")
            
            if 'original_text' in diffs and diffs['original_text']:
                print(f"\nOriginal Text:")
                print(f"  {diffs['original_text'][:200]}{'...' if len(diffs['original_text']) > 200 else ''}")
            
            if 'punctuated_text' in diffs and diffs['punctuated_text']:
                print(f"\nPunctuated Text:")
                print(f"  {diffs['punctuated_text'][:200]}{'...' if len(diffs['punctuated_text']) > 200 else ''}")
            
            if 'diff_output' in diffs and diffs['diff_output']:
                print(f"\nDiff Output:")
                print(f"  {diffs['diff_output'][:300]}{'...' if len(diffs['diff_output']) > 300 else ''}")
            
            if 'alignment_summary' in diffs and diffs['alignment_summary']:
                print(f"\nAlignment Summary:")
                print(f"  {diffs['alignment_summary']}")
        else:
            print("\n⚠️  No diff information found in the file.")
            print("   Make sure to run processing with diff generation enabled:")
            print("   - Set 'alignment.generate_diffs: true' in config.yaml")
            print("   - Or use API with 'include_diffs=true' parameter")
        
        # Show basic file info
        print(f"\n📁 File Information:")
        print(f"  File: {final_json_path}")
        print(f"  Size: {final_json_path.stat().st_size} bytes")
        
        if 'segments' in data and data['segments']:
            segments = data['segments']
            print(f"  Segments: {len(segments)}")
            total_duration = sum(seg.get('end', 0) - seg.get('start', 0) for seg in segments)
            print(f"  Total Duration: {total_duration:.2f} seconds")
        
        print("---")
        
        # Ask if user wants to save the analysis
        save_analysis = questionary.confirm(
            "Would you like to save this analysis to a file?",
            default=False,
            style=OMOAI_STYLE
        ).ask()
        
        if save_analysis:
            default_output = final_json_path.parent / "quality_analysis.txt"
            output_path_str = questionary.text(
                "Enter path for analysis output file:",
                default=str(default_output),
                style=OMOAI_STYLE
            ).ask()
            
            if output_path_str:
                output_path = Path(output_path_str)
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write("OMOAI Quality Analysis Report\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(f"File: {final_json_path}\n")
                        f.write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
                        
                        if 'quality_metrics' in data and data['quality_metrics']:
                            f.write("Quality Metrics:\n")
                            metrics = data['quality_metrics']
                            for key, value in metrics.items():
                                f.write(f"  {key}: {value}\n")
                            f.write("\n")
                        
                        if 'diffs' in data and data['diffs']:
                            f.write("Diff Information:\n")
                            diffs = data['diffs']
                            for key, value in diffs.items():
                                if value:
                                    f.write(f"  {key}:\n")
                                    f.write(f"    {value}\n\n")
                    
                    print(f"[SUCCESS] Analysis saved to: {output_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to save analysis: {e}")
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON file: {e}")
    except Exception as e:
        print(f"[ERROR] Failed to analyze file: {e}")
    
    # Wait for user to continue
    questionary.press_any_key_to_continue(
        "Press any key to return to the main menu...",
        style=OMOAI_STYLE
    ).ask()


def run_interactive_cli() -> None:
    """Main entry point for the interactive CLI."""
    print("\n🎧 Welcome to OMOAI Interactive CLI")
    print("Audio Transcription and Summarization Pipeline")
    print("=" * 50)
    
    while True:
        try:
            action = questionary.select(
                "Select an action:",
                choices=[
                    "Run Full Pipeline",
                    "Run Individual Stages",
                    "View Quality Metrics & Diffs",
                    "Configuration",
                    "Exit"
                ],
                style=OMOAI_STYLE
            ).ask()
            
            if not action or action == "Exit":
                print("\nGoodbye! 👋")
                break
            elif action == "Run Full Pipeline":
                _run_full_pipeline()
            elif action == "Run Individual Stages":
                _show_individual_stages_menu()
            elif action == "View Quality Metrics & Diffs":
                _show_quality_analysis()
            elif action == "Configuration":
                _show_configuration()
                
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred: {e}")
            continue


if __name__ == "__main__":
    run_interactive_cli()