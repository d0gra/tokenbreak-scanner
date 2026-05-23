"""CLI entrypoint for the TokenBreak model scanner."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .inspector import inspect_model
from .models import RiskLevel, ScannerReport
from .validator import AttackValidationResult, validate_attack

console = Console(stderr=True)


def _fragility_bar(fragility: float, width: int = 20) -> str:
    """Render a mini progress bar for fragility score."""
    filled = int(fragility * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {fragility:.2f}"


def _build_table(report: ScannerReport) -> Table:
    """Build a Rich table for a scanner report."""
    table = Table(title=f"TokenBreak Scan Report: {report.model_name}", show_header=False)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    # Risk level with color
    risk_color = {
        RiskLevel.LOW: "green",
        RiskLevel.HIGH: "red",
        RiskLevel.UNKNOWN: "yellow",
    }.get(report.risk_level, "white")

    table.add_row("Model Name", report.model_name)
    table.add_row("Model Type", report.model_type)
    table.add_row("Model Family", report.model_family)
    table.add_row("Tokenizer Class", report.tokenizer_class)
    table.add_row("Tokenizer Algorithm", report.tokenizer_algorithm.value)
    table.add_row("Vocab Size", str(report.vocab_size) if report.vocab_size else "N/A")
    table.add_row("Confidence Score", f"{report.confidence_score:.2f}")
    table.add_row(
        "Vulnerable to TokenBreak",
        Text("YES", style="bold red") if report.vulnerable_to_tokenbreak else Text("NO", style="bold green"),
    )
    table.add_row("Risk Level", Text(report.risk_level.value, style=f"bold {risk_color}"))
    table.add_row("Source", report.source)

    # Fragility score (ground-truth test result)
    fragility = report.config_metadata.get("fragility_score", None)
    if fragility is not None and isinstance(fragility, (int, float)) and fragility > 0:
        table.add_row("Token Fragility", _fragility_bar(float(fragility)))

    table.add_row("Recommendation", report.recommendation)

    # Evidence tree
    if report.detection_sources:
        table.add_row("", "")
        table.add_row("Detection Sources", Text("(evidence tree)", style="dim"))
        for i, src in enumerate(report.detection_sources, 1):
            bullet = f"  {i}. [{src.signal}]"
            detail = f"inferred={src.inferred or 'N/A'}, weight={src.weight:.2f}"
            if src.reason:
                detail += f" - {src.reason}"
            table.add_row(bullet, detail)

    return table


def _build_batch_summary_table(reports: list[ScannerReport]) -> Table:
    """Build a summary table for batch scan results."""
    table = Table(title="TokenBreak Batch Scan Summary")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Algorithm", style="white")
    table.add_column("Vulnerable", style="white")
    table.add_column("Confidence", style="white")
    table.add_column("Fragility", style="white")

    for report in reports:
        vuln_text = Text("YES", style="bold red") if report.vulnerable_to_tokenbreak else Text("NO", style="bold green")
        fragility = report.config_metadata.get("fragility_score", None)
        fragility_str = f"{fragility:.2f}" if isinstance(fragility, (int, float)) else "N/A"
        table.add_row(
            report.model_name,
            report.tokenizer_algorithm.value,
            vuln_text,
            f"{report.confidence_score:.2f}",
            fragility_str,
        )
    return table


def _print_json(report: ScannerReport, attack_result: Optional[AttackValidationResult] = None) -> None:
    """Print report as JSON."""
    data = report.model_dump(mode="json", exclude={"config_metadata", "tokenizer_metadata"})
    if attack_result is not None:
        data["attack_validation"] = attack_result.model_dump(mode="json")
    click.echo(json.dumps(data, indent=2))


def _print_table(report: ScannerReport, attack_result: Optional[AttackValidationResult] = None) -> None:
    """Print report as a Rich table."""
    click.echo()
    table = _build_table(report)
    console.print(table)

    if attack_result is not None:
        click.echo()
        if attack_result.success:
            console.print(
                Panel(
                    f"[bold red]Attack Validation: VULNERABLE[/bold red]\n"
                    f"Original text classified as: {attack_result.original_label} "
                    f"(confidence: {attack_result.original_confidence:.4f})\n"
                    f"Manipulated text: {attack_result.manipulated_text}\n"
                    f"Manipulated text classified as: {attack_result.manipulated_label} "
                    f"(confidence: {attack_result.manipulated_confidence:.4f})\n"
                    f"Bypass successful: TokenBreak evades detection.",
                    title="Live Attack Test",
                    border_style="red",
                )
            )
        else:
            console.print(
                Panel(
                    f"[bold green]Attack Validation: NOT VULNERABLE[/bold green]\n"
                    f"Original text classified as: {attack_result.original_label} "
                    f"(confidence: {attack_result.original_confidence:.4f})\n"
                    f"Manipulated text: {attack_result.manipulated_text or 'N/A'}\n"
                    f"TokenBreak did not produce a successful bypass.",
                    title="Live Attack Test",
                    border_style="green",
                )
            )
    click.echo()


def _scan_one(
    source: str,
    download: bool,
    trust_remote_code: bool,
    test_attack: bool,
    threshold: float,
    output_format: str,
) -> int:
    """Scan a single model and return exit code."""
    try:
        report = inspect_model(
            source,
            download=download,
            trust_remote_code=trust_remote_code,
        )
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 2
    except Exception as exc:
        console.print(f"[bold red]Unexpected error during inspection:[/bold red] {exc}")
        return 2

    attack_result: Optional[AttackValidationResult] = None
    if test_attack:
        if report.vulnerable_to_tokenbreak:
            try:
                attack_result = validate_attack(
                    source,
                    threshold=threshold,
                    download=download,
                    trust_remote_code=trust_remote_code,
                )
            except Exception as exc:
                console.print(f"[bold yellow]Warning:[/bold yellow] Attack validation failed: {exc}")
        else:
            console.print(
                "[bold yellow]Skipping attack test:[/bold yellow] "
                "Model is not flagged as vulnerable (Unigram tokenizer detected)."
            )

    if output_format == "json":
        _print_json(report, attack_result)
    else:
        _print_table(report, attack_result)

    if report.risk_level == RiskLevel.HIGH:
        return 1
    return 0


def _discover_model_dirs(root: Path) -> list[Path]:
    """Discover model directories under a root path.

    A directory is considered a model dir if it contains at least one of:
    config.json, tokenizer.json, tokenizer_config.json, or safetensors files.
    """
    model_dirs: list[Path] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        has_config = (entry / "config.json").exists()
        has_tokenizer_json = (entry / "tokenizer.json").exists()
        has_tokenizer_config = (entry / "tokenizer_config.json").exists()
        has_safetensors = any(entry.glob("*.safetensors"))
        has_gguf = any(entry.glob("*.gguf"))
        if has_config or has_tokenizer_json or has_tokenizer_config or has_safetensors or has_gguf:
            model_dirs.append(entry)
    return model_dirs


@click.command(name="tokenbreak-scan")
@click.argument("source")
@click.option(
    "--output",
    "output_format",
    type=click.Choice(["json", "table"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format for the report.",
)
@click.option(
    "--download",
    is_flag=True,
    default=False,
    help="Download model files from HuggingFace/custom repo if source is a model ID.",
)
@click.option(
    "--trust-remote-code",
    is_flag=True,
    default=False,
    help="Trust remote code when loading tokenizers. WARNING: executes arbitrary code.",
)
@click.option(
    "--test-attack",
    is_flag=True,
    default=False,
    help="Run a live TokenBreak attack validation against the model. "
         "Requires model weights and a classification head.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.995,
    show_default=True,
    help="Confidence threshold for TokenBreak attack validation.",
)
@click.option(
    "--hf-token",
    default=None,
    envvar="HF_TOKEN",
    help="HuggingFace API token for gated/private models. Also read from HF_TOKEN env var.",
)
@click.option(
    "--batch",
    is_flag=True,
    default=False,
    help="Treat SOURCE as a directory and scan all model subdirectories within it.",
)
@click.version_option(version=__version__)
def main(
    source: str,
    output_format: str,
    download: bool,
    trust_remote_code: bool,
    test_attack: bool,
    threshold: float,
    hf_token: Optional[str],
    batch: bool,
) -> None:
    """Scan MODEL_PATH_OR_ID for TokenBreak tokenizer vulnerabilities.

    SOURCE can be a local directory containing model files
    (config.json, tokenizer.json, etc.) or a HuggingFace / custom model ID.

    Use --batch to scan all model directories under SOURCE.

    Use --hf-token (or set HF_TOKEN env var) for gated/private models.
    """
    # Set HF token for gated model access
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    # Batch scan mode
    if batch:
        root = Path(source)
        if not root.is_dir():
            console.print(f"[bold red]Error:[/bold red] --batch requires a directory: '{source}'")
            sys.exit(2)

        model_dirs = _discover_model_dirs(root)
        if not model_dirs:
            console.print(f"[bold yellow]No model directories found under: {source}[/bold yellow]")
            sys.exit(0)

        console.print(f"[bold]Scanning {len(model_dirs)} model(s) under {source}...[/bold]\n")

        reports: list[ScannerReport] = []
        had_error = False
        for md in model_dirs:
            try:
                report = inspect_model(str(md), download=download, trust_remote_code=trust_remote_code)
                reports.append(report)
                console.print(
                    f"  {md.name:40s} "
                    f"{'[red]VULN[/red]' if report.vulnerable_to_tokenbreak else '[green]SAFE[/green]'} "
                    f"({report.tokenizer_algorithm.value}, conf={report.confidence_score:.2f})"
                )
            except Exception as exc:
                console.print(f"  [bold red]ERROR[/bold red] {md.name}: {exc}")
                had_error = True

        click.echo()
        if reports:
            console.print(_build_batch_summary_table(reports))

        # Exit code: 1 if any model vulnerable, 2 if any error, 0 if all safe
        if had_error:
            sys.exit(2)
        if any(r.vulnerable_to_tokenbreak for r in reports):
            sys.exit(1)
        sys.exit(0)

    # Single model scan
    exit_code = _scan_one(source, download, trust_remote_code, test_attack, threshold, output_format)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
