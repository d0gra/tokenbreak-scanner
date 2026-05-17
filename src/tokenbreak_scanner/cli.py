"""CLI entrypoint for the TokenBreak model scanner."""

from __future__ import annotations

import json
import sys
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


def _print_json(report: ScannerReport, attack_result: Optional[AttackValidationResult] = None) -> None:
    """Print report as JSON."""
    data = report.model_dump(mode="json")
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
    help="Trust remote code when loading tokenizers.",
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
@click.version_option(version=__version__)
def main(
    source: str,
    output_format: str,
    download: bool,
    trust_remote_code: bool,
    test_attack: bool,
    threshold: float,
) -> None:
    """Scan MODEL_PATH_OR_ID for TokenBreak tokenizer vulnerabilities.

    SOURCE can be a local directory containing model files
    (config.json, tokenizer.json, etc.) or a HuggingFace / custom model ID.
    """
    try:
        report = inspect_model(
            source,
            download=download,
            trust_remote_code=trust_remote_code,
        )
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(2)
    except Exception as exc:
        console.print(f"[bold red]Unexpected error during inspection:[/bold red] {exc}")
        sys.exit(2)

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
                console.print(
                    f"[bold yellow]Warning:[/bold yellow] Attack validation failed: {exc}"
                )
        else:
            console.print(
                "[bold yellow]Skipping attack test:[/bold yellow] "
                "Model is not flagged as vulnerable (Unigram tokenizer detected)."
            )

    if output_format == "json":
        _print_json(report, attack_result)
    else:
        _print_table(report, attack_result)

    # Exit codes for CI pipelines
    if report.risk_level == RiskLevel.HIGH:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
