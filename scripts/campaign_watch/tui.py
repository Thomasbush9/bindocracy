#!/usr/bin/env python3
"""Read-only campaign-watch PROTOTYPE (Python >=3.11, Textual 8.2.8).

Local: python tui.py --snapshot /path/to/snapshot.json
SSH:   python tui.py --host cluster-alias --snapshot /absolute/path/snapshot.json
Optional --ssh-config PATH uses an existing OpenSSH config without changing it.
Only reads snapshots; never invokes the collector or controls campaign jobs.
Keys: j/k or arrows move; h/Escape returns to campaigns; l/Enter drills down.
GPU figures are optional sampled Slurm accounting, not instantaneous device load.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import shlex
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Footer, Header, Static

MAX_BYTES = 1024 * 1024
MAX_STDERR = 64 * 1024
FETCH_TIMEOUT = 15
STATES = {"complete", "running", "queued", "not_submitted", "needs_attention", "unknown"}
COUNTS = ("valid", "running_jobs", "pending_jobs", "failed_jobs", "scheduler_warning_jobs", "active_gpus")


def timestamp(value: Any, location: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{location}: expected UTC ISO timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{location}: invalid timestamp") from exc
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise ValueError(f"{location}: timestamp must include UTC timezone")
    return result


def validate_snapshot(data: Any) -> dict[str, Any]:
    """Reject malformed snapshots before they can replace the last-good view."""
    def obj(value: Any, where: str) -> dict:
        if not isinstance(value, dict):
            raise ValueError(f"{where}: expected object")
        return value

    def text(value: Any, where: str, *, nonempty: bool = True) -> None:
        if not isinstance(value, str) or (nonempty and not value.strip()):
            raise ValueError(f"{where}: expected {'nonempty ' if nonempty else ''}string")

    def strings(value: Any, where: str) -> None:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{where}: expected string array")

    def count(value: Any, where: str) -> None:
        if type(value) is not int or value < 0:
            raise ValueError(f"{where}: expected nonnegative integer")

    def number(value: Any, where: str) -> None:
        if type(value) not in (int, float):
            raise ValueError(f"{where}: expected finite nonnegative number")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or value < 0:
            raise ValueError(f"{where}: expected finite nonnegative number")

    def records(value: Any, where: str) -> list[dict]:
        if not isinstance(value, list):
            raise ValueError(f"{where}: expected array")
        identifiers = set()
        for item in value:
            obj(item, where)
            text(item.get("id"), f"{where}.id")
            if item["id"] in identifiers:
                raise ValueError(f"{where}: duplicate id {item['id']!r}")
            identifiers.add(item["id"])
            text(item.get("name"), f"{where}.name")
        return value

    obj(data, "snapshot")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("snapshot: unsupported schema_version (expected 1)")
    timestamp(data.get("generated_at"), "generated_at")
    count(data.get("refresh_seconds"), "refresh_seconds")
    if data["refresh_seconds"] < 1:
        raise ValueError("refresh_seconds: must be positive")
    collector = obj(data.get("collector"), "collector")
    text(collector.get("host"), "collector.host")
    text(collector.get("job_id"), "collector.job_id")
    strings(data.get("errors"), "errors")
    for campaign in records(data.get("campaigns"), "campaigns"):
        where = f"campaign {campaign['id']!r}"
        text(campaign.get("root"), f"{where}.root")
        if "observed_at" not in campaign:
            raise ValueError(f"{where}: missing observed_at")
        if campaign["observed_at"] is not None:
            timestamp(campaign["observed_at"], f"{where}.observed_at")
        strings(campaign.get("warnings"), f"{where}.warnings")
        for stage in records(campaign.get("stages"), f"{where}.stages"):
            location = f"{where}, stage {stage['id']!r}"
            if not isinstance(stage.get("state"), str) or stage["state"] not in STATES:
                raise ValueError(f"{location}: invalid state")
            if not isinstance(stage.get("unit"), str) or stage["unit"] not in {"parents", "children", "designs"}:
                raise ValueError(f"{location}: invalid unit")
            if "planned" not in stage:
                raise ValueError(f"{location}: missing planned")
            if stage["planned"] is not None:
                count(stage["planned"], f"{location}.planned")
            for key in COUNTS:
                count(stage.get(key), f"{location}.{key}")
            if stage["planned"] is not None and stage["valid"] > stage["planned"]:
                raise ValueError(f"{location}: valid exceeds planned")
            for key in ("eta_seconds", "throughput_per_hour"):
                if key not in stage:
                    raise ValueError(f"{location}: missing {key}")
                if stage[key] is not None:
                    number(stage[key], f"{location}.{key}")
            strings(stage.get("warnings"), f"{location}.warnings")
            if "gpu_utilization" in stage:
                gpu_location = f"{location}.gpu_utilization"
                gpu = obj(stage["gpu_utilization"], gpu_location)
                text(gpu.get("source"), f"{gpu_location}.source")
                for key in ("observed_at", "mean_percent", "min_percent", "max_percent", "mean_memory_mib", "error"):
                    if key not in gpu:
                        raise ValueError(f"{gpu_location}: missing {key}")
                if gpu["observed_at"] is not None:
                    timestamp(gpu["observed_at"], f"{gpu_location}.observed_at")
                if gpu["error"] is not None:
                    text(gpu["error"], f"{gpu_location}.error")
                for key in ("eligible_jobs", "queried_jobs", "sampled_jobs", "low_util_jobs", "memory_sampled_jobs"):
                    count(gpu.get(key), f"{gpu_location}.{key}")
                if not gpu["sampled_jobs"] <= gpu["queried_jobs"] <= gpu["eligible_jobs"]:
                    raise ValueError(f"{gpu_location}: expected sampled <= queried <= eligible")
                if gpu["low_util_jobs"] > gpu["sampled_jobs"] or gpu["memory_sampled_jobs"] > gpu["queried_jobs"]:
                    raise ValueError(f"{gpu_location}: sample counts inconsistent")
                for key in ("mean_percent", "min_percent", "max_percent"):
                    if gpu["sampled_jobs"]:
                        number(gpu[key], f"{gpu_location}.{key}")
                        if gpu[key] > 100:
                            raise ValueError(f"{gpu_location}.{key}: exceeds 100%")
                    elif gpu[key] is not None:
                        raise ValueError(f"{gpu_location}.{key}: must be null without samples")
                if gpu["sampled_jobs"] and not gpu["min_percent"] <= gpu["mean_percent"] <= gpu["max_percent"]:
                    raise ValueError(f"{gpu_location}: expected min <= mean <= max")
                if gpu["memory_sampled_jobs"]:
                    number(gpu["mean_memory_mib"], f"{gpu_location}.mean_memory_mib")
                elif gpu["mean_memory_mib"] is not None:
                    raise ValueError(f"{gpu_location}.mean_memory_mib: must be null without samples")
                if (gpu["sampled_jobs"] or gpu["memory_sampled_jobs"]) and gpu["observed_at"] is None:
                    raise ValueError(f"{gpu_location}: samples require observed_at")
            if stage.get("job_id") is not None:
                text(stage["job_id"], f"{location}.job_id")
            if "runtime_minutes_by_arm" in stage:
                for arm, minutes in obj(stage["runtime_minutes_by_arm"], f"{location}.runtime_minutes_by_arm").items():
                    text(arm, f"{location}.arm")
                    number(minutes, f"{location}.runtime_minutes_by_arm.{arm}")
    return data


def reject_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite number {value}")


def plain(value: Any) -> str:
    """Remote text is data, never Rich markup or terminal control sequences."""
    return "".join(char if char.isprintable() else " " for char in str(value))


async def read_bounded(stream: asyncio.StreamReader, maximum: int, label: str) -> bytes:
    chunks = bytearray()
    while True:
        chunk = await stream.read(min(65536, maximum + 1 - len(chunks)))
        if not chunk:
            return bytes(chunks)
        chunks.extend(chunk)
        if len(chunks) > maximum:
            raise ValueError(f"{label} exceeds {maximum:,}-byte limit")


async def fetch_snapshot(path: str, host: str | None = None, ssh_config: str | None = None) -> dict[str, Any]:
    if host:
        command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10"]
        if ssh_config:
            command.extend(["-F", ssh_config])
        # SSH passes its command through the remote login shell. Quote the one
        # user-supplied operand; -- prevents a filename from becoming an option.
        command.extend(["--", host, f"exec head -c {MAX_BYTES + 1} -- {shlex.quote(path)}"])
    else:
        # A subprocess also bounds local/NFS reads without blocking the UI or
        # leaving a timed-out asyncio.to_thread file read in flight.
        command = [sys.executable, "-I", "-S", "-c",
                   "import sys\nwith open(sys.argv[1], 'rb') as f:\n sys.stdout.buffer.write(f.read(int(sys.argv[2])))",
                   path, str(MAX_BYTES + 1)]
    process = await asyncio.create_subprocess_exec(
        *command, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        limit=65536,
    )
    readers = [asyncio.create_task(read_bounded(process.stdout, MAX_BYTES, "snapshot")),
               asyncio.create_task(read_bounded(process.stderr, MAX_STDERR, "stderr")),
               asyncio.create_task(process.wait())]
    try:
        try:
            stdout, stderr, returncode = await asyncio.wait_for(asyncio.gather(*readers), FETCH_TIMEOUT)
        except TimeoutError as exc:
            raise ValueError(f"snapshot read timed out after {FETCH_TIMEOUT}s") from exc
        if returncode:
            detail = plain(stderr.decode("utf-8", errors="replace"))[:2000]
            raise ValueError(f"{'SSH' if host else 'Local read'} exited {returncode}: {detail or 'no diagnostic'}")
        data = json.loads(stdout.decode("utf-8"), parse_constant=reject_constant)
        return validate_snapshot(data)
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        for reader in readers:
            if not reader.done():
                reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        await process.wait()


def duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "unavailable"
    if seconds >= 86400:
        return f"{seconds / 86400:.1f}d"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.0f}m"
    return f"{seconds:.0f}s"


def units(stage: dict) -> str:
    planned = "?" if stage["planned"] is None else f"{stage['planned']:,}"
    return f"{stage['valid']:,}/{planned} {stage['unit']}"


def progress(stage: dict) -> str:
    planned = stage["planned"]
    if planned is None or planned == 0:
        return "n/a"
    fraction = stage["valid"] / planned
    filled = int(10 * fraction)
    return f"[{'#' * filled}{'.' * (10 - filled)}] {fraction:.0%}"


def summary_stage(campaign: dict) -> dict | None:
    """Use the same stage for campaign progress and GPU accounting; never sum percentages."""
    candidates = [stage for stage in campaign["stages"] if stage["state"] != "not_submitted"]
    return next((stage for stage in candidates if stage["id"] == "optimization"), candidates[-1] if candidates else None)


class CampaignWatch(App):
    TITLE = "Campaign Watch — READ-ONLY PROTOTYPE"
    BINDINGS = [("q", "quit", "Quit"), ("r", "refresh", "Refresh"),
                ("escape", "campaigns", "Campaigns"), ("tab", "focus_next", "Next pane"),
                Binding("j", "move_down", "Down", show=False),
                Binding("k", "move_up", "Up", show=False),
                ("h", "campaigns", "Back"), ("l", "drill_down", "Open")]
    CSS = """
    Screen { layout: vertical; }
    #source { height: auto; max-height: 3; padding: 0 1; color: $text-muted; }
    #status { height: auto; max-height: 5; padding: 0 1; }
    #campaigns { height: 7; min-height: 4; border: solid $primary; }
    #campaign-title { height: auto; max-height: 3; padding: 0 1; }
    #stages { height: 1fr; min-height: 6; border: solid $primary; }
    #eta-note { height: auto; padding: 0 1; color: $warning; }
    #details-scroll { height: 8; min-height: 3; border: solid $secondary; }
    #details { height: auto; padding: 0 1; }
    """

    def __init__(self, snapshot_path: str, host: str | None, interval: int, ssh_config: str | None = None):
        super().__init__()
        self.snapshot_path = snapshot_path
        self.host = host
        self.interval = interval
        self.ssh_config = ssh_config
        self.snapshot: dict | None = None
        self.selected_campaign: str | None = None
        self.selected_stage: str | None = None
        self.fetching = False
        self.fetch_error: str | None = None
        self.last_success: datetime | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        source = f"SSH {self.host}" if self.host else f"LOCAL {socket.gethostname()}"
        yield Static(f"Source: {plain(source)}  |  File: {plain(self.snapshot_path)}\nRead-only; j/k or arrows move/scroll; h/Esc campaigns; l/Enter opens; Tab switches panes.", id="source", markup=False)
        yield Static("Waiting for first snapshot…", id="status", markup=False)
        yield DataTable(id="campaigns", cursor_type="row", zebra_stripes=True)
        yield Static("No campaign selected", id="campaign-title", markup=False)
        yield DataTable(id="stages", cursor_type="row", zebra_stripes=True)
        yield Static("ETA is conditional remaining execution time at recorded concurrency; excludes queue wait. No ETA is invented for future stages.", id="eta-note", markup=False)
        with VerticalScroll(id="details-scroll"):
            yield Static("No snapshot loaded.", id="details", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        campaigns = self.query_one("#campaigns", DataTable)
        for key, label in (("name", "Campaign"), ("stage", "Summary stage"), ("units", "Unique valid/planned"),
                           ("jobs", "Run/Pend/Fail"), ("gpus", "GPUs"), ("gpu_util", "GPU% [n/run]"),
                           ("age", "Data age"), ("warnings", "Warnings")):
            campaigns.add_column(label, key=key)
        stages = self.query_one("#stages", DataTable)
        for key, label in (("name", "Stage"), ("state", "State"), ("progress", "Progress"),
                           ("units", "Unique valid/planned"), ("jobs", "Run/Pend/Fail"),
                           ("gpus", "GPUs"), ("gpu_util", "GPU% [n/run]"), ("eta", "ETA*"), ("warnings", "Warnings")):
            stages.add_column(label, key=key)
        campaigns.focus()
        self.set_interval(self.interval, self.action_refresh)
        self.set_interval(1, self.update_ages)
        self.action_refresh()

    def action_refresh(self) -> None:
        if self.fetching:
            return
        self.fetching = True
        self.update_ages()
        self.run_worker(self.load_snapshot(), name="snapshot-read")

    async def load_snapshot(self) -> None:
        try:
            snapshot = await fetch_snapshot(self.snapshot_path, self.host, self.ssh_config)
            self.snapshot = snapshot
            self.last_success = datetime.now(timezone.utc)
            self.fetch_error = None
            self.render_snapshot()
        except Exception as exc:
            self.fetch_error = plain(f"{type(exc).__name__}: {exc}")[:2200]
        finally:
            self.fetching = False
            self.update_ages()

    def campaign(self) -> dict | None:
        if self.snapshot:
            return next((item for item in self.snapshot["campaigns"] if item["id"] == self.selected_campaign), None)
        return None

    def age(self, observed_at: str | None) -> tuple[str, bool]:
        if observed_at is None:
            return "NEVER OBSERVED", True
        seconds = (datetime.now(timezone.utc) - timestamp(observed_at, "observed_at")).total_seconds()
        if seconds < -30:
            return "CLOCK SKEW", True
        threshold = max(120, 2 * self.snapshot["refresh_seconds"]) if self.snapshot else 120
        stale = seconds > threshold
        return f"{'STALE ' if stale else ''}{duration(max(0, seconds))}", stale

    def gpu_status(self, gpu: dict, campaign: dict) -> str:
        if any(self.age(value)[1] for value in (
                gpu["observed_at"], campaign["observed_at"], self.snapshot["generated_at"])):
            return "STALE"
        if self.fetch_error or self.snapshot["errors"]:
            return "LAST GOOD"
        if gpu["error"] or gpu["sampled_jobs"] < gpu["queried_jobs"]:
            return "PARTIAL"
        return ""

    def gpu_summary(self, stage: dict | None, campaign: dict) -> str:
        gpu = stage.get("gpu_utilization") if stage else None
        if gpu is None or not gpu["eligible_jobs"]:
            return "—"
        coverage = f"[{gpu['sampled_jobs']}/{gpu['eligible_jobs']}]"
        if not gpu["sampled_jobs"]:
            return f"— {coverage}"
        status = self.gpu_status(gpu, campaign)
        return f"{status + ' ' if status else ''}{gpu['mean_percent']:.0f}% {coverage}"

    def update_ages(self) -> None:
        lines = Text()
        if self.snapshot:
            age, stale = self.age(self.snapshot["generated_at"])
            campaign_stale = any(self.age(item["observed_at"])[1] for item in self.snapshot["campaigns"])
            state = "STALE / LAST GOOD" if self.fetch_error or stale or campaign_stale or self.snapshot["errors"] else "FRESH"
            lines.append(f"{state} | Snapshot age: {age}", style="bold yellow" if state != "FRESH" else "bold green")
            collector = self.snapshot["collector"]
            lines.append(f" | Collector: {plain(collector['host'])} / job {plain(collector['job_id'])}")
            lines.append(f"\nGenerated: {plain(self.snapshot['generated_at'])} | Poll: {self.interval}s")
            if self.last_success:
                lines.append(f" | Last fetch: {self.last_success.strftime('%H:%M:%S UTC')}")
            if self.snapshot["errors"]:
                lines.append(f" | {len(self.snapshot['errors'])} source error(s), see details", style="bold red")
            table = self.query_one("#campaigns", DataTable)
            for item in self.snapshot["campaigns"]:
                label, old = self.age(item["observed_at"])
                table.update_cell(item["id"], "age", Text(label, style="yellow" if old else "green"), update_width=True)
                table.update_cell(item["id"], "gpu_util", Text(self.gpu_summary(summary_stage(item), item)), update_width=True)
        else:
            lines.append("NO SNAPSHOT — waiting for readable, valid data", style="bold yellow")
        if self.fetching:
            lines.append(" | Fetching…")
        if self.fetch_error:
            lines.append(f"\nFETCH ERROR: {self.fetch_error}", style="bold red")
        self.query_one("#status", Static).update(lines)
        selected = self.campaign()
        if selected:
            age, _ = self.age(selected["observed_at"])
            self.query_one("#campaign-title", Static).update(
                f"{plain(selected['name'])} — observed {age} | Root: {plain(selected['root'])}")
            stages = self.query_one("#stages", DataTable)
            for stage in selected["stages"]:
                stages.update_cell(stage["id"], "gpu_util", Text(self.gpu_summary(stage, selected)), update_width=True)
            self.render_details()

    def render_snapshot(self) -> None:
        campaigns = self.snapshot["campaigns"]
        if not any(item["id"] == self.selected_campaign for item in campaigns):
            self.selected_campaign = campaigns[0]["id"] if campaigns else None
            self.selected_stage = None
        table = self.query_one("#campaigns", DataTable)
        table.clear()
        for item in campaigns:
            # A campaign has no additive design total across pipeline stages.
            stage = summary_stage(item)
            totals = [sum(stage[key] for stage in item["stages"]) for key in ("running_jobs", "pending_jobs", "failed_jobs", "active_gpus")]
            warnings = len(item["warnings"]) + sum(len(stage["warnings"]) + stage["scheduler_warning_jobs"] for stage in item["stages"])
            uncertain = any(stage["state"] == "unknown" for stage in item["stages"])
            cells = [item["name"], stage["name"] if stage else "—", units(stage) if stage else "—",
                     "?/?/?" if uncertain else f"{totals[0]}/{totals[1]}/{totals[2]}",
                     "?" if uncertain else str(totals[3]), self.gpu_summary(stage, item),
                     self.age(item["observed_at"])[0], str(warnings)]
            table.add_row(*(Text(plain(cell)) for cell in cells), key=item["id"])
        if self.selected_campaign is not None:
            table.move_cursor(row=table.get_row_index(self.selected_campaign))
        self.render_stages()

    def render_stages(self) -> None:
        campaign = self.campaign()
        table = self.query_one("#stages", DataTable)
        table.clear()
        if campaign:
            if not any(stage["id"] == self.selected_stage for stage in campaign["stages"]):
                self.selected_stage = campaign["stages"][0]["id"] if campaign["stages"] else None
            for stage in campaign["stages"]:
                eta = "—" if stage["eta_seconds"] is None else "~" + duration(stage["eta_seconds"])
                uncertain = stage["state"] == "unknown"
                cells = [stage["name"], stage["state"], progress(stage), units(stage),
                         "?/?/?" if uncertain else f"{stage['running_jobs']}/{stage['pending_jobs']}/{stage['failed_jobs']}",
                         "?" if uncertain else str(stage["active_gpus"]), self.gpu_summary(stage, campaign), eta,
                         f"{len(stage['warnings'])} notes; {stage['scheduler_warning_jobs']} jobs"]
                table.add_row(*(Text(plain(cell)) for cell in cells), key=stage["id"])
            if self.selected_stage is not None:
                table.move_cursor(row=table.get_row_index(self.selected_stage))
        else:
            self.selected_stage = None
            self.query_one("#campaign-title", Static).update("No campaigns in snapshot")
        self.render_details()

    def render_details(self) -> None:
        if not self.snapshot:
            return
        lines = [f"SOURCE ERROR: {plain(error)}" for error in self.snapshot["errors"]]
        campaign = self.campaign()
        if campaign:
            lines.extend(f"CAMPAIGN WARNING: {plain(warning)}" for warning in campaign["warnings"])
            stage = next((stage for stage in campaign["stages"] if stage["id"] == self.selected_stage), None)
            if stage:
                lines.append(f"Selected stage: {plain(stage['name'])} | {units(stage)} | Job: {plain(stage.get('job_id') or 'not supplied')}")
                rate = stage["throughput_per_hour"]
                rate_label = "unavailable" if rate is None else f"{rate:.2f} unique {stage['unit']}/h"
                lines.append(f"Throughput: {rate_label} | ETA: {duration(stage['eta_seconds'])} (conditional; excludes queue wait)")
                gpu = stage.get("gpu_utilization")
                if gpu is None:
                    lines.append("GPU utilization: unavailable — telemetry not supplied by this snapshot.")
                else:
                    status = self.gpu_status(gpu, campaign) if gpu["observed_at"] else "UNAVAILABLE"
                    label = f"GPU accounting{f' ({status})' if status else ''}"
                    if gpu["sampled_jobs"]:
                        lines.append(f"{label}: mean {gpu['mean_percent']:.1f}% | min/max {gpu['min_percent']:.1f}/{gpu['max_percent']:.1f}%"
                                     f" | samples {gpu['sampled_jobs']}/{gpu['eligible_jobs']} running ({gpu['queried_jobs']} queried)")
                        lines.append(f"Low utilization (<=10%): {gpu['low_util_jobs']}/{gpu['sampled_jobs']} samples; may include startup, not proof of a stall.")
                    else:
                        reason = "no eligible running single-GPU tasks" if not gpu["eligible_jobs"] else "no valid utilization samples"
                        lines.append(f"{label}: unavailable — {reason}; samples 0/{gpu['eligible_jobs']} ({gpu['queried_jobs']} queried).")
                        lines.append("Min/max and low-utilization count: unavailable without utilization samples.")
                    memory = f"{gpu['mean_memory_mib'] / 1024:.1f} GiB" if gpu["memory_sampled_jobs"] else "unavailable"
                    lines.append(f"Mean GPU memory: {memory} | samples {gpu['memory_sampled_jobs']}/{gpu['eligible_jobs']} running")
                    lines.append(f"GPU source: {plain(gpu['source'])} | observed: {plain(gpu['observed_at'] or 'never')} | age: {self.age(gpu['observed_at'])[0]}")
                    lines.append("Slurm accounting samples, not instantaneous GPU load or proof of computational efficiency.")
                    if gpu["error"]:
                        lines.append(f"GPU TELEMETRY ERROR: {plain(gpu['error'])}")
                if stage.get("runtime_minutes_by_arm"):
                    lines.append("Calibrated runtime by arm (minutes): " + ", ".join(f"{plain(arm)}={minutes:.2f}" for arm, minutes in stage["runtime_minutes_by_arm"].items()))
                if stage["scheduler_warning_jobs"]:
                    lines.append(f"Scheduler warnings: {stage['scheduler_warning_jobs']} job(s); independently valid outputs remain valid.")
                lines.extend(f"STAGE WARNING: {plain(warning)}" for warning in stage["warnings"])
        if not lines:
            lines.append("No campaign details or source warnings.")
        lines.append("Counts are unique units within each stage, not summed across the pipeline. Scroll this pane for all warnings.")
        self.query_one("#details", Static).update("\n".join(lines))

    @on(DataTable.RowHighlighted, "#campaigns")
    def highlight_campaign(self, event: DataTable.RowHighlighted) -> None:
        if event.cursor_row != event.data_table.cursor_row:
            return
        identifier = event.row_key.value
        if self.snapshot and any(item["id"] == identifier for item in self.snapshot["campaigns"]):
            if identifier != self.selected_campaign:
                self.selected_campaign = identifier
                self.selected_stage = None
                self.render_stages()
                self.update_ages()

    @on(DataTable.RowSelected, "#campaigns")
    def select_campaign(self, event: DataTable.RowSelected) -> None:
        self.query_one("#stages", DataTable).focus()

    @on(DataTable.RowHighlighted, "#stages")
    def highlight_stage(self, event: DataTable.RowHighlighted) -> None:
        if event.cursor_row != event.data_table.cursor_row:
            return
        campaign = self.campaign()
        if campaign and any(stage["id"] == event.row_key.value for stage in campaign["stages"]):
            self.selected_stage = event.row_key.value
            self.render_details()

    @on(DataTable.RowSelected, "#stages")
    def select_stage(self, event: DataTable.RowSelected) -> None:
        self.query_one("#details-scroll", VerticalScroll).focus()

    def action_move_down(self) -> None:
        if isinstance(self.focused, DataTable):
            self.focused.action_cursor_down()
        elif isinstance(self.focused, VerticalScroll):
            self.focused.action_scroll_down()

    def action_move_up(self) -> None:
        if isinstance(self.focused, DataTable):
            self.focused.action_cursor_up()
        elif isinstance(self.focused, VerticalScroll):
            self.focused.action_scroll_up()

    def action_drill_down(self) -> None:
        if self.focused is self.query_one("#campaigns", DataTable):
            self.query_one("#stages", DataTable).focus()
        elif self.focused is self.query_one("#stages", DataTable):
            self.query_one("#details-scroll", VerticalScroll).focus()

    def action_campaigns(self) -> None:
        self.query_one("#campaigns", DataTable).focus()


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only campaign-watch PROTOTYPE; consumes existing JSON snapshots only.")
    parser.add_argument("--snapshot", required=True, help="Snapshot file path (absolute remote path recommended; no shell expansion)")
    parser.add_argument("--host", help="OpenSSH host alias; omitted for local/offline review")
    parser.add_argument("--ssh-config", help="Optional OpenSSH configuration file, passed as ssh -F PATH")
    parser.add_argument("--interval", type=int, default=30, help="Fetch interval in seconds (minimum 15; default 30)")
    args = parser.parse_args()
    if args.interval < 15:
        parser.error("--interval must be at least 15 seconds")
    if not args.snapshot or "\x00" in args.snapshot:
        parser.error("--snapshot must be a nonempty path without NUL bytes")
    if args.host is not None and (not args.host or args.host.startswith("-") or any(char.isspace() or not char.isprintable() for char in args.host)):
        parser.error("--host must be a nonempty SSH alias without whitespace, control characters, or a leading dash")
    if args.ssh_config and not args.host:
        parser.error("--ssh-config requires --host")
    path = args.snapshot if args.host else str(Path(args.snapshot).expanduser().resolve())
    config = str(Path(args.ssh_config).expanduser().resolve()) if args.ssh_config else None
    CampaignWatch(path, args.host, args.interval, config).run()


if __name__ == "__main__":
    main()
