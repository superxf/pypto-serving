# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Merge serving SA_PROFILE and Simpler host STRACE on their monotonic clock.

Run the server with ``PYPTO_RUNTIME_LOG=v9`` and capture its combined stdout/stderr.
Then run ``python -m pypto_serving.tools.profile.merge_simpler_strace
--serving-trace trace.json --server-log server.log --output merged.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


DEVICE_READY_RE = re.compile(r"\[chip_process pid=(?P<pid>\d+) dev=(?P<device>\d+)\] ready")


def _runtime_parser() -> tuple[Any, Any, Any, Any]:
    try:
        from simpler_setup.tools.strace_timing import (
            count_record_heads,
            group_invocations,
            parse_drop_summaries,
            parse_spans,
        )
    except ModuleNotFoundError:
        runtime_root = Path(os.environ.get(
            "PYPTO_RUNTIME_ROOT", Path(__file__).resolve().parents[3].parent / "pypto" / "runtime"
        ))
        if not (runtime_root / "simpler_setup").is_dir():
            raise RuntimeError("simpler_setup is unavailable; install the PyPTO runtime") from None
        sys.path.insert(0, str(runtime_root))
        from simpler_setup.tools.strace_timing import (  # noqa: E402
            count_record_heads,
            group_invocations,
            parse_drop_summaries,
            parse_spans,
        )
    return count_record_heads, group_invocations, parse_drop_summaries, parse_spans


def merge_traces(serving: dict, server_log: str, invocations: list[Any]) -> dict:
    """Add one Simpler host lane per device, keeping only profiled invocations."""
    serving_events = serving["traceEvents"]
    serving_spans = [event for event in serving_events if event.get("ph") == "X"]
    if not serving_spans:
        raise ValueError("serving trace has no complete spans")
    start_us = min(float(event["ts"]) for event in serving_spans)
    end_us = max(float(event["ts"]) + float(event["dur"]) for event in serving_spans)

    pid_to_device = {
        int(match["pid"]): int(match["device"])
        for match in DEVICE_READY_RE.finditer(server_log)
    }
    if not pid_to_device or len(set(pid_to_device.values())) != len(pid_to_device):
        raise ValueError("server log has no unique chip_process pid-to-device mappings")

    selected = []
    candidate_roots = []
    for invocation in invocations:
        if invocation.pid not in pid_to_device:
            continue
        root = invocation.root()
        if root is None or root.name not in {"chip.run", "simpler_run"}:
            continue
        root_start = root.ts / 1000.0
        root_end = (root.ts + root.dur) / 1000.0
        candidate_roots.append((root_start, root_end))
        if root_start < end_us and root_end > start_us:
            selected.append(invocation)

    if not selected:
        if not candidate_roots:
            raise ValueError("server log contains no mapped Simpler host invocations")
        raise ValueError("serving trace and host STRACE do not overlap on CLOCK_MONOTONIC")
    covered = {invocation.pid for invocation in selected}
    missing = sorted(set(pid_to_device) - covered)
    if missing:
        raise ValueError(f"missing profiled host STRACE invocations for device PIDs {missing}")
    counts: dict[int, int] = defaultdict(int)
    for invocation in selected:
        counts[pid_to_device[invocation.pid]] += 1
    if len(set(counts.values())) != 1:
        raise ValueError(f"incomplete per-device Simpler invocation counts: {dict(sorted(counts.items()))}")

    host_start_us = min(invocation.root().ts / 1000.0 for invocation in selected)

    timed = [float(event["ts"]) for event in serving_events if "ts" in event]
    origin_us = min(timed + [host_start_us])
    serving_pids = {event["pid"] for event in serving_events if isinstance(event.get("pid"), int)}
    host_pid = max(serving_pids, default=0) + 1
    events = []
    for event in serving_events:
        copied = dict(event)
        if "ts" in copied:
            copied["ts"] = float(copied["ts"]) - origin_us
        events.append(copied)

    events.append({
        "ph": "M", "name": "process_name", "pid": host_pid,
        "args": {"name": f"Simpler host STRACE ({len(pid_to_device)} device lanes)"},
    })
    events.append({
        "ph": "M", "name": "process_sort_index", "pid": host_pid,
        "args": {"sort_index": len(serving_pids)},
    })
    for index, device in enumerate(sorted(pid_to_device.values())):
        events.extend([
            {"ph": "M", "name": "thread_name", "pid": host_pid, "tid": device,
             "args": {"name": f"device {device}"}},
            {"ph": "M", "name": "thread_sort_index", "pid": host_pid, "tid": device,
             "args": {"sort_index": index}},
        ])

    by_device: dict[int, list[Any]] = defaultdict(list)
    omitted_device_spans = 0
    for invocation in selected:
        by_device[pid_to_device[invocation.pid]].append(invocation)
    for device, device_invocations in by_device.items():
        for invocation in sorted(device_invocations, key=lambda item: item.root().ts):
            for span in invocation.spans:
                if span.is_device:
                    omitted_device_spans += 1
                    continue
                stage = span.name
                for root_name in ("chip.run", "simpler_run"):
                    if stage == root_name:
                        stage = "simpler_run"
                        break
                    if stage.startswith(f"{root_name}."):
                        stage = stage[len(root_name) + 1:]
                        break
                events.append({
                    "ph": "X", "name": f"inv={invocation.inv} | {stage}",
                    "cat": "strace.host", "pid": host_pid, "tid": device,
                    "ts": span.ts / 1000.0 - origin_us, "dur": span.dur / 1000.0,
                    "args": {"device": device, "invocation": invocation.inv,
                             "hid": invocation.hid, "strace_name": span.name,
                             "strace_depth": span.depth, "attrs": span.attrs,
                             "clock": "host CLOCK_MONOTONIC"},
                })

    return {
        "displayTimeUnit": "ms",
        "traceEvents": events,
        "metadata": {
            "description": "Serving SA_PROFILE and Simpler host STRACE on CLOCK_MONOTONIC",
            "omitted_device_clock_spans": omitted_device_spans,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-trace", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.serving_trace.resolve():
        parser.error("--output must differ from --serving-trace")

    log = args.server_log.read_text(encoding="utf-8", errors="replace")
    count_heads, group, drop_summaries, parse = _runtime_parser()
    lines = log.splitlines()
    spans = list(parse(lines))
    if not spans:
        parser.error("server log contains no Simpler [STRACE] records; rerun with PYPTO_RUNTIME_LOG=v9")
    if count_heads(lines) != len(spans):
        parser.error("server log contains incomplete Simpler [STRACE] records")
    drops = drop_summaries(lines)
    if any(report["total"] for report in drops.values()):
        parser.error(f"server log reports dropped Simpler records: {drops}")

    serving = json.loads(args.serving_trace.read_text(encoding="utf-8"))
    try:
        merged = merge_traces(serving, log, group(spans))
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, separators=(",", ":")) + "\n", encoding="utf-8")
    host_count = sum(event.get("cat") == "strace.host" for event in merged["traceEvents"])
    print(f"wrote {args.output}: {host_count} Simpler host spans")


if __name__ == "__main__":
    main()
