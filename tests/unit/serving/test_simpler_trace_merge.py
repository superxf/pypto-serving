# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import unittest
from dataclasses import dataclass

from pypto_serving.tools.profile.merge_simpler_strace import _runtime_parser, merge_traces


@dataclass
class Span:
    name: str
    ts: int
    dur: int
    depth: int = 0
    attrs: str = ""
    is_device: bool = False


@dataclass
class Invocation:
    pid: int
    inv: int
    hid: str
    spans: list[Span]

    def root(self) -> Span:
        return self.spans[0]


SERVING = {"traceEvents": [
    {"ph": "M", "name": "process_name", "pid": 7, "args": {"name": "serving"}},
    {"ph": "X", "name": "decode", "pid": 7, "tid": 1, "ts": 1000, "dur": 100},
]}
LOG = "[chip_process pid=101 dev=0] ready\n[chip_process pid=102 dev=1] ready\n"


def _invocation(pid: int, inv: int, ts: int) -> Invocation:
    return Invocation(pid, inv, "abcd", [
        Span("chip.run", ts, 60_000),
        Span("chip.run.bind", ts + 1_000, 10_000, depth=1),
    ])


class SimplerTraceMergeTests(unittest.TestCase):
    def test_runtime_parser_handles_adjacent_records(self) -> None:
        record = (
            "[STRACE] v=1 pid=101 tid=1 inv=2 hid=abcd depth=0 "
            "name=chip.run ts=1020000 dur=60000"
            "[STRACE] v=1 pid=101 tid=1 inv=2 hid=abcd depth=1 "
            "name=chip.run.bind ts=1021000 dur=10000"
        )
        count_heads, group, _drops, parse = _runtime_parser()
        self.assertEqual(count_heads([record]), 2)
        invocations = group(list(parse([record])))
        self.assertEqual(len(invocations), 1)
        self.assertEqual(invocations[0].root().name, "chip.run")
        self.assertEqual(len(invocations[0].spans), 2)

    def test_merge_two_rank_host_spans_and_exclude_warmup(self) -> None:
        invocations = [
            _invocation(101, 1, 100_000),
            _invocation(101, 2, 1_020_000),
            _invocation(102, 2, 1_030_000),
        ]
        merged = merge_traces(SERVING, LOG, invocations)
        events = merged["traceEvents"]
        host = [event for event in events if event.get("cat") == "strace.host"]

        self.assertEqual(len(host), 4)
        self.assertEqual({event["tid"] for event in host}, {0, 1})
        self.assertEqual({event["args"]["invocation"] for event in host}, {2})
        self.assertEqual(next(event for event in host if event["name"] == "inv=2 | bind")["dur"], 10)
        self.assertEqual(next(event for event in events if event.get("name") == "decode")["ts"], 0)
        self.assertTrue(any(
            event.get("name") == "thread_name" and event["args"]["name"] == "device 1"
            for event in events
        ))

    def test_merge_rejects_missing_rank_and_clock_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing profiled host STRACE"):
            merge_traces(SERVING, LOG, [_invocation(101, 2, 1_020_000)])
        with self.assertRaisesRegex(ValueError, "do not overlap on CLOCK_MONOTONIC"):
            merge_traces(SERVING, LOG, [
                _invocation(101, 2, 2_000_000), _invocation(102, 2, 2_000_000),
            ])

    def test_merge_omits_device_clock(self) -> None:
        invocation = _invocation(101, 2, 1_020_000)
        invocation.spans.append(Span("chip.run.device_wall", 0, 50, is_device=True))
        merged = merge_traces(SERVING, LOG, [invocation, _invocation(102, 2, 1_030_000)])
        self.assertEqual(merged["metadata"]["omitted_device_clock_spans"], 1)
        self.assertFalse(any(
            event.get("args", {}).get("strace_name") == "chip.run.device_wall"
            for event in merged["traceEvents"]
        ))

    def test_merge_rejects_partial_rank_log(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete per-device Simpler invocation counts"):
            merge_traces(SERVING, LOG, [
                _invocation(101, 2, 1_020_000),
                _invocation(101, 3, 1_040_000),
                _invocation(102, 2, 1_030_000),
            ])


if __name__ == "__main__":
    unittest.main()
