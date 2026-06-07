# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
from pathlib import Path

from tests.e2e.accuracy.wan21.video_metadata import ffprobe_video


def test_ffprobe_video_extracts_dimensions(monkeypatch, tmp_path: Path):
    video_path = tmp_path / "sample.mp4"

    def fake_run(cmd, check, text, capture_output):
        assert check is True
        assert text is True
        assert capture_output is True
        assert cmd[-1] == str(video_path)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="width=320\nheight=480\nnb_frames=5\nr_frame_rate=8/1\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert ffprobe_video(video_path) == {
        "width": "320",
        "height": "480",
        "nb_frames": "5",
        "r_frame_rate": "8/1",
    }
