# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
from pathlib import Path


def ffprobe_video(path: Path) -> dict[str, str]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate",
            "-of",
            "default=nokey=0:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    entries = {}
    for line in result.stdout.splitlines():
        key, value = line.split("=", 1)
        entries[key] = value
    return entries
