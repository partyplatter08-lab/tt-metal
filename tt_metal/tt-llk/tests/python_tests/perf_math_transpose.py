# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    DestAccumulation,
    PerfRunType,
    Transpose,
)
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    MATH_TRANSPOSE_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)

# Explicit matrix: same-format only, no double face-transpose, math transpose rules per format.
_PERF_MATH_TRANSPOSE_CASES = [
    (
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        Transpose.No,
        Transpose.Yes,
    ),
    (
        InputOutputFormat(DataFormat.Int32, DataFormat.Int32),
        Transpose.No,
        Transpose.No,
    ),
    (
        InputOutputFormat(DataFormat.Int32, DataFormat.Int32),
        Transpose.No,
        Transpose.Yes,
    ),
    (
        InputOutputFormat(DataFormat.Int32, DataFormat.Int32),
        Transpose.Yes,
        Transpose.No,
    ),
]


@pytest.mark.perf
@pytest.mark.parametrize(
    "formats,unpack_transpose_faces,math_transpose_faces",
    _PERF_MATH_TRANSPOSE_CASES,
    ids=[
        f"fmt:{fmt}-unpack:{u.name}-math:{m.name}"
        for fmt, u, m in _PERF_MATH_TRANSPOSE_CASES
    ],
)
def test_perf_math_transpose(
    perf_report,
    formats,
    unpack_transpose_faces,
    math_transpose_faces,
):
    tile_count = 16

    configuration = PerfConfig(
        "sources/math_transpose_perf.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[
            MATH_TRANSPOSE_FACES(math_transpose_faces),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
            UNPACK_TRANS_FACES(unpack_transpose_faces),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=(
            DestAccumulation.Yes
            if formats.input_format.is_32_bit()
            else DestAccumulation.No
        ),
    )

    configuration.run(perf_report)
