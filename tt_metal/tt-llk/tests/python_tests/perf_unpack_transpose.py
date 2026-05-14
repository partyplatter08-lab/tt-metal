# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import PerfRunType, Transpose
from helpers.param_config import input_output_formats
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)


def _perf_unpack_transpose_matrix():
    """
    Valid (formats, unpack_transpose_faces, unpack_transpose_within_face) tuples.

    Int32 is omitted: identity-only transpose with both unpack flags No was always skipped.
    Bfp8_b/Float16 pairs avoid Int32 output and unsupported transpose-on-Int32 cases from the old sweep.
    """
    fmts = input_output_formats([DataFormat.Bfp8_b, DataFormat.Float16])
    transpose_modes = [
        (Transpose.No, Transpose.Yes),
        (Transpose.Yes, Transpose.No),
        (Transpose.Yes, Transpose.Yes),
    ]
    return [(f, uf, uw) for f in fmts for uf, uw in transpose_modes]


_PERF_UNPACK_TRANSPOSE_CASES = _perf_unpack_transpose_matrix()


@pytest.mark.perf
@pytest.mark.parametrize(
    "formats,unpack_transpose_faces,unpack_transpose_within_face",
    _PERF_UNPACK_TRANSPOSE_CASES,
    ids=[
        f"fmt:{f}-uf:{uf.name}-uw:{uw.name}"
        for f, uf, uw in _PERF_UNPACK_TRANSPOSE_CASES
    ],
)
def test_perf_unpack_transpose(
    perf_report,
    formats,
    unpack_transpose_faces,
    unpack_transpose_within_face,
):
    tile_count = 16

    configuration = PerfConfig(
        "sources/unpack_transpose_perf.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.UNPACK_ISOLATE],
        templates=[],
        runtimes=[
            TILE_COUNT(tile_count),
            UNPACK_TRANS_FACES(unpack_transpose_faces),
            UNPACK_TRANS_WITHIN_FACE(unpack_transpose_within_face),
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
    )

    configuration.run(perf_report)
