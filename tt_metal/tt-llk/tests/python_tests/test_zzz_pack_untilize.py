# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.data_format_inference import infer_data_formats
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    UntilizeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    format_dict,
)
from helpers.param_config import (
    exclude_fp8_e4m3_on_wormhole,
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator_v2 import StimuliSpec, generate_stimuli_v2
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    NUM_FACES,
    TILE_COUNT,
    TILE_DST_CT_OFFSET,
    generate_input_dim,
)
from helpers.utils import passed_test


def _pack_untilize_zzz_formats():
    """Valid I/O format pairs: no Bfp8_b output, no Int32 mixing, no Fp8_e4m3 on Wormhole."""
    base = input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Int32,
            DataFormat.Bfp8_b,
            DataFormat.Fp8_e4m3,
        ]
    )
    base = exclude_fp8_e4m3_on_wormhole(base)
    return [
        f
        for f in base
        if f.output_format != DataFormat.Bfp8_b
        and not (
            (f.input_format == DataFormat.Int32) ^ (f.output_format == DataFormat.Int32)
        )
    ]


def _pack_untilize_zzz_dest_acc_modes(formats):
    modes = get_valid_dest_accumulation_modes(formats)
    valid_modes = []
    for dest_acc in modes:
        data_formats = infer_data_formats(
            formats.input_format,
            formats.output_format,
            dest_acc,
            False,
        )
        if (
            formats.input_format == DataFormat.Float16
            and data_formats.pack_src.is_32_bit()
            and dest_acc == DestAccumulation.No
        ):
            continue
        valid_modes.append(dest_acc)
    return valid_modes


@parametrize(
    formats=_pack_untilize_zzz_formats(),
    dest_acc=lambda formats: _pack_untilize_zzz_dest_acc_modes(formats),
    input_dimensions=[[64, 64], [32, 128], [128, 128], [32, 64]],
    #  TODO add DestSync::Full tests when we have a solution for the static_assert in _llk_pack_untilize_init_ that requires block_ct_dim to be less or equal to 8,
    #  which is currently a limitation for testing DestSync::Full with the Untilize blocks calculation algorithm.
    dest_sync=[DestSync.Half],
    tile_dst_ct_offset=[0],  # Non-zero offsets are tracked in #1449
)
def test_pack_untilize(
    formats,
    dest_acc,
    input_dimensions,
    dest_sync,
    tile_dst_ct_offset,
):
    data_formats = infer_data_formats(
        formats.input_format,
        formats.output_format,
        dest_acc,
        False,
    )

    sfpu_false_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli_v2(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=sfpu_false_spec,
        spec_B=sfpu_false_spec,
    )

    generate_golden = get_golden_generator(UntilizeGolden)

    golden_tensor = generate_golden(src_A, formats.output_format, input_dimensions)

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    # _llk_pack_untilize_init_ has a static_assert that checks if block_ct_dim is less or equal to 8.
    # TODO: Update this logic to accept more than 8 tiles per block if the static_assert changes in the future.
    _, block_ct_dim = get_num_blocks_and_num_tiles_in_block(
        dest_sync,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Untilize,
    )

    configuration = TestConfig(
        "sources/pack_untilize_test.cpp",
        formats,
        templates=[
            generate_input_dim(
                input_dimensions,
                input_dimensions,
                block_ct_dim,
            ),
            DEST_SYNC(dest_sync),
            TILE_DST_CT_OFFSET(tile_dst_ct_offset),
        ],
        runtimes=[TILE_COUNT(tile_cnt_A), NUM_FACES(4)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            sfpu=False,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=unpack_to_dest,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
