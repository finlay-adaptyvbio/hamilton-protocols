import math
from pathlib import Path

from adaptyv_lab import Protocol
from pydantic import BaseModel, Field
from hamilton_protocols import LAYOUTS_PATH


class MaxPlateProtocolParams(BaseModel):
    plates: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Number of plates to process",
    )
    probe_columns: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Number of probe columns",
    )
    probe_rows: int = Field(
        default=8,
        ge=1,
        le=8,
        description="Number of probe rows",
    )
    well_volume: float = Field(
        default=250.0,
        ge=180.0,
        le=250.0,
        description="Volume of each well",
    )


def max_plate_protocol(
    params: MaxPlateProtocolParams,
    simulate: bool = False,
    protocol: Protocol | None = None,
) -> Protocol:
    plates = params.plates
    probe_columns = params.probe_columns
    probe_rows = params.probe_rows
    well_volume = params.well_volume

    protocol = Protocol.from_layout(
        name="Max Plate Protocol",
        layout_file=LAYOUTS_PATH / Path("bli.lay"),
        simulator_mode=simulate,
    )
    if not protocol.deck:
        msg = "Deck layout not loaded. Check that layouts/bli.lay exists."
        raise ValueError(msg)

    max_plates_src = protocol.deck.get_plate_stack("F4")
    max_plates_dst = protocol.deck.get_plate_stack("F3")
    max_plate = protocol.deck.get_plate("E4")
    loading_buffer = protocol.deck.get_reservoir("B3")
    regeneration_buffer = protocol.deck.get_reservoir("B1")
    holder_tips = protocol.deck.get_tip_rack("A4")
    buffer_tips = protocol.deck.get_tip_rack("A3")
    hv_tips = protocol.deck.get_tip_rack("E3")
    regen_tips = hv_tips[:4:2, -1]

    if not all(
        tip is not None for tip in [buffer_tips.tip, regen_tips.tip, holder_tips.tip]
    ):
        msg = "Tips not loaded. Check that A3, E3, and A4 are tip racks."
        raise ValueError(msg)

    buffer_dispense_cycles = math.ceil(well_volume / buffer_tips.tip.max_volume)
    buffer_volume = well_volume / buffer_dispense_cycles

    protocol.initialize()
    protocol.pickup_tips(buffer_tips).eject_tips(holder_tips)

    for i in range(plates):
        protocol.grip_get(max_plates_src[::-1][i], grip_width=81.5)
        protocol.grip_place(max_plate)

        protocol.pickup_tips(holder_tips)
        for _ in range(buffer_dispense_cycles):
            protocol.aspirate(
                loading_buffer,
                volume=buffer_volume,
            ).dispense(
                max_plate,
                volume=buffer_volume,
            )
        protocol.eject_tips(mode=1)

        protocol.pickup_tips(regen_tips)
        for col in range(probe_columns):
            protocol.aspirate(regeneration_buffer[:4:2, -1], volume=well_volume)
            for row in range(math.ceil(probe_rows / 2)):
                row_end = min(row + 5, probe_rows)
                step = 4
                protocol.dispense(
                    max_plate[row:row_end:step, col],
                    volume=well_volume,
                )
        protocol.eject_tips(regen_tips)

        protocol.grip_get(max_plate, grip_width=81.5, grip_height=12.0).grip_place(
            max_plates_dst[i]
        )

    protocol.pickup_tips(regen_tips).eject_tips(mode=1)
    protocol.pickup_tips(holder_tips).eject_tips(buffer_tips)

    return protocol
