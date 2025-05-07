import math
from pathlib import Path
from typing import Literal

from adaptyv_lab import Protocol
from pydantic import BaseModel, Field

from hamilton_protocols import LAYOUTS_PATH


class ProteinPurificationProtocolParams(BaseModel):
    """Parameters for the protein purification protocol."""

    sample_volume: float = Field(
        default=50.0,
        ge=10.0,
        description="Volume of sample to purify (in uL).",
    )
    bead_volume: float = Field(
        default=50.0,
        ge=10.0,
        description="Total volume of beads to use (in uL).",
    )
    elution_volume: float = Field(
        default=40.0,
        ge=30.0,
        description="Volume to use for elution (in uL).",
    )
    wash_cycles: int = Field(
        default=4,
        ge=1,
        description="Number of wash cycles to perform.",
    )
    elution_cycles: int = Field(
        default=1,
        ge=1,
        description="Number of elution cycles to perform.",
    )
    bead_type: Literal["STXT", "His"] = Field(
        default="STXT",
        description="Type of magnetic beads to use.",
    )


def protein_purification_protocol(
    params: ProteinPurificationProtocolParams,
    simulate: bool = False,
    protocol: Protocol | None = None,
) -> Protocol:
    """Purify proteins using magnetic beads."""
    if not protocol:
        protocol = Protocol.from_layout(
            name="Protein Purification",
            layout_file=LAYOUTS_PATH / Path("purification.lay"),
            simulator_mode=simulate,
        )
    if not protocol.deck:
        msg = "Deck layout not loaded. Check that layouts/purification.lay exists."
        raise ValueError(msg)

    # plates
    mag_holder = protocol.deck.get_plate("D3")
    holder = protocol.deck.get_plate("C4")
    crude_samples = protocol.deck.get_plate("C3")
    pure_samples = protocol.deck.get_plate("C2")
    final_samples = protocol.deck.get_plate("C1")

    # reservoirs
    wash_buffer = protocol.deck.get_reservoir("B3")
    elution_buffer = protocol.deck.get_reservoir("B2")
    regen_buffer = protocol.deck.get_reservoir("B1")
    waste = protocol.deck.get_reservoir("D1")

    # tips
    holder_tips = protocol.deck.get_tip_rack("A4")
    bead_tips = protocol.deck.get_tip_rack("A3")
    wash_tips = protocol.deck.get_tip_rack("A2")
    elution_tips = protocol.deck.get_tip_rack("A1")
    sample_tips = protocol.deck.get_tip_rack("D2")
    sample_tips_src = protocol.deck.get_tip_rack_stack("E2")

    protocol.initialize()

    sample_vol = params.sample_volume
    bead_vol = params.bead_volume
    elution_vol = params.elution_volume
    wash_cycles = params.wash_cycles
    elution_cycles = params.elution_cycles
    eq_vol = wash_vol = min(150, bead_vol * 5)

    match params.bead_type:
        case "STXT":
            bead_type = "STXT"
        case "His":
            raise NotImplementedError("His purification not implemented yet.")
        case _:
            raise ValueError(f"Invalid bead type: {params.bead_type}")

    # add beads
    protocol.pickup_tips(bead_tips).eject_tips(holder_tips)
    protocol.pickup_tips(holder_tips).aspirate(
        mag_holder, 0, mix_cycles=5, mix_volume=bead_vol / 2, liquid_height=2
    )
    asp_cycles = math.ceil(bead_vol / bead_tips.tip.max_volume)
    asp_vol = bead_vol / asp_cycles
    for _ in range(asp_cycles):
        protocol.aspirate(mag_holder, asp_vol).dispense(
            waste, asp_vol, liquid_height=15
        )
    protocol.eject_tips(holder_tips).pickup_tips(holder_tips).eject_tips(bead_tips)

    protocol.grip_get(mag_holder).grip_place(holder)

    # equilibrate beads
    protocol.pickup_tips(wash_tips).eject_tips(holder_tips).pickup_tips(holder_tips)
    for _ in range(wash_cycles - 1):
        asp_cycles = math.ceil(eq_vol / wash_tips.tip.max_volume)
        asp_vol = eq_vol / asp_cycles
        for _ in range(asp_cycles):
            protocol.aspirate(wash_buffer, asp_vol).dispense(holder, asp_vol)

        mix_vol = (
            eq_vol * 0.8
            if eq_vol <= wash_tips.tip.max_volume
            else wash_tips.tip.max_volume
        )
        protocol.aspirate(holder, 0, mix_cycles=25, mix_volume=mix_vol, liquid_height=2)
        protocol.grip_get(holder).grip_place(mag_holder)
        protocol.aspirate(
            mag_holder, 0, mix_cycles=5, mix_volume=mix_vol, liquid_height=2
        )

    asp_cycles = math.ceil(eq_vol / wash_tips.tip.max_volume)
    asp_vol = eq_vol / asp_cycles
    for _ in range(asp_cycles):
        protocol.aspirate(holder, asp_vol).dispense(waste, asp_vol, liquid_height=15)

    protocol.eject_tips(holder_tips).pickup_tips(holder_tips).eject_tips(wash_tips)

    # add samples
    protocol.grip_get(mag_holder).grip_place(holder)
    protocol.pickup_tips(sample_tips).eject_tips(holder_tips).pickup_tips(holder_tips)

    asp_cycles = math.ceil(sample_vol / sample_tips.tip.max_volume)
    asp_vol = sample_vol / asp_cycles
    for _ in range(asp_cycles):
        protocol.aspirate(crude_samples, asp_vol, liquid_height=0.1).dispense(
            holder, asp_vol
        )

    mix_vol = (
        sample_vol * 0.8
        if sample_vol <= sample_tips.tip.max_volume
        else sample_tips.tip.max_volume
    )
    protocol.aspirate(holder, 0, mix_cycles=25, mix_volume=mix_vol, liquid_height=2)
    protocol.grip_get(holder).grip_place(mag_holder)
    protocol.aspirate(mag_holder, 0, mix_cycles=5, mix_volume=mix_vol, liquid_height=2)

    asp_cycles = math.ceil(sample_vol / sample_tips.tip.max_volume)
    asp_vol = sample_vol / asp_cycles
    for _ in range(asp_cycles):
        protocol.aspirate(holder, asp_vol).dispense(waste, asp_vol, liquid_height=15)

    protocol.eject_tips(holder_tips).pickup_tips(holder_tips).eject_tips(sample_tips)

    # wash
    protocol.grip_get(mag_holder).grip_place(holder)

    return protocol
