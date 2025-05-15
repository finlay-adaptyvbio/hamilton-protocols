from pathlib import Path
from adaptyv_lab import Protocol
from pydantic import BaseModel, Field

from hamilton_protocols import LAYOUTS_PATH


class TwistPlateParams(BaseModel):
    """
    Parameters for condiguring a Twist plate.
    """

    plate_id: str = Field(..., description="ID of the plate to be twisted.")
    stock_concentration: float = Field(
        default=100, description="Stock target concentration of the DNA in nM."
    )
    diluted_concentration: float = Field(
        default=4, description="Diluted target concentration of the DNA in nM."
    )
    diluted_volume: float = Field(
        default=50, description="Volume of the diluted DNA in uL."
    )

    @property
    def stock_volume(self) -> float:
        """
        Calculate the volume of the stock DNA needed to achieve the desired diluted concentration.
        """
        return (
            self.diluted_concentration * self.diluted_volume
        ) / self.stock_concentration

    @property
    def water_volume(self) -> float:
        """
        Calculate the volume of water needed to achieve the desired diluted concentration.
        """
        return self.diluted_volume - self.stock_volume


class DNAReconstitutionParams(BaseModel):
    """
    Parameters for DNA reconstitution.
    """

    plates: list[TwistPlateParams] = Field(
        default_factory=list,
        description="List of Twist plate parameters for DNA reconstitution.",
        max_length=4,
        title="Twist Plates",
    )


def dna_reconstitution_protocol(
    params: DNAReconstitutionParams,
    simulate: bool = False,
    protocol: Protocol | None = None,
) -> Protocol:
    """
    Reconstitute and normalise DNA in Twist plate(s).

    @tag: DNA
    """
    plates = params.plates

    if not protocol:
        protocol = Protocol.from_layout(
            name="DNA Reconstitution",
            layout_file=LAYOUTS_PATH / Path("reconstitute.lay"),
            simulator_mode=simulate,
        )
    if not protocol.deck:
        msg = "Deck layout not loaded. Check that layouts/max.lay exists."
        raise ValueError(msg)

    # stacks
    rec_plates_src = protocol.deck.get_plate_stack("F1")[: len(plates)]
    rec_plates_dst = protocol.deck.get_plate_stack("F3")[: len(plates)][::-1]
    rec_tips_src = protocol.deck.get_tip_rack_stack("E2")
    dil_plates_src = protocol.deck.get_plate_stack("F2")[: len(plates)]
    dil_plates_dst = protocol.deck.get_plate_stack("F4")[: len(plates)][::-1]
    dil_tips_src = protocol.deck.get_tip_rack_stack("E3")[: len(plates)]

    # plates
    rec_plate = protocol.deck.get_plate("C4")
    dil_plate = protocol.deck.get_plate("C3")

    # tips
    holder_tips = protocol.deck.get_tip_rack("A4")
    dil_tips = protocol.deck.get_tip_rack("D2")

    # reservoirs
    water = protocol.deck.get_reservoir("B5")
