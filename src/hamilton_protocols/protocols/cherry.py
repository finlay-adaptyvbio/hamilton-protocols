import pandas as pd
from adaptyv_lab import Protocol
from pydantic import BaseModel, Field, field_validator

from hamilton_protocols import LAYOUTS_PATH
from hamilton_protocols.api.main import CSVData


class CherryPickPlateParams(BaseModel):
    """Parameters for configuring a Cherry Pick plate."""

    plate_id: str = Field(..., description="The ID of the plate.")
    csv_data: str = Field(
        ...,
        description="CSV data containing the mapping of source to destination wells.",
    )

    @field_validator("csv_data")
    @classmethod
    def validate_csv_data(cls, v):
        if v is None:
            return v

        try:
            data = CSVData(v)
        except Exception as e:
            raise ValueError(f"Invalid CSV data: {e!s}")

        required_columns = [
            "Source Plate",
            "Source Well",
            "Destination Plate",
            "Destination Well",
        ]
        missing = [col for col in required_columns if col not in data.df.columns]
        if missing:
            raise ValueError(
                f"Well mapping CSV missing required columns: {', '.join(missing)}"
            )
        return v

    @property
    def df(self) -> pd.DataFrame:
        """Parse well mapping CSV data into a pandas DataFrame."""
        return CSVData(self.csv_data).df

    @property
    def mapping(self) -> dict[str, dict[str, str]]:
        """Get a mapping of source wells to destination wells."""
        plate_map = {}
        for src_plate_id in self.df["Source Plate"].unique():
            if src_plate_id not in plate_map:
                plate_map[src_plate_id] = {}
            src_wells = self.df[self.df["Source Plate"] == src_plate_id][
                "Source Well"
            ].to_list()
            dst_wells = self.df[self.df["Source Plate"] == src_plate_id][
                "Destination Well"
            ].to_list()
            well_map = {
                src_well: dst_well
                for src_well, dst_well in zip(src_wells, dst_wells, strict=False)
            }
            plate_map[src_plate_id] = well_map
        return plate_map


class CherryPickProtocolParams(BaseModel):
    """Parameters for configuring a Cherry Pick protocol."""

    plates: list[CherryPickPlateParams] = Field(
        default_factory=list,
        title="Cherry Pick Plates",
        description="List of Cherry Pick plate parameters.",
        min_length=1,
        max_length=4,
    )

    @property
    def source_plates(self) -> list[str]:
        """Get the source plate names from the parameters."""
        return sorted(
            {
                source_plate
                for plate in self.plates
                for source_plate in plate.df["Source Plate"].unique()
            }
        )


def cherry_pick_protocol(
    params: CherryPickProtocolParams,
    simulate: bool = False,
    protocol: Protocol | None = None,
) -> Protocol:
    """Cherry pick protocol"""
    protocol = Protocol.from_layout(
        name="Cherry Pick Protocol",
        layout_file=LAYOUTS_PATH / "cherry-pick.lay",
        simulator_mode=simulate,
    )
    if not protocol.deck:
        msg = "Protocol deck is not defined. Please check the layout file."
        raise ValueError(msg)

    plates = params.plates
    src_plate_ids = params.source_plates

    tip_rack = protocol.deck.get_tip_rack("E1")
    src_plates = protocol.deck.get_plate_stack("F4")[: len(src_plate_ids)]
    dst_plates = protocol.deck.get_plate_stack("F3")[: len(src_plate_ids)][::-1]
    src_plate = protocol.deck.get_plate("C3")
    dst_plate = protocol.deck.get_plate("C2")

    tips = tip_rack.iterlabware()

    for plate in plates:
        mapping = plate.mapping
        for src_plate_id, src_plate_map in mapping.items():
            src_plate_wells = [k for k in src_plate_map.keys()]

            protocol.grip_get(src_plates.pop()).grip_place(src_plate)

            for group in src_plate[src_plate_wells].itergroups():
                group_src_wells = [pos.alphanumeric for pos in group]
                group_dst_wells = [src_plate_map[pos.alphanumeric] for pos in group]
                tips_to_use = next(tips)
                protocol.pickup_tips(tips_to_use)
                protocol.aspirate(src_plate[group_src_wells], 20).dispense(
                    dst_plate[group_dst_wells], 20
                )
                protocol.eject_tips(tips_to_use)

            protocol.grip_get(src_plate).grip_place(dst_plates.pop())
    return protocol
