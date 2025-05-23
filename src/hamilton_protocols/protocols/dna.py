import math
from pathlib import Path

import pandas as pd
from adaptyv_lab import Protocol
from pydantic import BaseModel, Field, field_validator

from hamilton_protocols import LAYOUTS_PATH
from hamilton_protocols.api.main import CSVData
from hamilton_protocols.utils import alpha_to_index


class DNAPlateParams(BaseModel):
    """
    Parameters for configuring a Twist plate with CSV data.
    """

    plate_id: str = Field(..., description="ID of the plate to be twisted.")
    csv_data: str = Field(
        ..., description="Base64-encoded CSV containing well mapping information"
    )
    dilute: bool = Field(default=False, description="Whether to dilute the samples")
    diluted_concentration: float | None = Field(
        default=None, description="Concentration of diluted samples"
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

        required_columns = ["well", "volume"]
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
    def mapping(self) -> dict[str, float]:
        """Get well mapping as a dictionary."""
        return dict(zip(self.df["well"], self.df["volume"], strict=False))


class DNAReconstitutionParams(BaseModel):
    """
    Parameters for DNA reconstitution with multiple plates and CSV data.
    """

    plates: list[DNAPlateParams] = Field(
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
        msg = "Deck layout not loaded. Check that layouts/reconstitute.lay exists."
        raise ValueError(msg)

    # stacks
    rec_plates_src = protocol.deck.get_plate_stack("F1")[: len(plates)]
    rec_plates_dst = protocol.deck.get_plate_stack("F3")[: len(plates)][::-1]
    rec_tips_src = protocol.deck.get_tip_rack_stack("E1")[: len(plates)]
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

    protocol.initialize()

    col_idx = 0

    for i, plate_params in enumerate(plates):
        plate_id = plate_params.plate_id
        print(f"Reconstituting {plate_id}")
        mapping = plate_params.mapping
        max_col = max([alpha_to_index(well)[1] for well in mapping.keys()]) + 1

        protocol.grip_get(rec_plates_src.pop()).grip_place(rec_plate)
        # protocol.grip_get(dil_plates_src.pop()).grip_place(dil_plate)

        for col in range(max_col):
            if col_idx >= rec_tips_src[-1].definition.columns:
                protocol.grip_get(rec_tips_src.pop()).grip_place(
                    rec_tips_src[-1], waste=True
                )
                col_idx = 0
            for row in range(2):
                tips = rec_tips_src[-1][row::2, col_idx]
                plate = rec_plate[row::2, col]
                volumes = [
                    min(mapping.get(pos.alphanumeric, 0), tips.tip.max_volume)
                    for pos in plate.positions
                ]
                protocol.pickup_tips(tips)
                protocol.aspirate(water[::4, -1], volume=volumes)
                protocol.dispense(plate, volume=volumes)
                protocol.eject_tips(mode=1)

            col_idx += 1

        protocol.grip_get(rec_plate).grip_place(rec_plates_dst.pop())
        # protocol.grip_get(dil_plate).grip_place(dil_plates_dst.pop())

        # clean up
        protocol.grip_eject()

    return protocol


class DNADilutionSourcePlateParams(BaseModel):
    """
    Parameters for DNA dilution with multiple plates and CSV data.
    """

    plate_id: str = Field(..., description="ID of the plate to be diluted.")
    source_well: str = Field(
        default="A1", description="Source well for the dilution (e.g., A1, B2, etc.)"
    )
    destination_well: str = Field(
        default="A1",
        description="Destination well for the dilution (e.g., A1, B2, etc.)",
    )
    rows: int = Field(
        default=8, ge=1, le=8, description="Number of rows to be diluted (1-8)."
    )
    cols: int = Field(
        default=12, ge=1, le=12, description="Number of columns to be diluted (1-12)."
    )
    final_concentration: float = Field(
        default=4.0, description="Final concentration of the diluted sample"
    )
    final_volume: float = Field(
        default=100.0, description="Final volume of the diluted sample"
    )
    stock_concentration: float = Field(
        default=100.0, description="Concentration of the stock solution"
    )


class DNADilutionDestinationPlateParams(BaseModel):
    """
    Parameters for DNA dilution with multiple source plates per dilution plate.
    """

    plate_id: str = Field(..., description="ID of the final dilution plate.")
    source_plates: list[DNADilutionSourcePlateParams] = Field(
        default_factory=list,
        description="List of source plates for the dilution.",
        title="Source Plates",
    )


class DNADilutionProtocolParams(BaseModel):
    """
    Parameters for DNA dilution with multiple dilution plates and source plates.
    Each dilution plate can have multiple source plates.
    """

    plates: list[DNADilutionDestinationPlateParams] = Field(
        default_factory=list,
        description="List of dilution plate parameters for DNA dilution.",
        max_length=6,
        title="Dilution Plates",
    )

    @property
    def source_plates(self) -> list[str]:
        """Get the list of source plates from the dilution plates."""
        return sorted(
            {
                source_plate.plate_id
                for plate in self.plates
                for source_plate in plate.source_plates
            }
        )


def dna_dilution_protocol(
    params: DNADilutionProtocolParams,
    simulate: bool = False,
    protocol: Protocol | None = None,
) -> Protocol:
    """
    Dilute DNA from reconstituted plates with optional remapping of wells.

    @tag: DNA
    """
    plates = params.plates
    source_plate_ids = params.source_plates
    print(source_plate_ids)

    if not protocol:
        protocol = Protocol.from_layout(
            name="DNA Reconstitution",
            layout_file=LAYOUTS_PATH / Path("reconstitute.lay"),
            simulator_mode=simulate,
        )
    if not protocol.deck:
        msg = "Deck layout not loaded. Check that layouts/reconstitute.lay exists."
        raise ValueError(msg)

    # stacks
    rec_plates_src = protocol.deck.get_plate_stack("F1")[: len(source_plate_ids)]
    rec_plates_dst = protocol.deck.get_plate_stack("F3")[: len(source_plate_ids)][::-1]
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

    protocol.initialize()

    for plate_params in plates:
        source_plates = plate_params.source_plates
        print(f"Creating {plate_params.plate_id}")
        protocol.grip_get(dil_plates_src[-1]).grip_place(dil_plate)
        protocol.grip_get(dil_tips_src.pop()).grip_place(dil_tips)
        protocol.pickup_tips(dil_tips).eject_tips(holder_tips)
        for i, source_plate in enumerate(source_plates):
            print(
                f"  Source Plate: {source_plate.plate_id} at index {source_plate_ids.index(source_plate.plate_id)}"
            )
            print(f"    Source Well: {source_plate.source_well}")
            print(f"    Destination Well: {source_plate.destination_well}")
            print(f"    Rows: {source_plate.rows}")
            print(f"    Columns: {source_plate.cols}")

            rows = source_plate.rows
            cols = source_plate.cols

            col_offset = sum(source_plates[k].cols for k in range(i))
            row_offset = sum(source_plates[k].rows for k in range(i))

            print(row_offset, col_offset)

            protocol.grip_get(rec_plates_src.pop()).grip_place(rec_plate)

            print(
                holder_tips[
                    -(row_offset + rows) * 2 :: 2, -(col_offset + cols) * 2 :: 2
                ]
            )

    return protocol


class FAbMappingParams(BaseModel):
    """Parameters for FAb mapping."""

    plate_id: str = Field(..., description="ID of the target plate")
    csv_data: str = Field(
        ..., description="Base64-encoded CSV file containing FAb mapping information"
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

        required_columns = ["name", "destination", "plate:well"]
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
    def mapping(self) -> dict[str, dict[str, list[dict[str, str]]]]:
        """Parse well mapping CSV data into a dictionary."""
        df = self.df
        df["plate"] = df["plate:well"].str.split(":").str[0]
        df["well"] = df["plate:well"].str.split(":").str[1]
        mapping = {
            plate: {well: [] for well in df[df["plate"] == plate]["well"].unique()}
            for plate in df["plate"].unique()
        }
        for (plate_well), group in df.groupby("plate:well"):
            destination = [
                {"name": name, "well": dest}
                for name, dest in zip(
                    group["name"].tolist(), group["destination"].tolist(), strict=False
                )
            ]
            mapping[plate_well.split(":")[0]][plate_well.split(":")[1]].extend(
                destination
            )
        return mapping

    @property
    def dil_plates(self) -> list[str]:
        """Get the list of dilution plates from the mapping."""
        return list(self.mapping.keys())


class FAbMappingProtocolParams(BaseModel):
    """Parameters for FAb mapping."""

    plates: list[FAbMappingParams] = Field(
        default_factory=list, title="Twist Plates", max_length=4
    )


def fab_mapping_protocol(
    params: FAbMappingProtocolParams,
    simulate: bool = False,
    protocol: Protocol | None = None,
) -> Protocol:
    """
    Mix heavy and light chain libraries in a plate.

    @tag: DNA
    """
    plates = params.plates

    if not protocol:
        protocol = Protocol.from_layout(
            name="FAb Mapping",
            layout_file=LAYOUTS_PATH / Path("fab-map.lay"),
            simulator_mode=simulate,
        )
    if not protocol.deck:
        msg = "Deck layout not loaded. Check that layouts/fab-map.lay exists."
        raise ValueError(msg)

    n_dil_plates = sum(len(plate.dil_plates) for plate in plates)
    n_fab_plates = len(plates)

    # well overrides
    well_overrides = {
        "P-DIL-0160": {
            "A3": 0,
            "D3": 1,
            "C4": 2,
            "H5": 3,
            "D6": 4,
            "D8": 5,
        },
        "P-DIL-0161": {},
        "P-DIL-0162": {},
    }

    # stacks
    dil_plates_src = [
        plate
        for stack in ["F3", "F4"]
        for plate in protocol.deck.get_plate_stack(stack)
    ][:n_dil_plates]
    dil_plates_dst = [
        plate
        for stack in ["F4", "F5"]
        for plate in protocol.deck.get_plate_stack(stack)
    ][:n_dil_plates][::-1]
    fab_plates_src = protocol.deck.get_plate_stack("F1")[:n_fab_plates]
    fab_plates_dst = protocol.deck.get_plate_stack("F2")[:n_fab_plates][::-1]

    # plates
    dil_plate = protocol.deck.get_plate("C3")
    fab_plate = protocol.deck.get_plate("C2")

    # tips
    lv_tip_stack = protocol.deck.get_tip_rack_stack("E1")[:1]

    # carriers
    tube_carrier = protocol.deck.get_tube_carrier("C1")

    lv_tip_idx = 0

    protocol.initialize()

    for plate_params in plates:
        mapping = plate_params.mapping
        protocol.grip_get(fab_plates_src.pop()).grip_place(fab_plate)

        print(sum(len(v) for v in mapping.values()))

        for src_plate in mapping:
            print(f"Adding chains from {src_plate}")
            protocol.grip_get(dil_plates_src.pop()).grip_place(dil_plate)
            for src_well in mapping[src_plate]:
                print(f"  using well {src_well}")
                if lv_tip_idx >= len(lv_tip_stack[-1]):
                    protocol.grip_get(lv_tip_stack.pop()).grip_place(
                        lv_tip_stack[-1], waste=True
                    )
                    print("  getting new tip stack")
                    lv_tip_idx = 0

                tip_vol = 0
                n_dest = len(mapping[src_plate][src_well])
                protocol.pickup_tips(lv_tip_stack[-1].at(lv_tip_idx))
                lv_tip_idx += 1

                src = dil_plate[alpha_to_index(src_well)]
                if src_well in well_overrides[src_plate].keys():
                    new_well = well_overrides[src_plate][src_well]
                    src = tube_carrier.at(new_well)
                    print(f"  using tube {new_well + 1} instead of {src_well}")

                for idx, dest in enumerate(mapping[src_plate][src_well]):
                    print(f"    adding {dest['name']} to {dest['well']}")
                    asp_vol = min(
                        lv_tip_stack[-1].tip.max_volume,
                        (math.floor(lv_tip_stack[-1].tip.max_volume / 5) * 5),
                        5 * (n_dest - idx),
                    )
                    if tip_vol < 5:
                        protocol.aspirate(src, volume=asp_vol)
                        tip_vol += asp_vol

                    protocol.dispense(fab_plate[alpha_to_index(dest["well"])], volume=5)
                    tip_vol -= 5

                protocol.eject_tips(mode=1)

            protocol.grip_get(dil_plate).grip_place(dil_plates_dst.pop())

        protocol.grip_get(fab_plate).grip_place(fab_plates_dst.pop())

    return protocol
