from pathlib import Path
from adaptyv_lab import Protocol
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List
import pandas as pd
import math

from hamilton_protocols import LAYOUTS_PATH
from hamilton_protocols.api.main import CSVData
from hamilton_protocols.utils import alpha_to_index

from adaptyv_lab.config.liquid import DispenseMode, AspirateMode


class DNAPlateParams(BaseModel):
    """
    Parameters for configuring a Twist plate with CSV data.
    """

    plate_id: str = Field(..., description="ID of the plate to be twisted.")
    csv_data: str = Field(
        ..., description="Base64-encoded CSV containing well mapping information"
    )
    dilute: bool = Field(False, description="Whether to dilute the samples")
    diluted_concentration: Optional[float] = Field(
        None, description="Concentration of diluted samples"
    )

    @field_validator("csv_data")
    @classmethod
    def validate_csv_data(cls, v):
        if v is None:
            return v

        try:
            data = CSVData(v)
        except Exception as e:
            raise ValueError(f"Invalid CSV data: {str(e)}")

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
        return dict(zip(self.df["well"], self.df["volume"]))


class DNAReconstitutionParams(BaseModel):
    """
    Parameters for DNA reconstitution with multiple plates and CSV data.
    """

    plates: List[DNAPlateParams] = Field(
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

    for i, plate_params in enumerate(plates):
        mapping = plate_params.mapping
        max_col = max([alpha_to_index(well)[1] for well in mapping.keys()]) + 1

        protocol.grip_get(rec_plates_src.pop()).grip_place(rec_plate)
        # protocol.grip_get(dil_plates_src.pop()).grip_place(dil_plate)

        for col in range(max_col):
            for row in range(2):
                tips = rec_tips_src[-1][row::2, col]
                plate = rec_plate[row::2, col]
                volumes = [
                    min(mapping.get(pos.alphanumeric, 0), tips.tip.max_volume)
                    for pos in plate.positions
                ]
                protocol.pickup_tips(tips)
                protocol.aspirate(water[::4, -1], volume=volumes)
                protocol.dispense(plate, volume=volumes)
                protocol.eject_tips(mode=1)

        protocol.grip_get(rec_plate).grip_place(rec_plates_dst.pop())
        # protocol.grip_get(dil_plate).grip_place(dil_plates_dst.pop())

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
            raise ValueError(f"Invalid CSV data: {str(e)}")

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
                    group["name"].tolist(), group["destination"].tolist()
                )
            ]
            mapping[plate_well.split(":")[0]][plate_well.split(":")[1]].extend(
                destination
            )
        return mapping

    @property
    def dil_plates(self) -> List[str]:
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
    hv_tip_stack = protocol.deck.get_tip_rack_stack("E2")[:1]
    lv_tip_stack = protocol.deck.get_tip_rack_stack("E1")[:1]

    # carriers
    tube_carrier = protocol.deck.get_tube_carrier("C1")

    lv_tip_idx = 0
    hv_tip_idx = 0

    protocol.initialize()

    for plate_params in plates:
        mapping = plate_params.mapping
        protocol.grip_get(fab_plates_src.pop()).grip_place(fab_plate)

        for src_plate in mapping:
            print(f"Adding chains from {src_plate}")
            protocol.grip_get(dil_plates_src.pop()).grip_place(dil_plate)
            for src_well in mapping[src_plate]:
                print(f"  using well {src_well}")
                if lv_tip_idx > len(lv_tip_stack[-1]):
                    protocol.grip_get(lv_tip_stack.pop()).grip_place(
                        lv_tip_stack[-1], waste=True
                    )
                    lv_tip_idx = 0
                if hv_tip_idx > len(hv_tip_stack[-1]):
                    protocol.grip_get(hv_tip_stack.pop()).grip_place(
                        hv_tip_stack[-1], waste=True
                    )
                    hv_tip_idx = 0

                tip_vol = 0
                n_dest = len(mapping[src_plate][src_well])
                if lv_tip_stack[-1].tip.max_volume >= n_dest * 5:
                    protocol.pickup_tips(lv_tip_stack[-1].at(lv_tip_idx))
                    lv_tip_idx += 1
                else:
                    protocol.pickup_tips(hv_tip_stack[-1].at(hv_tip_idx))
                    hv_tip_idx += 1

                src = dil_plate[alpha_to_index(src_well)]
                if src_well in well_overrides[src_plate].keys():
                    new_well = well_overrides[src_plate][src_well]
                    src = tube_carrier.at(new_well)
                    print(f"  using tube {new_well + 1} instead of {src_well}")

                for idx, dest in enumerate(mapping[src_plate][src_well]):
                    print(f"    adding {dest['name']} to {dest['well']}")
                    asp_vol = min(
                        hv_tip_stack[-1].tip.max_volume,
                        (math.floor(hv_tip_stack[-1].tip.max_volume / 5) * 5),
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
