import math
from pathlib import Path

from adaptyv_lab import Protocol
from pydantic import BaseModel, Field, model_validator

from hamilton_protocols import LAYOUTS_PATH
from hamilton_protocols.utils import alpha_to_index


class MaxPlateParams(BaseModel):
    """Parameters for configuring a MAX plate."""

    columns: int = Field(
        default=2,
        ge=1,
        le=12,
        description="Number of expression columns",
        title="Columns",
    )
    rows: int = Field(
        default=8, ge=1, le=8, description="Number of expression rows", title="Rows"
    )
    well_volume: float = Field(
        default=250.0,
        ge=180.0,
        le=250.0,
        description="Volume of each well in μL",
        title="Well Volume (μL)",
    )

    model_config = {
        "title": "MAX Plate Configuration",
        "json_schema_extra": {
            "presets": [
                {
                    "name": "Pivot",
                    "values": {
                        "columns": 2,
                        "rows": 8,
                        "well_volume": 250.0,
                    },
                },
                {
                    "name": "Pro",
                    "values": {
                        "columns": 4,
                        "rows": 8,
                        "well_volume": 250.0,
                    },
                },
            ]
        },
    }


class MaxPlateProtocolParams(BaseModel):
    """Parameters for the MAX plate preparation protocol."""

    plates: list[MaxPlateParams] = Field(
        default_factory=list,
        description="List of MAX plate configurations",
        max_length=4,
        title="MAX Plates",
    )

    class Config:
        title = "MAX Plate Protocol Parameters"


class LoadingPlateParams(BaseModel):
    """Parameters for configuring a loading plate."""

    columns: int = Field(
        default=12,
        ge=1,
        le=12,
        description="Number of expression columns",
        title="Columns",
    )
    rows: int = Field(
        default=8, ge=1, le=8, description="Number of expression rows", title="Rows"
    )
    source_well: str = Field(
        default="A1", description="Source well on expression plate", title="Source Well"
    )
    destination_well: str = Field(
        default="A1",
        description="Destination well on loading plate",
        title="Destination Well",
    )
    expression_df: int = Field(
        default=5,
        ge=1,
        description="Dilution factor for expression",
        title="Expression D_f",
    )
    replicates: int = Field(
        default=1, ge=1, description="Number of replicates", title="Replicates"
    )
    final_df: int = Field(
        default=40,
        ge=1,
        description="Final dilution factor of expression",
        title="Final D_f",
    )
    dilution_buffer: str = Field(
        default="K",
        description="Dilution buffer",
        pattern="^[KL]$",
        title="Dilution Buffer",
    )
    expression_volume: float = Field(
        default=6.0,
        ge=1.0,
        le=10.0,
        description="Volume of expressed proteins",
        title="Expression Volume (μL)",
    )
    diluted: bool = Field(
        default=True,
        description="Whether the expression plate is already diluted",
        title="Diluted",
    )
    well_volume: float = Field(
        default=80.0,
        ge=40.0,
        le=90.0,
        description="Volume of each well in μL",
        title="Well Volume (μL)",
    )

    @property
    def diluted_expression_volume(self) -> float:
        """Volume of diluted expression to add to each well in μL."""
        return self.well_volume / (self.final_df / self.expression_df)

    @property
    def dilution_volume(self) -> float:
        """Volume of d buffer to add to each expression well in μL."""
        return self.expression_df * self.expression_volume - self.expression_volume

    @model_validator(mode="after")
    def validate_wells(self):
        """Check that source & destination well don't conflict with plate size."""
        src_row, src_col = alpha_to_index(self.source_well)
        dst_row, dst_col = alpha_to_index(self.destination_well)
        if src_col + self.columns * 2 > 24 or src_row + self.rows * 2 > 16:
            raise ValueError(
                f"Source well {self.source_well} exceeds plate dimensions."
            )
        if dst_col + self.columns * 2 > 24 or dst_row + self.rows * 2 > 16:
            raise ValueError(
                f"Destination well {self.destination_well} exceeds plate dimensions."
            )
        if self.columns * self.replicates > 12:
            raise ValueError(
                f"Number of columns ({self.columns}) and replicates ({self.replicates}) exceeds plate dimensions."
            )
        return self

    class Config:
        title = "Loading Plate Configuration"


class SampleParams(BaseModel):
    """Parameters for configuring a sample in a sample plate."""

    dilution_factor: float = Field(
        default=round(math.sqrt(10), 2),
        ge=1.0,
        description="Final dilution factor of expression",
        title="D_f",
    )
    volume: float = Field(
        default=40.0,
        ge=40.0,
        le=90.0,
        description="Volume of each sample well in μL",
        title="Sample Volume (μL)",
    )
    rows: int = Field(
        default=8, ge=1, le=8, description="Number of sample rows", title="Rows"
    )

    @property
    def transfer_volume(self) -> float:
        """Volume of sample to transfer to each well in μL."""
        return round(self.volume / (self.dilution_factor - 1), 2)

    @property
    def initial_conc_volume(self) -> float:
        """Volume of initial concentration sample in μL."""
        return self.transfer_volume + self.volume

    class Config:
        title = "Analyte Configuration"


class SamplePlateParams(BaseModel):
    """Parameters for configuring a sample plate."""

    samples: list[SampleParams] = Field(
        default_factory=lambda: [SampleParams()],
        description="List of sample configurations",
        min_length=1,
        max_length=8,
        title="Samples",
    )
    buffer_volume: float = Field(
        default=80.0,
        ge=40.0,
        le=90.0,
        description="Volume of each buffer well in μL",
        title="Buffer Volume (μL)",
    )
    dilution_buffer: str = Field(
        default="K",
        description="Dilution buffer (K or L)",
        pattern="^[KL]$",
        title="Dilution Buffer",
    )
    columns: int = Field(
        default=2, ge=1, le=4, description="Number of sample columns", title="Columns"
    )
    concentrations: int = Field(
        default=4,
        ge=1,
        le=6,
        description="Number of concentrations",
        title="Concentrations",
    )

    @model_validator(mode="after")
    def validate_samples(self):
        """Check that the number of samples doesn't exceed the plate size."""
        rows = sum(sample.rows for sample in self.samples)
        if rows > 8:
            raise ValueError("Total number of rows exceeds 8.")

        return self

    @property
    def c_plate(self) -> bool:
        """Check if the sample preparation requires a C plate."""
        return (self.concentrations + 1) * self.columns > 12

    @property
    def rows(self) -> int:
        """Total number of rows in the sample plate."""
        rows = sum(sample.rows for sample in self.samples)
        return rows

    class Config:
        title = "Sample Plate Configuration"


class BLIPlatePrepParams(BaseModel):
    """Parameters for the BLI plate preparation protocol.

    This protocol chains together the preparation of loading plates,
    and sample plates for Bio-Layer Interferometry (BLI) experiments.
    """

    loading_plates: list[LoadingPlateParams] = Field(
        default_factory=list,
        max_length=4,
        description="List of loading plate configurations. These plates contain the protein samples to be loaded onto the biosensors.",
        title="Loading Plates",
    )
    sample_plates: list[SamplePlateParams] = Field(
        default_factory=list,
        max_length=4,
        description="List of sample plate configurations. These plates contain the analyte samples at different concentrations.",
        title="Sample Plates",
    )

    class Config:
        title = "BLI Plate Preparation Parameters"


def max_plate_protocol(
    params: MaxPlateProtocolParams,
    simulate: bool = False,
    protocol: Protocol | None = None,
) -> Protocol:
    """Prepare MAX plates for BLI experiment.

    This protocol sets up plates containing buffer and regeneration solution for
    the biosensor tips.

    @tag: BLI
    """
    plates = params.plates

    if not protocol:
        protocol = Protocol.from_layout(
            name="MAX Plate Protocol",
            layout_file=LAYOUTS_PATH / Path("max.lay"),
            simulator_mode=simulate,
        )
    if not protocol.deck:
        msg = "Deck layout not loaded. Check that layouts/max.lay exists."
        raise ValueError(msg)

    max_plates_src = protocol.deck.get_plate_stack("F1")[: len(plates)]
    max_plates_dst = protocol.deck.get_plate_stack("F2")[: len(plates)][::-1]
    max_plate = protocol.deck.get_plate("E4")
    k_buffer = protocol.deck.get_reservoir("B3")
    regeneration_buffer = protocol.deck.get_reservoir("B1")
    holder_tips = protocol.deck.get_tip_rack("A4")
    buffer_tips = protocol.deck.get_tip_rack("A3")
    hv_tips = protocol.deck.get_tip_rack("E2")
    regen_tips = hv_tips[::2, 0]

    if not all(
        tip is not None for tip in [buffer_tips.tip, regen_tips.tip, holder_tips.tip]
    ):
        msg = "Tips not loaded. Check that A3, E2, and A4 are tip racks."
        raise ValueError(msg)

    protocol.initialize()
    protocol.pickup_tips(buffer_tips).eject_tips(holder_tips)

    for plate in plates:
        probe_columns = plate.columns
        probe_rows = plate.rows
        well_volume = plate.well_volume

        protocol.grip_get(max_plates_src.pop(), grip_width=81.5).grip_place(max_plate)

        protocol.transfer(
            source=k_buffer,
            destination=max_plate[: probe_rows * 2 : 2, : probe_columns * 2 : 2],
            tips=holder_tips[-probe_rows * 2 :: 2, -probe_columns * 2 :: 2],
            volume=well_volume,
        )

        protocol.pickup_tips(regen_tips)
        for col in range(probe_columns):
            for row in range(probe_rows // 4):
                protocol.aspirate(regeneration_buffer[::4, -1], volume=well_volume)
                protocol.dispense(
                    max_plate[row::2, probe_columns * 2 + col], volume=well_volume
                )
        protocol.eject_tips(regen_tips)

        protocol.grip_get(max_plate, grip_width=81.5).grip_place(max_plates_dst.pop())

    protocol.pickup_tips(regen_tips).eject_tips(mode=1)
    protocol.pickup_tips(holder_tips).eject_tips(buffer_tips)

    return protocol


def bli_plate_prep_protocol(
    params: BLIPlatePrepParams,
    simulate: bool = False,
    protocol: Protocol | None = None,
) -> Protocol:
    """Prepare all plates for a BLI experiment in one protocol.

    This protocol chains together the preparation of loading plates and
    sample plates for BLI experiments.

    @tag: BLI
    """
    protocol = Protocol.from_layout(
        name="BLI Plate Prep Protocol",
        layout_file=LAYOUTS_PATH / Path("bli.lay"),
        simulator_mode=simulate,
    )
    if not protocol.deck:
        msg = "Deck layout not loaded. Check that layouts/bli.lay exists."
        raise ValueError(msg)

    loading_plate_params = params.loading_plates
    sample_plate_params = params.sample_plates

    n_loading_plates = len(loading_plate_params)
    n_sample_plates = sum(2 if plate.c_plate else 1 for plate in sample_plate_params)
    n_gator_plates = n_loading_plates + n_sample_plates

    if not protocol:
        protocol = Protocol.from_layout(
            name="Loading Plate Protocol",
            layout_file=LAYOUTS_PATH / Path("bli.lay"),
            simulator_mode=simulate,
        )
    if not protocol.deck:
        msg = "Deck layout not loaded. Check that layouts/bli.lay exists."
        raise ValueError(msg)

    # plates
    gator_plate_src = [
        plate
        for stack in ["F4", "F5"]
        for plate in protocol.deck.get_plate_stack(stack)
    ][:n_gator_plates]
    gator_plate_dst = [
        plate
        for stack in ["F3", "F5"]
        for plate in protocol.deck.get_plate_stack(stack)
    ][:n_gator_plates][::-1]
    exp_plate_src = protocol.deck.get_plate_stack("F2")[:n_loading_plates]
    exp_plate_dst = protocol.deck.get_plate_stack("F1")[:n_loading_plates][::-1]
    a_plate = b_plate = protocol.deck.get_plate("E5")
    c_plate = protocol.deck.get_plate("E4")
    exp_plate = protocol.deck.get_plate("C3")

    # tips
    holder_tips = protocol.deck.get_tip_rack("A4")
    k_buffer_tips = protocol.deck.get_tip_rack("A3")
    l_buffer_tips = protocol.deck.get_tip_rack("A2")
    hv_tips = protocol.deck.get_tip_rack("E2")
    loading_tips_src = protocol.deck.get_tip_rack_stack("E3")[:n_loading_plates]
    loading_tips = protocol.deck.get_tip_rack("D2")
    sample_tips = protocol.deck.get_tip_rack("A1")

    # reservoirs
    l_buffer = protocol.deck.get_reservoir("B4")
    k_buffer = protocol.deck.get_reservoir("B3")
    d_buffer = protocol.deck.get_reservoir("B2")

    # carriers
    tube_carrier = protocol.deck.get_tube_carrier("C1")

    protocol.initialize()

    # loading
    for loading_plate in loading_plate_params:
        cols = loading_plate.columns
        rows = loading_plate.rows
        src_well = alpha_to_index(loading_plate.source_well)
        dst_well = alpha_to_index(loading_plate.destination_well)
        replicates = loading_plate.replicates
        well_vol = loading_plate.well_volume

        dil_vol = loading_plate.dilution_volume
        dil_exp_vol = loading_plate.diluted_expression_volume

        match loading_plate.dilution_buffer:
            case "K":
                buffer = k_buffer
                buffer_tips = k_buffer_tips
            case "L":
                buffer = l_buffer
                buffer_tips = l_buffer_tips
            case _:
                raise ValueError(
                    f"Invalid dilution buffer: {loading_plate.dilution_buffer}"
                )

        protocol.grip_get(gator_plate_src.pop()).grip_place(a_plate)
        protocol.pickup_tips(buffer_tips).eject_tips(holder_tips)

        for q in [(0, 1), (1, 0)]:
            protocol.transfer(
                source=buffer,
                destination=a_plate[q[0] :: 2, q[1] :: 2],
                tips=holder_tips,
                volume=well_vol,
            )
        protocol.transfer(
            source=buffer,
            destination=a_plate[::2, ::2],
            tips=holder_tips,
            volume=well_vol - dil_exp_vol,
        )

        protocol.pickup_tips(holder_tips).eject_tips(buffer_tips)

        protocol.grip_get(loading_tips_src.pop()).grip_place(loading_tips)
        protocol.grip_get(exp_plate_src.pop()).grip_place(exp_plate)

        protocol.pickup_tips(loading_tips).eject_tips(holder_tips)
        if not loading_plate.diluted:
            protocol.transfer(
                source=d_buffer,
                destination=exp_plate[
                    src_well[0] : src_well[0] + rows,
                    src_well[1] * cols : src_well[1] + cols,
                ],
                tips=holder_tips[-rows * 2 :, -cols * 2 :],
                volume=dil_vol,
            )
        for i in range(replicates):
            protocol.transfer(
                source=exp_plate[
                    src_well[0] : src_well[0] + rows, src_well[1] : src_well[1] + cols
                ],
                destination=a_plate[
                    dst_well[0] * 2 : (dst_well[0] + rows) * 2 : 2,
                    (dst_well[1] + i * cols) * 2 : (dst_well[1] + (i + 1) * cols)
                    * 2 : 2,
                ],
                tips=holder_tips[-rows * 2 :, -cols * 2 :],
                volume=dil_exp_vol,
            )
        protocol.pickup_tips(holder_tips).eject_tips(loading_tips)

        protocol.grip_get(loading_tips).grip_place(loading_tips, waste=True)
        protocol.grip_get(a_plate).grip_place(gator_plate_dst.pop())
        protocol.grip_get(exp_plate).grip_place(exp_plate_dst.pop())

    # sample
    for i, sample_plate in enumerate(sample_plate_params):
        buffer_vol = sample_plate.buffer_volume
        cols = sample_plate.columns
        rows = sample_plate.rows
        n_conc = sample_plate.concentrations

        match sample_plate.dilution_buffer:
            case "K":
                buffer = k_buffer
                buffer_tips = k_buffer_tips
            case "L":
                buffer = l_buffer
                buffer_tips = l_buffer_tips
            case _:
                raise ValueError(
                    f"Invalid dilution buffer: {sample_plate.dilution_buffer}"
                )

        protocol.pickup_tips(buffer_tips).eject_tips(holder_tips)

        protocol.grip_get(gator_plate_src.pop()).grip_place(b_plate)
        if sample_plate.c_plate:
            protocol.grip_get(gator_plate_src.pop()).grip_place(c_plate)

        for q in [(0, 1), (1, 0), (1, 1)]:
            protocol.transfer(
                source=buffer,
                destination=b_plate[q[0] :: 2, q[1] :: 2],
                tips=holder_tips,
                volume=buffer_vol,
            )
        if sample_plate.c_plate:
            for q in [(0, 1), (1, 0), (1, 1)]:
                protocol.transfer(
                    source=buffer,
                    destination=c_plate[q[0] :: 2, q[1] :: 2],
                    tips=holder_tips,
                    volume=buffer_vol,
                )

        samples = sample_plate.samples
        for j, sample in enumerate(samples):
            rows = sample.rows
            transfer_vol = sample.transfer_volume
            init_conc_vol = sample.initial_conc_volume
            sample_vol = sample.volume

            row_offset = sum(samples[k].rows for k in range(j))
            col_offset = sum(sample_plate_params[k].columns for k in range(i))
            tip_offset = sum(len(sample_plate_params[k].samples) for k in range(i))

            protocol.transfer(
                source=buffer,
                destination=b_plate[
                    row_offset * 2 : (rows + row_offset) * 2 : 2,
                    cols * 2 :: 2,
                ],
                tips=holder_tips[-rows * 2 :: 2, cols * 2 :: 2],
                volume=sample_vol,
            )
            if sample_plate.c_plate:
                protocol.transfer(
                    source=buffer,
                    destination=c_plate[
                        row_offset * 2 : (rows + row_offset) * 2 : 2,
                        ::2,
                    ],
                    tips=holder_tips[-rows * 2 :: 2, ::2],
                    volume=sample_vol,
                )

            protocol.aliquot(
                source=tube_carrier[j + tip_offset, 0],
                destination=b_plate[
                    row_offset * 2 : (rows + row_offset) * 2 : 2,
                    : cols * 2 : 2,
                ],
                tips=hv_tips[j + tip_offset, 0],
                volume=init_conc_vol,
            )

            protocol.pickup_tips(holder_tips).eject_tips(buffer_tips)
            protocol.pickup_tips(sample_tips).eject_tips(holder_tips)

            protocol.pickup_tips(
                holder_tips[
                    -(rows + row_offset) * 2 :: 2, -(cols + col_offset) * 2 :: 2
                ]
            )
            wells = b_plate[row_offset * 2, : cols * 2 * n_conc : cols * 2].to_list()
            if sample_plate.c_plate:
                wells += c_plate[
                    row_offset * 2, : (n_conc - len(wells)) * cols * 2 : cols * 2
                ].to_list()

            for w_idx in range(len(wells) - 1):
                protocol.aspirate(wells[w_idx], volume=transfer_vol).dispense(
                    wells[w_idx + 1], volume=transfer_vol
                )
            protocol.aspirate(wells[-1], volume=transfer_vol)
            protocol.eject_tips(mode=2)

            protocol.pickup_tips(holder_tips).eject_tips(sample_tips)
        protocol.grip_get(b_plate).grip_place(gator_plate_dst.pop())
        if sample_plate.c_plate:
            protocol.grip_get(c_plate).grip_place(gator_plate_dst.pop())

    return protocol
