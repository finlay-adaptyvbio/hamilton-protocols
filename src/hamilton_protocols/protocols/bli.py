import math
from pathlib import Path

from adaptyv_lab import Protocol
from pydantic import BaseModel, Field, field_validator
from hamilton_protocols import LAYOUTS_PATH


def alpha_to_index(alpha: str) -> tuple[int, int]:
    """Convert alphanumeric well position (e.g., A1) to row, column indices."""
    try:
        row = ord(alpha[0].upper()) - ord("A")
        col = int(alpha[1:]) - 1
    except (IndexError, ValueError):
        raise ValueError(f"Invalid well name: {alpha}")
    return row, col


def get_volume_per_channel(
    wells: int, num_channels: int, max_volume: float, volume_per_well: float
) -> list[float]:
    """Calculate the volume per channel based on total volume and number of channels."""
    if num_channels <= 0:
        raise ValueError("Number of channels must be greater than zero.")
    if volume_per_well > max_volume:
        raise ValueError("Volume per well exceeds maximum volume.")
    # Calculate wells per channel based on max volume
    wells_per_channel_max = int(max_volume / volume_per_well)

    # Check if all wells can be distributed
    if wells > wells_per_channel_max * num_channels:
        raise ValueError(
            f"Cannot distribute {wells} wells across {num_channels} channels with max {wells_per_channel_max} wells per channel."
        )

    # Initialize volumes for each channel
    volumes = [0.0] * num_channels

    # First, distribute wells evenly
    wells_per_channel_min = wells // num_channels
    remaining_wells = wells % num_channels

    for i in range(num_channels):
        # Allocate the minimum number of wells to each channel
        volumes[i] = wells_per_channel_min * volume_per_well

    # Distribute remaining wells one by one
    for i in range(remaining_wells):
        volumes[i] += volume_per_well

    return volumes


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

    class Config:
        title = "MAX Plate Configuration"


class MaxPlateProtocolParams(BaseModel):
    """Parameters for the MAX plate preparation protocol."""

    plates: list[MaxPlateParams] = Field(
        default_factory=lambda: [MaxPlateParams()],
        description="List of MAX plate configurations",
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
        title="Expression Dilution Factor",
    )
    replicates: int = Field(
        default=1, ge=1, description="Number of replicates", title="Replicates"
    )
    final_df: int = Field(
        default=80,
        ge=1,
        description="Final dilution factor of expression",
        title="Final Dilution Factor",
    )
    dilution_buffer: str = Field(
        default="L",
        description="Dilution buffer",
        pattern="^[KL]$",
        title="Dilution Buffer",
    )
    well_volume: float = Field(
        default=80.0,
        ge=40.0,
        le=90.0,
        description="Volume of each well in μL",
        title="Well Volume (μL)",
    )

    @property
    def expression_volume(self) -> float:
        """Volume of expression to add to each well in μL."""
        return self.well_volume / (self.final_df / self.expression_df)

    @property
    def dilution_volume(self) -> float:
        """Volume of dilution buffer to add to each well in μL."""
        return self.well_volume - self.expression_volume

    @field_validator("final_df", "expression_df", "well_volume")
    def validate_dilution_volume(cls, v, values):
        """Validate that dilution volume is positive."""
        if (
            "final_df" in values
            and "expression_df" in values
            and "well_volume" in values
        ):
            dilution_volume = values["well_volume"] - (
                values["well_volume"] / (values["final_df"] / values["expression_df"])
            )
            if dilution_volume <= 0:
                raise ValueError("Dilution volume must be greater than 0")
        return v

    class Config:
        title = "Loading Plate Configuration"


class LoadingPlateProtocolParams(BaseModel):
    """Parameters for the loading plate preparation protocol."""

    plates: list[LoadingPlateParams] = Field(
        default_factory=lambda: [LoadingPlateParams()],
        description="List of loading plate configurations",
        title="Loading Plates",
    )

    class Config:
        title = "Loading Plate Protocol Parameters"


class SamplePlateParams(BaseModel):
    """Parameters for configuring a sample plate."""

    concentrations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of concentrations",
        title="Concentrations",
    )
    dilution_factor: float = Field(
        default=round(math.sqrt(10), 2),
        ge=1.0,
        le=100.0,
        description="Final dilution factor of expression",
        title="Dilution Factor",
    )
    dilution_buffer: str = Field(
        default="K",
        description="Dilution buffer (K or L)",
        pattern="^[KL]$",
        title="Dilution Buffer",
    )
    sample_volume: float = Field(
        default=40.0,
        ge=40.0,
        le=90.0,
        description="Volume of each sample well in μL",
        title="Sample Volume (μL)",
    )
    buffer_volume: float = Field(
        default=80.0,
        ge=40.0,
        le=90.0,
        description="Volume of each buffer well in μL",
        title="Buffer Volume (μL)",
    )
    rows: int = Field(
        default=8, ge=1, le=8, description="Number of sample rows", title="Rows"
    )
    columns: int = Field(
        default=12, ge=1, le=12, description="Number of sample columns", title="Columns"
    )

    @property
    def transfer_volume(self) -> float:
        """Volume of sample to transfer to each well in μL."""
        return self.sample_volume / (self.dilution_factor - 1)

    @property
    def inital_conc_volume(self) -> float:
        """Volume of initial concentration sample in μL."""
        return self.sample_volume / (self.dilution_factor - 1)

    class Config:
        title = "Sample Plate Configuration"


class SamplePlateProtocolParams(BaseModel):
    """Parameters for the sample plate preparation protocol."""

    plates: list[SamplePlateParams] = Field(
        default_factory=lambda: [SamplePlateParams()],
        description="List of sample plate configurations",
        title="Sample Plates",
    )

    class Config:
        title = "Sample Plate Protocol Parameters"


class BLIPlatePrepParams(BaseModel):
    """Parameters for the BLI plate preparation protocol.

    This protocol chains together the preparation of loading plates,
    and sample plates for Bio-Layer Interferometry (BLI) experiments.
    """

    loading_plates: list[LoadingPlateParams] = Field(
        default_factory=lambda: [LoadingPlateParams()],
        description="List of loading plate configurations. These plates contain the protein samples to be loaded onto the biosensors.",
        title="Loading Plates",
    )
    sample_plates: list[SamplePlateParams] = Field(
        default_factory=lambda: [SamplePlateParams()],
        description="List of sample plate configurations. These plates contain the analyte samples at different concentrations.",
        title="Sample Plates",
    )

    class Config:
        title = "BLI Plate Preparation Parameters"


def max_plate_protocol(
    params: MaxPlateProtocolParams,
    simulate: bool | None = None,
    protocol: Protocol | None = None,
) -> Protocol:
    """Prepare MAX plates for BLI experiment.

    This protocol sets up plates containing buffer and regeneration solution for
    the biosensor tips.

    Parameters
    ----------
    params : MaxPlateProtocolParams
        Protocol parameters
    simulate : bool | None, optional
        Whether to run in simulation mode, by default None
    protocol : Protocol | None, optional
        Existing protocol to add steps to, by default None

    Returns
    -------
    Protocol
        Configured protocol
    """
    plates = params.plates

    if not protocol:
        protocol = Protocol.from_layout(
            name="MAX Plate Protocol",
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

    protocol.initialize()
    protocol.pickup_tips(buffer_tips).eject_tips(holder_tips)

    for i in range(len(plates)):
        probe_columns = plates[i].columns
        probe_rows = plates[i].rows
        well_volume = plates[i].well_volume
        buffer_dispense_cycles = math.ceil(well_volume / buffer_tips.tip.max_volume)
        buffer_volume = well_volume / buffer_dispense_cycles
        regen_aspirate_volume = get_volume_per_channel(
            wells=probe_rows,
            num_channels=2,
            max_volume=regen_tips.tip.max_volume,
            volume_per_well=well_volume,
        )

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
            protocol.aspirate(
                regeneration_buffer[:4:2, -1], volume=regen_aspirate_volume
            )
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


def loading_plate_protocol(
    params: LoadingPlateProtocolParams,
    simulate: bool = False,
    protocol: Protocol | None = None,
) -> Protocol:
    """Prepare loading plates for BLI experiment.

    This protocol sets up plates containing protein samples to be loaded onto the biosensors.

    Parameters
    ----------
    params : LoadingPlateProtocolParams
        Protocol parameters
    simulate : bool, optional
        Whether to run in simulation mode, by default False
    protocol : Protocol | None, optional
        Existing protocol to add steps to, by default None

    Returns
    -------
    Protocol
        Configured protocol
    """
    plates = params.plates

    if not protocol:
        protocol = Protocol.from_layout(
            name="Loading Plate Protocol",
            layout_file=LAYOUTS_PATH / Path("bli.lay"),
            simulator_mode=simulate,
        )
    if not protocol.deck:
        msg = "Deck layout not loaded. Check that layouts/bli.lay exists."
        raise ValueError(msg)

    loading_plate_src = [
        protocol.deck.get_plate_stack("F1"),
        protocol.deck.get_plate_stack("F2"),
    ]
    loading_plate_dst = [
        protocol.deck.get_plate_stack("F1"),
        protocol.deck.get_plate_stack("F5"),
    ]
    loading_plate = protocol.deck.get_plate("E5")
    expression_plate_src = protocol.deck.get_plate_stack("F3")
    expression_plate = protocol.deck.get_plate("C3")
    k_buffer = protocol.deck.get_reservoir("B3")
    l_buffer = protocol.deck.get_reservoir("B4")
    holder_tips = protocol.deck.get_tip_rack("A4")
    buffer_tips = protocol.deck.get_tip_rack("A3")
    sample_tips_src = protocol.deck.get_tip_rack("E2")
    sample_tips = protocol.deck.get_tip_rack("D2")

    if not all(tip is not None for tip in [buffer_tips.tip, holder_tips.tip]):
        msg = "Tips not loaded. Check that A3 and A4 are tip racks."
        raise ValueError(msg)

    protocol.initialize()
    protocol.pickup_tips(buffer_tips).eject_tips(holder_tips)

    for i in range(len(plates)):
        columns = plates[i].columns
        rows = plates[i].rows
        src_well = alpha_to_index(plates[i].source_well)
        dst_well = alpha_to_index(plates[i].destination_well)
        replicates = plates[i].replicates
        well_vol = plates[i].well_volume
        dil_volume = plates[i].dilution_volume
        exp_vol = plates[i].expression_volume

        loading_src_stack = 0 if i < len(loading_plate_src) else 1
        loading_dst_stack = 0 if i < len(loading_plate_dst) else 1

        match plates[i].dilution_buffer:
            case "K":
                buffer = k_buffer
            case "L":
                buffer = l_buffer

        dilution_dispense_cycles = math.ceil(dil_volume / buffer_tips.tip.max_volume)
        dil_well_vol = dil_volume / dilution_dispense_cycles
        buffer_dispense_cycles = math.ceil(well_vol / buffer_tips.tip.max_volume)
        buffer_well_volume = well_vol / buffer_dispense_cycles

        protocol.grip_get(loading_plate_src[loading_src_stack][::-1][i])
        protocol.grip_place(loading_plate)

        protocol.pickup_tips(holder_tips[rows:columns])
        for q in [(0, 1), (1, 0)]:
            for _ in range(dilution_dispense_cycles):
                protocol.aspirate(buffer, volume=dil_well_vol).dispense(
                    loading_plate[q[0] : q[1]], volume=dil_well_vol
                )
        protocol.eject_tips(mode=1)

        protocol.pickup_tips(buffer_tips)
        for _ in range(buffer_dispense_cycles):
            protocol.aspirate(buffer, volume=buffer_well_volume).dispense(
                loading_plate, volume=buffer_well_volume
            )

        protocol.grip_get(sample_tips_src[::-1][i])
        protocol.grip_place(sample_tips)

        protocol.grip_get(expression_plate_src[::-1][i])
        protocol.grip_place(expression_plate)

        protocol.pickup_tips(sample_tips)
        for i in range(replicates):
            protocol.aspirate(
                expression_plate[src_well[0], src_well[1]], volume=exp_vol
            ).dispense(
                loading_plate[dst_well[0], dst_well[1] + i * columns],
                volume=exp_vol,
            )
        protocol.eject_tips(mode=2)

        protocol.grip_get(sample_tips)
        protocol.grip_place(sample_tips, waste=True)

        protocol.pickup_tips(holder_tips).eject_tips(buffer_tips)
        protocol.grip_get(loading_plate)
        protocol.grip_place(loading_plate_dst[loading_dst_stack][i])

    return protocol


def sample_plate_protocol(
    params: SamplePlateProtocolParams,
    simulate: bool = False,
    protocol: Protocol | None = None,
) -> Protocol:
    """Prepare sample plates for BLI experiment.

    This protocol sets up plates containing analyte samples at different concentrations.

    Parameters
    ----------
    params : SamplePlateProtocolParams
        Protocol parameters
    simulate : bool, optional
        Whether to run in simulation mode, by default False
    protocol : Protocol | None, optional
        Existing protocol to add steps to, by default None

    Returns
    -------
    Protocol
        Configured protocol
    """
    # Placeholder implementation - to be completed
    if protocol is None:
        protocol = Protocol.from_layout(
            name="Sample Plate Protocol",
            layout_file=LAYOUTS_PATH / Path("bli.lay"),
            simulator_mode=simulate,
        )

    # Mark as initialized
    if not protocol.commands:
        protocol.initialize()

    return protocol


def bli_plate_prep_protocol(
    params: BLIPlatePrepParams,
    simulate: bool = False,
    protocol: Protocol | None = None,
) -> Protocol:
    """Prepare all plates for a BLI experiment in one protocol.

    This protocol chains together the preparation of MAX plates, loading plates,
    and sample plates for Bio-Layer Interferometry (BLI) experiments.

    Parameters
    ----------
    params : BLIPlatePrepParams
        Protocol parameters including configurations for all plate types
    simulate : bool, optional
        Whether to run in simulation mode, by default False
    protocol : Protocol | None, optional
        Existing protocol to add steps to, by default None

    Returns
    -------
    Protocol
        Configured protocol
    """
    protocol = Protocol.from_layout(
        name="BLI Plate Prep Protocol",
        layout_file=LAYOUTS_PATH / Path("bli.lay"),
        simulator_mode=simulate,
    )
    if not protocol.deck:
        msg = "Deck layout not loaded. Check that layouts/bli.lay exists."
        raise ValueError(msg)

    protocol = max_plate_protocol(
        params=MaxPlateProtocolParams(plates=params.max_plates),
        simulate=simulate,
        protocol=protocol,
    )
    protocol = loading_plate_protocol(
        params=LoadingPlateProtocolParams(plates=params.loading_plates),
        simulate=simulate,
        protocol=protocol,
    )
    protocol = sample_plate_protocol(
        params=SamplePlateProtocolParams(plates=params.sample_plates),
        simulate=simulate,
        protocol=protocol,
    )

    return protocol
