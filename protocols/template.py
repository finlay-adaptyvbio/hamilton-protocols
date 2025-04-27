"""
Example protocol template with enhanced metadata and documentation.

This template demonstrates how to create a Hamilton protocol with proper
parameter validation, documentation, and tagging.

@tag: example
@tag: template
"""

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from adaptyv_lab import Protocol


class ExampleProtocolParams(BaseModel):
    """Parameters for the example protocol.

    This model defines the parameters for the example protocol with
    proper validation, defaults, and descriptions.
    """

    # Required parameters with validation
    source_plate: str = Field(
        ...,  # ... means required (no default)
        description="Name or position of the source plate",
    )
    dest_plate: str = Field(
        ..., description="Name or position of the destination plate"
    )

    # Required numeric parameters with validation
    volume: float = Field(
        50.0, gt=0.0, le=1000.0, description="Volume to transfer in microliters"
    )

    # Optional parameters with defaults
    mix_cycles: int = Field(0, ge=0, le=10, description="Number of mixing cycles")

    # Optional list parameters
    positions: Optional[List[str]] = Field(
        None, description="Specific positions to transfer from (default: all positions)"
    )

    # Boolean parameters
    use_liquid_following: bool = Field(
        False, description="Whether to use liquid following"
    )

    # Advanced parameters (might be hidden in UI by default)
    submerge_depth: float = Field(
        2.0,
        description="Depth to submerge tips in mm",
    )

    class Config:
        """Pydantic configuration."""

        title = "Example Protocol Parameters"


def example_protocol(params: ExampleProtocolParams, simulate: bool = False) -> Protocol:
    """Execute a simple transfer protocol.

    This is an example protocol that demonstrates proper parameter handling,
    documentation, and protocol structure. It performs a simple transfer
    from a source plate to a destination plate.

    Parameters
    ----------
    params : ExampleProtocolParams
        Protocol parameters validated by Pydantic
    simulate : bool, optional
        Whether to run in simulation mode, by default False

    Returns
    -------
    Protocol
        The configured protocol ready for execution

    @tag: transfer
    """
    # Create protocol object with layout
    protocol = Protocol.from_layout(
        name="Example Protocol",
        layout_file=Path("layouts/simple.lay"),
        simulator_mode=simulate,
    )

    # Validate that deck is loaded
    if not protocol.deck:
        msg = "Deck layout not loaded. Check that layouts/simple.lay exists."
        raise ValueError(msg)

    # Get labware references
    try:
        source_plate = protocol.deck.get_plate(params.source_plate)
        dest_plate = protocol.deck.get_plate(params.dest_plate)
        tip_rack = protocol.deck.get_tip_rack("A1")  # Example - adjust as needed
    except ValueError as e:
        raise ValueError(f"Error finding labware: {e}")

    # Prepare positions
    if params.positions:
        # Use specified positions
        pass
    else:
        # Use all positions by default
        [pos.alphanumeric for pos in source_plate.positions]

    # Build protocol steps
    protocol.initialize()

    # Pick up tips
    protocol.pickup_tips(tip_rack)

    # Configure aspirate parameters
    aspirate_params = {
        "mix_cycles": params.mix_cycles,
        "liquid_following": int(params.use_liquid_following),
        "submerge_depth": params.submerge_depth,
    }

    # Execute transfer
    protocol.aspirate(labware=source_plate, volume=params.volume, **aspirate_params)

    protocol.dispense(
        labware=dest_plate,
        volume=params.volume,
        mix_cycles=params.mix_cycles,
    )

    # Eject tips
    protocol.eject_tips(mode=1)  # Eject to default waste

    return protocol


# This pattern enables the protocol to be run directly for testing
if __name__ == "__main__":
    # Example parameters for testing
    test_params = ExampleProtocolParams(
        source_plate="A1",
        dest_plate="B1",
        volume=100.0,
        mix_cycles=3,
    )

    # Create protocol
    protocol = example_protocol(test_params, simulate=True)

    # Print protocol commands
    for i, cmd in enumerate(protocol.commands):
        print(f"[{i + 1}] {cmd.__class__.__name__}: {cmd.command}")

    # Optionally run the protocol
    # asyncio.run(protocol.run())
