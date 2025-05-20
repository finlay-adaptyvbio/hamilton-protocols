import base64


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
    wells_per_channel_max = int(max_volume / volume_per_well)
    if wells > wells_per_channel_max * num_channels:
        raise ValueError(
            f"Cannot distribute {wells} wells across {num_channels} channels with max {wells_per_channel_max} wells per channel."
        )

    volumes = [0.0] * num_channels
    wells_per_channel_min = wells // num_channels
    remaining_wells = wells % num_channels

    for i in range(num_channels):
        volumes[i] = round(wells_per_channel_min * volume_per_well, 2)
    for i in range(remaining_wells):
        volumes[i] += volume_per_well

    return [round(vol / 10) * 10 for vol in volumes]


def dataframe_to_base64(df):
    # Convert DataFrame to CSV string
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()

    # Encode as base64
    csv_bytes = csv_string.encode("utf-8")
    base64_bytes = base64.b64encode(csv_bytes)
    base64_string = base64_bytes.decode("utf-8")

    return base64_string
