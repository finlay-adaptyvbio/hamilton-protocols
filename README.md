# Hamilton Protocols

A library for Hamilton liquid handling protocols with FastAPI-based REST API.

## Installation

Install dependencies using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Running the API

Start the API server:

```bash
# Production mode
uv run python -m hamilton_protocols.api

# Development mode with hot reloading
uv run python -m hamilton_protocols.api --dev

# Custom port
uv run python -m hamilton_protocols.api --port 8080
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## API Endpoints

- `GET /protocols` - List all available protocols
- `GET /protocols/{protocol_id}` - Get specific protocol details
- `POST /protocols/{protocol_id}/run` - Run a protocol with parameters
- `GET /logs/analyze/{log_file}` - Analyze log file execution times
- `GET /logs/analyze-all` - Analyze all log files

## Protocol Development

Create new protocols in `src/hamilton_protocols/protocols/`. Each protocol needs:

1. **Parameter Model** - Define protocol inputs using Pydantic:

```python
from pydantic import BaseModel, Field, field_validator
from hamilton_protocols.api.main import CSVData

class MyProtocolParams(BaseModel):
    csv_data: str = Field(..., description="Base64-encoded CSV data")

    @field_validator('csv_data')
    @classmethod
    def validate_csv(cls, v):
        data = CSVData(v)
        required_columns = ['Source Well', 'Destination Well']
        missing = [col for col in required_columns if col not in data.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")
        return v
```

2. **Protocol Function** - Implement the protocol logic:

```python
from adaptyv_lab import Protocol
from hamilton_protocols import LAYOUTS_PATH

def my_protocol(params: MyProtocolParams, simulate: bool = False) -> Protocol:
    protocol = Protocol.from_layout(
        name="My Protocol",
        layout_file=LAYOUTS_PATH / "my-layout.lay",
        simulator_mode=simulate,
    )

    # Protocol implementation using params.csv_data
    data = CSVData(params.csv_data)
    # ... protocol steps ...

    return protocol
```

See [cherry.py](/src/hamilton_protocols/protocols/cherry.py) for a complete example with CSV validation and well mapping.
