import base64
import io
import json
import os
from collections.abc import Callable
from traceback import format_exc
from typing import Any, Generic, TypeVar

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..registry import registry
from .log_analyzer import (
    analyze_all_logs,
    analyze_log_file,
    get_analysis,
    get_combined_analysis,
)

# Define generic type for CSV data
T = TypeVar("T")


class CSVData(Generic[T]):
    """Container for parsed CSV data with validation and transformation capabilities using pandas."""

    def __init__(
        self,
        base64_data: str,
        transform_func: Callable[[pd.DataFrame], Any] | None = None,
        dtype: dict | None = None,
    ):
        self.base64_data = base64_data
        self.df = self._decode_and_parse(dtype)
        self.data = transform_func(self.df) if transform_func else self.df

    def _decode_and_parse(self, dtype: dict | None = None) -> pd.DataFrame:
        """Decode base64 string and parse as CSV using pandas."""
        try:
            # Decode base64 string
            csv_bytes = base64.b64decode(self.base64_data)

            # Parse CSV data with pandas
            df = pd.read_csv(io.BytesIO(csv_bytes), dtype=dtype)
            return df
        except Exception as e:
            raise ValueError(f"Failed to parse CSV data: {e!s}")

    def to_dict(self, orient="records"):
        """Convert DataFrame to dictionary."""
        return self.df.to_dict(orient=orient)

    def __iter__(self):
        # For backwards compatibility with dict-based approach
        return iter(self.to_dict())

    def __getitem__(self, key):
        # Support both integer indexing and column access
        if isinstance(key, int):
            if 0 <= key < len(self.df):
                return self.df.iloc[key].to_dict()
            raise IndexError(
                f"Index {key} out of bounds for CSV data with {len(self.df)} rows"
            )
        return self.df[key]

    def __len__(self):
        return len(self.df)


def validate_csv_data(v: str) -> str:
    """
    Pydantic V2 field_validator for CSV data fields.

    Usage in Pydantic models:
    ```
    from pydantic import field_validator

    class MyModel(BaseModel):
        csv_data: str

        # Validate that csv_data is valid base64-encoded CSV
        @field_validator('csv_data')
        @classmethod
        def validate_csv(cls, v):
            return validate_csv_data(v)

        # Optional: validate required columns
        @field_validator('csv_data')
        @classmethod
        def validate_columns(cls, v):
            data = CSVData(v)
            if len(data.df) == 0:
                raise ValueError("CSV data contains no rows")

            required_columns = ['column1', 'column2']
            missing = [col for col in required_columns if col not in data.df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {', '.join(missing)}")
            return v
    ```
    """
    try:
        # Just try to parse it to validate
        CSVData(v)
        return v
    except Exception as e:
        raise ValueError(f"Invalid CSV data: {e!s}")


app = FastAPI(title="hamilton-runner API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Discover protocols
registry.discover_protocols()


class RunProtocolRequest(BaseModel):
    params: dict[str, Any] = {}
    simulate: bool = False


@app.get("/protocols", response_model=list[dict[str, Any]])
async def get_protocols():
    """Get all available protocols."""
    protocols = []
    for name, info in registry.protocols.items():
        # Get schema from params model
        schema = info.params_model.model_json_schema()

        protocols.append(
            {
                "id": name.lower().replace(" ", "-"),
                "name": name,
                "description": info.description,
                "tags": list(info.tags),
                "params_schema": schema,
            }
        )
    return protocols


@app.get("/protocols/{protocol_id}", response_model=dict[str, Any])
async def get_protocol(protocol_id: str):
    """Get a specific protocol by ID."""
    # Convert ID back to protocol name
    protocol_name = protocol_id.replace("-", " ").title()

    info = registry.get_protocol(protocol_name)
    if not info:
        raise HTTPException(status_code=404, detail="Protocol not found")

    # Get schema from params model
    schema = info.params_model.model_json_schema()

    return {
        "id": protocol_id,
        "name": protocol_name,
        "description": info.description,
        "tags": list(info.tags),
        "params_schema": schema,
    }


@app.get("/logs/analyze/{log_file}")
async def analyze_log(log_file: str):
    """Analyze command execution times from a log file."""
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "hamilton-logs"
    )
    log_path = os.path.join(log_dir, log_file)

    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Log file not found")

    try:
        command_times = analyze_log_file(log_path)
        if not command_times:
            return {"error": "No command times found in log file"}

        analysis = get_analysis(command_times)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/analyze-all")
async def analyze_all_log_files():
    """Analyze all log files in the hamilton-logs directory."""
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "hamilton-logs"
    )

    if not os.path.exists(log_dir):
        raise HTTPException(status_code=404, detail="Log directory not found")

    try:
        all_analyses = analyze_all_logs(log_dir)
        if not all_analyses:
            return {
                "error": "No log files found or no command times found in any log file"
            }

        # Get individual file analyses
        file_analyses = {file: analysis for file, analysis in all_analyses.items()}

        # Get combined analysis across all files
        combined_analysis = get_combined_analysis(all_analyses)

        return {"individual_files": file_analyses, "combined": combined_analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/protocols/{protocol_id}/run")
async def run_protocol(protocol_id: str, request: RunProtocolRequest):
    """
    Run a protocol with given parameters and stream results in real-time.

    For protocols that require CSV file input:
    1. Convert your CSV file to a base64-encoded string on the frontend
    2. Pass the base64 string as the value for the CSV parameter in the request
    3. The protocol will decode and parse the CSV data automatically using pandas

    CSV data handling features:
    - Automatic type conversion (specify dtypes in get_mapping_data)
    - Pandas DataFrame operations for filtering, sorting, and analysis
    - Validation of required columns and data types with Pydantic V2 field_validators
    - Easy access to both row-based and column-based data
    - Support for multiple CSV files via nested models

    Multi-CSV approach:
    For protocols requiring multiple CSV files, you can:
    1. Include CSV fields in nested model structures
    2. Add a master CSV at the parent level for shared data
    3. Add plate-specific CSV files at the child model level
    4. Filter and join data across files using pandas operations

    Validation with Pydantic V2:
    CSV data is validated with Pydantic's field_validator decorators:
    1. Format validation - checks that the base64 string is valid CSV data
    2. Schema validation - checks that required columns exist
    3. Type validation - checks that numeric columns contain valid numbers

    Example model with CSV validation:
    ```python
    from pydantic import BaseModel, Field, field_validator

    class MyParams(BaseModel):
        csv_data: str = Field(..., description="Base64-encoded CSV data")

        @field_validator('csv_data')
        @classmethod
        def validate_csv(cls, v):
            # Validate format
            try:
                CSVData(v)
            except Exception as e:
                raise ValueError(f"Invalid CSV data: {str(e)}")
            return v

        @field_validator('csv_data')
        @classmethod
        def validate_columns(cls, v):
            # Validate content
            data = CSVData(v)
            # Check required columns
            return v
    ```

    Example request with multiple CSV files:
    ```json
    {
      "params": {
        "master_csv": "base64_encoded_master_data",
        "plates": [
          {
            "plate_id": "PLATE-123",
            "well_mapping_csv": "base64_encoded_mapping_data",
            "concentration_csv": "base64_encoded_concentration_data"
          },
          {
            "plate_id": "PLATE-456",
            "well_mapping_csv": "base64_encoded_mapping_data_2"
          }
        ]
      },
      "simulate": false
    }
    ```

    Example JavaScript code to encode a CSV file:
    ```javascript
    // Read file from input element
    const fileInput = document.getElementById('csvFile');
    const file = fileInput.files[0];

    const reader = new FileReader();
    reader.onload = function(event) {
        // Get file content as base64
        const base64String = event.target.result.split(',')[1];

        // Add to your params object
        const params = {
            plate_id: "PLATE-123",
            csv_data: base64String
        };

        // Send the request
        fetch(`/protocols/${protocolId}/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ params, simulate: false })
        });
    };

    // Read file as data URL (base64)
    reader.readAsDataURL(file);
    ```
    """
    # Convert ID back to protocol name
    protocol_name = protocol_id.replace("-", " ").title()

    info = registry.get_protocol(protocol_name)
    if not info:
        raise HTTPException(status_code=404, detail="Protocol not found")

    async def stream_results():
        try:
            # Create parameter model instance
            params_instance = info.params_model(**request.params)

            # Create protocol instance
            protocol = info.func(params=params_instance, simulate=request.simulate)

            # Send initial metadata including deck layout
            deck_layout = protocol.deck.to_json() if protocol.deck else None
            yield (
                json.dumps(
                    {
                        "type": "metadata",
                        "command_count": len(protocol.commands),
                        "deck_layout": deck_layout,
                    }
                )
                + "\n"
            )

            # Stream results as they become available
            async for result in protocol.run():
                formatted_result = {
                    "type": "result",
                    "status": result.status,
                    "errors": [str(e) for e in result.errors],
                    "data": result.data,
                }
                yield json.dumps(formatted_result) + "\n"

            # Send completion message
            yield json.dumps({"type": "complete", "status": "success"}) + "\n"

        except Exception as e:
            print(format_exc())
            error_msg = {"type": "error", "detail": str(e)}
            yield json.dumps(error_msg) + "\n"

    return StreamingResponse(stream_results(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
