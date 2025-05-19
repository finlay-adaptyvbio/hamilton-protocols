from traceback import format_exc
from typing import Any
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from ..registry import registry
from .log_analyzer import (
    analyze_log_file,
    get_analysis,
    analyze_all_logs,
    get_combined_analysis,
)

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
    """Run a protocol with given parameters and stream results in real-time."""
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
