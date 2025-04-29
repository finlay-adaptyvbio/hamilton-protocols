from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ..registry import registry
from pydantic import BaseModel

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


@app.post("/protocols/{protocol_id}/run")
async def run_protocol(protocol_id: str, request: RunProtocolRequest):
    """Run a protocol with given parameters."""
    # Convert ID back to protocol name
    protocol_name = protocol_id.replace("-", " ").title()

    info = registry.get_protocol(protocol_name)
    if not info:
        raise HTTPException(status_code=404, detail="Protocol not found")

    try:
        # Create parameter model instance
        params_instance = info.params_model(**request.params)

        # Create protocol instance
        protocol = info.func(params=params_instance, simulate=request.simulate)

        # Run protocol
        results = await protocol.run()

        # Format results for JSON response
        response = {
            "status": "success",
            "command_count": len(protocol.commands),
            "results": [
                {
                    "status": r.status,
                    "errors": [str(e) for e in r.errors],
                    "data": r.data,
                }
                for r in results
            ],
        }

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
