import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run the Hamilton Protocols API")
    parser.add_argument(
        "--dev", action="store_true", help="Run in development mode with hot reloading"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to run the API on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the API on")

    args = parser.parse_args()

    uvicorn.run(
        "hamilton_protocols.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.dev,
    )


if __name__ == "__main__":
    main()
