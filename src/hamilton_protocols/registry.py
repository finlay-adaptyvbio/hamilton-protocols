import importlib
import inspect
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from adaptyv_lab.logging import get_logger
from pydantic import BaseModel

from hamilton_protocols import PROTOCOLS_PATH


class ProtocolInfo:
    """Information about a protocol function."""

    def __init__(
        self,
        name: str,
        func: Callable,
        params_model: type[BaseModel],
        module_path: str,
        description: str = "",
        tags: set[str] = None,
        last_modified: float = None,
    ):
        self.name = name
        self.func = func
        self.params_model = params_model
        self.module_path = module_path
        self.description = description or ""
        self.tags = tags or set()
        self.last_modified = last_modified or time.time()

    def __repr__(self) -> str:
        return f"ProtocolInfo(name={self.name}, module={self.module_path}, tags={self.tags}), params_model={self.params_model}"


class ProtocolRegistry:
    """Registry of available protocols."""

    def __init__(self):
        self.protocols: dict[str, ProtocolInfo] = {}
        self.logger = get_logger(__name__)
        self._discovery_time = 0

    def discover_protocols(self, protocols_path: Path = PROTOCOLS_PATH) -> None:
        """Discover protocols in the given directory.

        Parameters
        ----------
        protocols_path : Path
            The directory to discover protocols in

        Raises
        ------
        ValueError
            If the protocols directory does not exist
        """
        self.logger.info(f"Discovering protocols in {protocols_path}")
        start_time = time.time()

        if not protocols_path.exists():
            self.logger.error(f"Protocols directory {protocols_path} does not exist")
            raise ValueError(f"Protocols directory {protocols_path} does not exist")

        # Add the parent directory to sys.path to make protocols importable
        parent_dir = str(protocols_path.parent.absolute())
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        # Clear existing protocols to prevent duplicates on rediscovery
        self.protocols = {}
        protocols_found = 0
        errors = 0

        for file_path in protocols_path.glob("**/*.py"):
            if file_path.name.startswith("_"):
                continue  # Skip private modules

            # Get file modification time for cache invalidation
            mod_time = file_path.stat().st_mtime

            # Convert the file path to a module path
            rel_path = file_path.relative_to(protocols_path.parent)
            module_parts = str(rel_path).replace(".py", "").replace(os.sep, ".")
            module_path = module_parts

            try:
                # Import the module
                self.logger.debug(f"Importing module {module_path}")
                module = importlib.import_module(module_path)

                # Extract tags from module docstring if available
                module_tags = set()
                if module.__doc__:
                    # Look for tags in the format "@tag: tag_name"
                    for line in module.__doc__.splitlines():
                        if line.strip().startswith("@tag:"):
                            tag = line.strip().split("@tag:", 1)[1].strip()
                            module_tags.add(tag)

                # Find protocol functions and their parameter models
                for name, obj in inspect.getmembers(module):
                    if not inspect.isfunction(obj):
                        continue

                    # Look for functions that have a parameters class as a parameter
                    params = self._find_params_model(obj, module)
                    if params:
                        protocol_name = self._format_protocol_name(name)

                        # Extract function-specific tags
                        func_tags = set(module_tags)  # Start with module tags
                        if obj.__doc__:
                            for line in obj.__doc__.splitlines():
                                if line.strip().startswith("@tag:"):
                                    tag = line.strip().split("@tag:", 1)[1].strip()
                                    func_tags.add(tag)

                        self.protocols[protocol_name] = ProtocolInfo(
                            name=protocol_name,
                            func=obj,
                            params_model=params,
                            module_path=module_path,
                            description=obj.__doc__ or "",
                            tags=func_tags,
                            last_modified=mod_time,
                        )
                        protocols_found += 1
                        self.logger.debug(f"Added protocol {protocol_name}")

            except Exception as e:
                self.logger.error(
                    f"Error loading protocol from {module_path}: {e}", exc_info=True
                )
                errors += 1

        self._discovery_time = time.time() - start_time
        self.logger.info(
            f"Discovered {protocols_found} protocols in {self._discovery_time:.2f}s with {errors} errors"
        )

    def _find_params_model(self, func: Callable, module: Any) -> type[BaseModel] | None:
        """Find the parameter model for a function."""
        self.logger.debug(f"Finding parameter model for {func.__name__}")
        signature = inspect.signature(func)

        # First, look for parameter annotations that are BaseModel subclasses
        for param_name, param in signature.parameters.items():
            if param.annotation == inspect.Parameter.empty:
                continue

            # Get the actual class
            param_class = param.annotation
            if isinstance(param_class, str):
                # Resolve forward references
                try:
                    param_class = eval(param_class, module.__dict__)
                except Exception as e:
                    self.logger.debug(
                        f"Error resolving forward reference {param_class}: {e}"
                    )
                    continue

            # Check if it's a subclass of BaseModel
            try:
                if issubclass(param_class, BaseModel):
                    return param_class
            except TypeError:
                continue  # Not a class

        # If no annotated parameter found, look for a matching params class in the module
        func_name = func.__name__
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseModel)
                and name.endswith("Params")
                and func_name.lower().replace("_", "")
                in name.lower().replace("params", "")
            ):
                return obj

        return None

    def _format_protocol_name(self, func_name: str) -> str:
        """Format a function name into a human-readable protocol name."""
        return " ".join(
            word.capitalize() for word in func_name.replace("_", " ").split()
        )

    def get_protocol(self, name: str) -> ProtocolInfo | None:
        """Get a protocol by name."""
        return self.protocols.get(name)

    def list_protocols(self) -> list[str]:
        """List all available protocols."""
        return list(self.protocols.keys())

    def get_protocols_by_tag(self, tag: str) -> list[ProtocolInfo]:
        """Get protocols that have a specific tag."""
        return [p for p in self.protocols.values() if tag in p.tags]

    def search_protocols(self, query: str) -> list[ProtocolInfo]:
        """Search for protocols by name or description."""
        query = query.lower()
        return [
            p
            for p in self.protocols.values()
            if query in p.name.lower() or query in p.description.lower()
        ]

    @property
    def tags(self) -> set[str]:
        """Get all tags across all protocols."""
        all_tags = set()
        for protocol in self.protocols.values():
            all_tags.update(protocol.tags)
        return all_tags

    @property
    def discovery_stats(self) -> dict[str, Any]:
        """Get statistics about the last discovery run."""
        return {
            "count": len(self.protocols),
            "discovery_time": self._discovery_time,
            "tags": list(self.tags),
        }


# Singleton registry instance
registry = ProtocolRegistry()
