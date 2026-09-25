"""Discover the same typed configuration and plugin interfaces used to execute."""

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from bindocracy.campaign.models import Site
from bindocracy.cli_output import emit, fail
from bindocracy.config.load import load_yaml, read_tool_name
from bindocracy.config.models import GeneralConfig
from bindocracy.filters.config import FilterConfig
from bindocracy.functions.run import FunctionRunConfig
from bindocracy.ranking.models import RankingPolicy
from bindocracy.runs.index import WorkflowIndex
from bindocracy.store.query import DesignQuery
from bindocracy.tools import load_configs, plugin_for, registered_tools


class SchemaKind(str, Enum):
    GENERAL = "general"
    SITE = "site"
    INDEX = "index"
    FILTER = "filter"
    RANK = "rank"
    QUERY = "query"
    FUNCTION = "function"


_SCHEMAS = {
    SchemaKind.GENERAL: GeneralConfig,
    SchemaKind.SITE: Site,
    SchemaKind.INDEX: WorkflowIndex,
    SchemaKind.FILTER: FilterConfig,
    SchemaKind.RANK: RankingPolicy,
    SchemaKind.QUERY: DesignQuery,
    SchemaKind.FUNCTION: FunctionRunConfig,
}


def register(app: typer.Typer, config_app: typer.Typer) -> None:
    tools = typer.Typer(help="Discover registered tools, without claiming they are installed.")
    app.add_typer(tools, name="tools")
    tools.command("list")(list_tools)
    tools.command("describe")(describe_tool)
    config_app.command("schema")(config_schema)
    config_app.command("check")(check_config)


def list_tools() -> None:
    """List registered names; use campaign check to determine launchability."""
    names = registered_tools()
    emit({"tools": names}, text="\n".join(names))


def describe_tool(tool: str) -> None:
    """Describe a registered tool's actual configuration, not a copied catalogue."""
    try:
        plugin = plugin_for(tool)
    except KeyError as error:
        fail(error, code="unknown_tool")
    emit(
        {
            "tool": tool,
            "availability": "registered",
            "config_schema": plugin.config_type.model_json_schema(),
            "launchability": "requires_preflight",
        }
    )


def config_schema(
    tool: Annotated[
        str | None, typer.Option("--tool", help="Registered tool configuration.")
    ] = None,
    kind: Annotated[
        SchemaKind | None, typer.Option("--kind", help="Shared configuration type.")
    ] = None,
) -> None:
    """Export JSON Schema directly from the models used by validation."""
    if tool is not None and kind is not None:
        fail("Choose --tool or --kind, not both", code="invalid_arguments")
    try:
        model = (
            plugin_for(tool).config_type
            if tool is not None
            else _SCHEMAS[kind or SchemaKind.GENERAL]
        )
    except KeyError as error:
        fail(error, code="unknown_tool")
    emit(model.model_json_schema())


def check_config(
    general: Annotated[Path, typer.Option("--general", help="General campaign YAML.")],
    model: Annotated[Path, typer.Option("--model", help="Tool configuration YAML.")],
    preflight: Annotated[
        bool,
        typer.Option(
            "--preflight/--no-preflight",
            help="Check declared files and target compatibility, not just schema.",
        ),
    ] = True,
) -> None:
    """Validate a configuration pair without creating plans or writing database rows."""
    try:
        if preflight:
            loaded = load_configs(general, model)
            record = loaded.to_record()
            result = {
                "valid": True,
                "preflight": True,
                "tool": loaded.tool,
                "general_config_id": record.general_config_id,
                "model_config_id": record.model_config_id,
            }
        else:
            load_yaml(general, GeneralConfig)
            tool = read_tool_name(model)
            load_yaml(model, plugin_for(tool).config_type)
            # Preflight/resolve can change the stored config, so schema-only
            # checks must not advertise an execution/config identity.
            result = {"valid": True, "preflight": False, "tool": tool}
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        fail(error)
    emit(
        result,
        text=f"Valid {result['tool']} configuration ({'preflight' if preflight else 'schema only'})",
    )
