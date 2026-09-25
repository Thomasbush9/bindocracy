"""One versioned machine interface, shared by every CLI command.

Use ``bindocracy --json ...`` for a success document on stdout or an error
on stderr. Human output remains the default. Library functions never print.
"""

from __future__ import annotations

import json
import sys
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn

import typer
from pydantic import BaseModel, ValidationError
from typer._click.exceptions import ClickException
from typer.core import TyperGroup

_MACHINE = ContextVar("bindocracy_json_output", default=False)


def _default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (Path, date, datetime)):
        return str(value) if isinstance(value, Path) else value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} as a CLI result")


def _dump(value: Any) -> str:
    return json.dumps(value, default=_default, allow_nan=False, indent=2)


def emit(payload: Any, *, text: str | None = None) -> None:
    """Print one result; JSON mode never mixes result data with human prose."""
    if _MACHINE.get():
        typer.echo(_dump({"schema_version": 1, "ok": True, "result": payload}))
    else:
        typer.echo(text if text is not None else _dump(payload))


def _error(error: Exception | str, code: str) -> None:
    if _MACHINE.get():
        details = (
            error.errors(include_url=False, include_context=False, include_input=False)
            if isinstance(error, ValidationError)
            else []
        )
        typer.echo(
            _dump(
                {
                    "schema_version": 1,
                    "ok": False,
                    "error": {"code": code, "message": str(error), "details": details},
                }
            ),
            err=True,
        )
    else:
        typer.echo(str(error), err=True)


def fail(error: Exception | str, *, code: str = "validation_error", exit_code: int = 2) -> NoReturn:
    """Report an expected refusal and preserve its nonzero process status."""
    _error(error, code)
    raise typer.Exit(code=exit_code)


class OutputGroup(TyperGroup):
    """Also render parser failures and unexpected exceptions in machine mode.

    The flag is a root option, before the command. Establishing the context here
    (before Click parses arguments) covers missing/invalid arguments as well as
    successful callbacks. No process-global output redirection is involved.
    """

    def main(self, args=None, *, standalone_mode=True, **kwargs):
        arguments = list(sys.argv[1:] if args is None else args)
        machine = "--json" in arguments[: arguments.index("--") if "--" in arguments else None]
        if not machine:
            return super().main(args=arguments, standalone_mode=standalone_mode, **kwargs)
        token = _MACHINE.set(True)
        result = None
        try:
            if "--help" in arguments:
                raise typer.BadParameter(
                    "Use --help without --json; tools list and config schema provide JSON discovery"
                )
            result = super().main(args=arguments, standalone_mode=False, **kwargs)
            status = result if isinstance(result, int) else 0
        except ClickException as error:
            _error(error, "invalid_arguments")
            status = error.exit_code
        except typer.Abort:
            _error("Interrupted", "interrupted")
            status = 130
        except Exception as error:  # noqa: BLE001 - encode unexpected failures at the CLI boundary
            _error(error, "internal_error")
            status = 1
        finally:
            _MACHINE.reset(token)
        if standalone_mode:
            raise SystemExit(status)
        return status if status else result
