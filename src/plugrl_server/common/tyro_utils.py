from __future__ import annotations

from typing import Any, Sequence, TypeVar, Union

from typing_extensions import Annotated

from tyro.conf._markers import Suppress
from tyro.constructors import ConstructorRegistry
from typing import Callable


T = TypeVar("T")

NestedCallableDict = dict[str, Callable[..., Any] | "NestedCallableDict"]


def subcommand_cli_from_nested_dict(
    subcommands: NestedCallableDict,
    *,
    prog: str | None = None,
    description: str | None = None,
    args: Sequence[str] | None = None,
    use_underscores: bool = False,
    console_outputs: bool = True,
    add_help: bool = True,
    config: Sequence[Any] | None = None,
    sort_subcommands: bool = False,
    registry: ConstructorRegistry | None = None,
) -> Any:
    import tyro

    # Called to specify the subcommand corresponding to each value in the
    # dictionary.
    def _make_constructor(v):
        return (
            # Call the function directly.
            v
            if isinstance(v, Callable)
            # If a dictionary: construct a struct type with one
            # field, which is a union representing the next level of
            # subcommands.
            else tuple[
                Annotated[
                    _make_recursive_union(v),
                    tyro.conf.arg(name=""),
                ]
            ]
        )

    def _make_recursive_union(subcommands: NestedCallableDict) -> type:
        return Union[  # type: ignore
            tuple(
                [
                    Annotated[
                        # The constructor function can return any object.
                        Any,
                        # We'll instantiate this object by invoking a subcommand with
                        # a custom constructor.
                        tyro.conf.subcommand(
                            name=k,
                            constructor=_make_constructor(v),
                        ),
                    ]
                    for k, v in subcommands.items()
                ]
                # Union types need at least two types. To support the case
                # where we only pass one subcommand in, we'll pad with `None`
                # but suppress it.
                + [Annotated[None, Suppress]]
            )
        ]

    # We need to form a union type, which requires at least two elements.
    return tyro.cli(
        _make_recursive_union(subcommands),
        prog=prog,
        description=description,
        args=args,
        use_underscores=use_underscores,
        console_outputs=console_outputs,
        add_help=add_help,
        # Don't create subcommands for union types within the config object.
        config=(tyro.conf.AvoidSubcommands,)
        + (tuple() if config is None else tuple(config)),
        registry=registry,
    )
