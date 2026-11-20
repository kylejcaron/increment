"""Arithmetic syntax shared by metric consumers; no evaluation."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from math import isfinite
from typing import Literal as Token

from pydantic_core import PydanticCustomError

type Operator = Token["+", "-", "*", "/"]
_OPERATORS: dict[type[ast.operator], Operator] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}


@dataclass(frozen=True, slots=True)
class Literal:
    value: int | float


@dataclass(frozen=True, slots=True)
class Reference:
    name: str


@dataclass(frozen=True, slots=True)
class Negation:
    operand: Formula


@dataclass(frozen=True, slots=True)
class Binary:
    operator: Operator
    left: Formula
    right: Formula


type Formula = Literal | Reference | Negation | Binary


def _from_ast(node: ast.expr) -> Formula:
    if isinstance(node, ast.Name):
        return Reference(node.id)
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, int) and not isinstance(value, bool):
            return Literal(value)
        if isinstance(value, float) and isfinite(value):
            return Literal(value)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return Negation(_from_ast(node.operand))
    elif isinstance(node, ast.BinOp) and (operator := _OPERATORS.get(type(node.op))) is not None:
        return Binary(operator, _from_ast(node.left), _from_ast(node.right))
    raise ValueError("Use metric or measure names, finite numbers, parentheses, and + - * /.")


def parse_formula(value: object) -> Formula:
    if not isinstance(value, str) or not value.isascii():
        raise PydanticCustomError("invalid_formula", "Write the formula as an ASCII string.")
    try:
        return _from_ast(ast.parse(value.strip(), mode="eval").body)
    except (ValueError, SyntaxError, RecursionError) as exc:
        raise PydanticCustomError(
            "invalid_formula", "Invalid arithmetic formula: {detail}", {"detail": str(exc)}
        ) from exc


def references(formula: Formula) -> Iterator[str]:
    """Yield referenced names in left-to-right order."""
    pending = [formula]
    while pending:
        node = pending.pop()
        match node:
            case Reference(name):
                yield name
            case Negation(operand):
                pending.append(operand)
            case Binary(_, left, right):
                pending.extend((right, left))
