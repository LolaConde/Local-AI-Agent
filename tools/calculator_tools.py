import ast
import math
import operator

from llama_index.core.tools import FunctionTool

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def calculator(expression: str) -> str:
    """Evaluate basic arithmetic without executing arbitrary Python code."""
    if len(expression) > 10000:
        raise ValueError("Expression is too long.")

    tree = ast.parse(expression, mode="eval")

    if sum(1 for _ in ast.walk(tree)) > 2000:
        raise ValueError("Expression is too complex.")

    def evaluate(node):
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ValueError("Only numeric constants are allowed.")
            value = node.value

        elif (
            isinstance(node, ast.BinOp)
            and type(node.op) in _BINARY_OPS
        ):
            value = _BINARY_OPS[type(node.op)](
                evaluate(node.left),
                evaluate(node.right),
            )

        elif (
            isinstance(node, ast.UnaryOp)
            and type(node.op) in _UNARY_OPS
        ):
            value = _UNARY_OPS[type(node.op)](
                evaluate(node.operand)
            )

        else:
            raise ValueError("Unsupported arithmetic expression.")

        if abs(value) > 1e100:
            raise ValueError("Result exceeds the supported range.")

        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Result must be finite.")

        return value

    return str(evaluate(tree.body))

calculator_tool = FunctionTool.from_defaults(
    fn=calculator,
    name="calculator",
    description=(
        "Required for ALL arithmetic calculations, including simple "
        "addition, subtraction, multiplication, division, percentages, "
        "averages, and totals. "
        "Input: an unevaluated arithmetic expression, such as "
        "'17 + 25' or '(120 * 15) / 100'. "
        "Output: the calculated result as a string. "
        "Always call this tool instead of calculating mentally. "
        "Pass the original expression, not an answer you computed yourself."
    ),
)