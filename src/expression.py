"""
Expression tree representation for symbolic regression.

Author: Anand
Date: March 2, 2025
GSoC 2026 - DeepChem Symbolic ML
"""

import torch
import numpy as np
from typing import Optional, Union, List, Tuple
from enum import Enum


class NodeType(Enum):
    """Types of nodes in expression tree."""
    OPERATOR = "operator"
    VARIABLE = "variable"
    CONSTANT = "constant"


class Operator(Enum):
    """Supported mathematical operators."""
    # Binary operators
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    POW = "pow"
    
    # Unary operators
    SIN = "sin"
    COS = "cos"
    EXP = "exp"
    LOG = "log"
    SQRT = "sqrt"
    NEG = "neg"
    
    @property
    def arity(self) -> int:
        """Number of arguments operator takes."""
        unary = {self.SIN, self.COS, self.EXP, self.LOG, self.SQRT, self.NEG}
        return 1 if self in unary else 2
    
    @property
    def symbol(self) -> str:
        """Human-readable symbol."""
        symbols = {
            self.ADD: "+",
            self.SUB: "-",
            self.MUL: "*",
            self.DIV: "/",
            self.POW: "^",
            self.SIN: "sin",
            self.COS: "cos",
            self.EXP: "exp",
            self.LOG: "log",
            self.SQRT: "sqrt",
            self.NEG: "-",
        }
        return symbols[self]


class ExpressionNode:
    """
    Node in expression tree.
    
    Can represent:
    - Operator (e.g., +, *, sin)
    - Variable (e.g., x)
    - Constant (e.g., 3.14)
    """
    
    def __init__(
        self,
        node_type: NodeType,
        operator: Optional[Operator] = None,
        value: Optional[float] = None,
        left: Optional['ExpressionNode'] = None,
        right: Optional['ExpressionNode'] = None,
    ):
        self.node_type = node_type
        self.operator = operator
        self.value = value
        self.left = left
        self.right = right
        
        # Validation
        if node_type == NodeType.OPERATOR:
            assert operator is not None, "Operator node must have operator"
            if operator.arity == 2:
                assert left is not None and right is not None, \
                    f"Binary operator {operator} needs two children"
            elif operator.arity == 1:
                assert left is not None, \
                    f"Unary operator {operator} needs one child"
        
        elif node_type == NodeType.CONSTANT:
            assert value is not None, "Constant node must have value"
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf (variable or constant)."""
        return self.node_type in [NodeType.VARIABLE, NodeType.CONSTANT]
    
    def depth(self) -> int:
        """Calculate depth of subtree rooted at this node."""
        if self.is_leaf():
            return 1
        
        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0
        return 1 + max(left_depth, right_depth)
    
    def size(self) -> int:
        """Count total nodes in subtree."""
        if self.is_leaf():
            return 1
        
        left_size = self.left.size() if self.left else 0
        right_size = self.right.size() if self.right else 0
        return 1 + left_size + right_size
    
    def copy(self) -> 'ExpressionNode':
        """Deep copy of subtree."""
        if self.is_leaf():
            return ExpressionNode(
                node_type=self.node_type,
                operator=self.operator,
                value=self.value,
                left=None,
                right=None,
            )
        
        # For operator nodes, copy children first
        left_copy = self.left.copy() if self.left else None
        right_copy = self.right.copy() if self.right else None
        
        return ExpressionNode(
            node_type=self.node_type,
            operator=self.operator,
            value=self.value,
            left=left_copy,
            right=right_copy,
        )
    
    def __str__(self) -> str:
        """Human-readable expression string."""
        if self.node_type == NodeType.VARIABLE:
            return "x"
        
        if self.node_type == NodeType.CONSTANT:
            return f"{self.value:.3f}"
        
        # Operator node
        op = self.operator
        
        if op.arity == 1:
            return f"{op.symbol}({self.left})"
        
        # Binary operator
        left_str = str(self.left)
        right_str = str(self.right)
        
        # Add parentheses for clarity
        if op in [Operator.ADD, Operator.SUB]:
            return f"({left_str} {op.symbol} {right_str})"
        elif op in [Operator.MUL, Operator.DIV]:
            return f"{left_str} {op.symbol} {right_str}"
        elif op == Operator.POW:
            return f"{left_str}^{right_str}"
        else:
            return f"{op.symbol}({left_str}, {right_str})"


class ExpressionTree:
    """
    Expression tree for symbolic regression.
    
    Represents a mathematical equation as a tree structure.
    Can evaluate on PyTorch tensors.
    """
    
    def __init__(self, root: ExpressionNode):
        self.root = root
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate expression on input tensor.
        
        Args:
            x: Input tensor of shape (batch_size,) or (batch_size, 1)
        
        Returns:
            Output tensor of same shape
        """
        return self._evaluate_node(self.root, x)
    
    def _evaluate_node(
        self, 
        node: ExpressionNode, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """Recursively evaluate node."""
        
        # Variable
        if node.node_type == NodeType.VARIABLE:
            return x
        
        # Constant
        if node.node_type == NodeType.CONSTANT:
            return torch.full_like(x, node.value)
        
        # Operator
        op = node.operator
        
        # Evaluate children
        left_val = self._evaluate_node(node.left, x) if node.left else None
        right_val = self._evaluate_node(node.right, x) if node.right else None
        
        # Apply operator
        if op == Operator.ADD:
            return left_val + right_val
        elif op == Operator.SUB:
            return left_val - right_val
        elif op == Operator.MUL:
            return left_val * right_val
        elif op == Operator.DIV:
            # Protect against division by zero
            return left_val / (right_val + 1e-8)
        elif op == Operator.POW:
            # Protect against overflow
            return torch.pow(torch.abs(left_val) + 1e-8, 
                           torch.clamp(right_val, -10, 10))
        elif op == Operator.SIN:
            return torch.sin(left_val)
        elif op == Operator.COS:
            return torch.cos(left_val)
        elif op == Operator.EXP:
            # Protect against overflow
            return torch.exp(torch.clamp(left_val, -10, 10))
        elif op == Operator.LOG:
            # Protect against log(0)
            return torch.log(torch.abs(left_val) + 1e-8)
        elif op == Operator.SQRT:
            return torch.sqrt(torch.abs(left_val) + 1e-8)
        elif op == Operator.NEG:
            return -left_val
        else:
            raise ValueError(f"Unknown operator: {op}")
    
    def complexity(self) -> int:
        """Return tree complexity (total node count)."""
        return self.root.size()
    
    def depth(self) -> int:
        """Return tree depth."""
        return self.root.depth()
    
    def __str__(self) -> str:
        """Human-readable equation."""
        return f"y = {self.root}"
    
    def to_latex(self) -> str:
        """Convert to LaTeX format (TODO for later)."""
        return str(self)


# Factory functions
def make_constant(value: float) -> ExpressionNode:
    """Create constant node."""
    return ExpressionNode(NodeType.CONSTANT, value=value)


def make_variable() -> ExpressionNode:
    """Create variable node (x)."""
    return ExpressionNode(NodeType.VARIABLE)


def make_operator(
    op: Operator, 
    left: ExpressionNode, 
    right: Optional[ExpressionNode] = None
) -> ExpressionNode:
    """Create operator node."""
    return ExpressionNode(
        NodeType.OPERATOR,
        operator=op,
        left=left,
        right=right,
    )


# Tests
if __name__ == "__main__":
    print("=" * 60)
    print("SYMBOLIC REGRESSION - EXPRESSION TREE TESTS")
    print("=" * 60)
    print()
    
    # Test 1: Linear
    print("=== Test 1: y = 2*x + 3 ===")
    x_node = make_variable()
    const_2 = make_constant(2.0)
    const_3 = make_constant(3.0)
    mul_node = make_operator(Operator.MUL, const_2, x_node)
    add_node = make_operator(Operator.ADD, mul_node, const_3)
    tree = ExpressionTree(add_node)
    print(f"Expression: {tree}")
    print(f"Complexity: {tree.complexity()}")
    print(f"Depth: {tree.depth()}")
    x_test = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_pred = tree.evaluate(x_test)
    y_expected = 2*x_test + 3
    print(f"Input: {x_test}")
    print(f"Output: {y_pred}")
    print(f"Expected: {y_expected}")
    print(f"✓ PASS" if torch.allclose(y_pred, y_expected) else "✗ FAIL")
    print()
    
    # Test 2: Quadratic
    print("=== Test 2: y = x^2 ===")
    x_node = make_variable()
    const_2_node = make_constant(2.0)
    pow_node = make_operator(Operator.POW, x_node, const_2_node)
    tree2 = ExpressionTree(pow_node)
    print(f"Expression: {tree2}")
    y_pred2 = tree2.evaluate(x_test)
    y_expected2 = x_test**2
    print(f"Input: {x_test}")
    print(f"Output: {y_pred2}")
    print(f"Expected: {y_expected2}")
    print(f"✓ PASS" if torch.allclose(y_pred2, y_expected2) else "✗ FAIL")
    print()
    
    # Test 3: Trig
    print("=== Test 3: y = sin(x) + cos(x) ===")
    x_node1 = make_variable()
    x_node2 = make_variable()
    sin_node = make_operator(Operator.SIN, x_node1)
    cos_node = make_operator(Operator.COS, x_node2)
    add_node = make_operator(Operator.ADD, sin_node, cos_node)
    tree3 = ExpressionTree(add_node)
    print(f"Expression: {tree3}")
    x_test_trig = torch.linspace(0, 2*np.pi, 5)
    y_pred3 = tree3.evaluate(x_test_trig)
    y_expected3 = torch.sin(x_test_trig) + torch.cos(x_test_trig)
    print(f"Input: {x_test_trig}")
    print(f"Output: {y_pred3}")
    print(f"Expected: {y_expected3}")
    print(f"✓ PASS" if torch.allclose(y_pred3, y_expected3, atol=1e-6) else "✗ FAIL")
    print()
    
    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)