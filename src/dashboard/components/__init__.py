"""
Dashboard components package.

Contains reusable UI components for the AgriFlux dashboard.
"""

from .comparison_widget import (
    ComparisonWidget,
    render_comparison_widget,
    render_multi_date_slider
)

__all__ = [
    'ComparisonWidget',
    'render_comparison_widget',
    'render_multi_date_slider'
]
