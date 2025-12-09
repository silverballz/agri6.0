"""
Unit tests for UI components
Tests CSS loading, font availability, responsive breakpoints, and color contrast
Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
"""

import pytest
import os
from pathlib import Path
import re
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dashboard.ui_components import (
    apply_custom_theme,
    load_custom_fonts,
    ColorScheme,
    Icons,
    HelpText
)


class TestCSSLoading:
    """Test CSS file loading and application"""
    
    def test_css_file_exists(self):
        """Test that custom_theme.css file exists"""
        css_path = Path(__file__).parent.parent / 'src' / 'dashboard' / 'styles' / 'custom_theme.css'
        assert css_path.exists(), f"CSS file not found at {css_path}"
    
    def test_css_file_readable(self):
        """Test that CSS file can be read"""
        css_path = Path(__file__).parent.parent / 'src' / 'dashboard' / 'styles' / 'custom_theme.css'
        
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert len(content) > 0, "CSS file is empty"
        assert '@import' in content or 'font-family' in content, "CSS file missing font imports"
    
    def test_css_contains_required_sections(self):
        """Test that CSS file contains all required sections"""
        css_path = Path(__file__).parent.parent / 'src' / 'dashboard' / 'styles' / 'custom_theme.css'
        
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_sections = [
            'GOOGLE FONTS IMPORT',
            'CSS VARIABLES',
            'TYPOGRAPHY',
            'GRID BACKGROUND PATTERN',
            'COMPONENT STYLING - CARDS',
            'COMPONENT STYLING - BUTTONS',
            'COMPONENT STYLING - METRICS',
            'COMPONENT STYLING - TABLES',
            'HOVER ANIMATIONS',
            'RESPONSIVE DESIGN - TABLET',
            'RESPONSIVE DESIGN - DESKTOP'
        ]
        
        for section in required_sections:
            assert section in content, f"CSS file missing required section: {section}"
    
    def test_css_variables_defined(self):
        """Test that CSS variables are properly defined"""
        css_path = Path(__file__).parent.parent / 'src' / 'dashboard' / 'styles' / 'custom_theme.css'
        
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_variables = [
            '--primary-color',
            '--secondary-color',
            '--accent-color',
            '--bg-primary',
            '--text-primary',
            '--border-color',
            '--spacing-md',
            '--radius-md',
            '--transition-base'
        ]
        
        for var in required_variables:
            assert var in content, f"CSS file missing required variable: {var}"


class TestFontAvailability:
    """Test font loading and availability"""
    
    def test_google_fonts_import(self):
        """Test that Google Fonts are imported"""
        css_path = Path(__file__).parent.parent / 'src' / 'dashboard' / 'styles' / 'custom_theme.css'
        
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for Google Fonts import
        assert 'fonts.googleapis.com' in content, "Google Fonts import not found"
        assert 'Inter' in content, "Inter font not imported"
        assert 'Roboto' in content, "Roboto font not imported"
    
    def test_font_family_declarations(self):
        """Test that font families are declared in CSS"""
        css_path = Path(__file__).parent.parent / 'src' / 'dashboard' / 'styles' / 'custom_theme.css'
        
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for font-family declarations
        assert 'font-family:' in content.lower(), "No font-family declarations found"
        
        # Check that Inter or Roboto is used
        font_families = re.findall(r'font-family:\s*([^;]+);', content, re.IGNORECASE)
        assert len(font_families) > 0, "No font-family declarations found"
        
        # At least one should use Inter or Roboto
        uses_custom_fonts = any('Inter' in f or 'Roboto' in f for f in font_families)
        assert uses_custom_fonts, "Custom fonts (Inter/Roboto) not used in font-family declarations"


class TestResponsiveBreakpoints:
    """Test responsive design media queries"""
    
    def test_tablet_breakpoint_exists(self):
        """Test that tablet breakpoint (768px) is defined"""
        css_path = Path(__file__).parent.parent / 'src' / 'dashboard' / 'styles' / 'custom_theme.css'
        
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for tablet media query
        assert '@media' in content, "No media queries found"
        assert '768px' in content, "Tablet breakpoint (768px) not found"
        assert 'max-width: 768px' in content, "Tablet max-width media query not found"
    
    def test_desktop_breakpoint_exists(self):
        """Test that desktop breakpoint (1024px) is defined"""
        css_path = Path(__file__).parent.parent / 'src' / 'dashboard' / 'styles' / 'custom_theme.css'
        
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for desktop media query
        assert '1024px' in content, "Desktop breakpoint (1024px) not found"
        assert 'min-width: 1024px' in content, "Desktop min-width media query not found"
    
    def test_responsive_adjustments(self):
        """Test that responsive adjustments are made in media queries"""
        css_path = Path(__file__).parent.parent / 'src' / 'dashboard' / 'styles' / 'custom_theme.css'
        
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract media query blocks
        media_queries = re.findall(r'@media[^{]+\{[^}]+\}', content, re.DOTALL)
        
        assert len(media_queries) >= 2, "Not enough media queries found (expected at least 2)"
        
        # Check that media queries contain style adjustments
        for mq in media_queries:
            # Should contain at least some CSS properties
            assert ':' in mq, f"Media query missing CSS properties: {mq[:100]}"


class TestColorContrast:
    """Test color contrast for accessibility"""
    
    def test_color_scheme_class_exists(self):
        """Test that ColorScheme class is defined"""
        assert hasattr(ColorScheme, 'PRIMARY'), "ColorScheme missing PRIMARY color"
        assert hasattr(ColorScheme, 'SECONDARY'), "ColorScheme missing SECONDARY color"
        assert hasattr(ColorScheme, 'TEXT_PRIMARY'), "ColorScheme missing TEXT_PRIMARY color"
        assert hasattr(ColorScheme, 'BG_DARK'), "ColorScheme missing BG_DARK color"
    
    def test_primary_color_value(self):
        """Test that primary color matches design specification"""
        assert ColorScheme.PRIMARY == "#4caf50", f"Primary color should be #4caf50, got {ColorScheme.PRIMARY}"
    
    def test_secondary_color_value(self):
        """Test that secondary color matches design specification"""
        assert ColorScheme.SECONDARY == "#2196F3", f"Secondary color should be #2196F3, got {ColorScheme.SECONDARY}"
    
    def test_color_format(self):
        """Test that colors are in valid hex format"""
        colors = [
            ColorScheme.PRIMARY,
            ColorScheme.SECONDARY,
            ColorScheme.SUCCESS,
            ColorScheme.WARNING,
            ColorScheme.ERROR
        ]
        
        hex_pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')
        
        for color in colors:
            assert hex_pattern.match(color), f"Invalid hex color format: {color}"
    
    def test_contrast_ratio_sufficient(self):
        """Test that text colors have sufficient contrast against backgrounds"""
        
        def hex_to_rgb(hex_color):
            """Convert hex color to RGB tuple"""
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        def relative_luminance(rgb):
            """Calculate relative luminance of RGB color"""
            r, g, b = [x / 255.0 for x in rgb]
            
            def adjust(c):
                return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
            
            r, g, b = adjust(r), adjust(g), adjust(b)
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        def contrast_ratio(color1, color2):
            """Calculate contrast ratio between two colors"""
            lum1 = relative_luminance(hex_to_rgb(color1))
            lum2 = relative_luminance(hex_to_rgb(color2))
            
            lighter = max(lum1, lum2)
            darker = min(lum1, lum2)
            
            return (lighter + 0.05) / (darker + 0.05)
        
        # Test text on background contrast (WCAG AA requires 4.5:1 for normal text)
        text_bg_contrast = contrast_ratio(ColorScheme.TEXT_PRIMARY, ColorScheme.BG_DARK)
        assert text_bg_contrast >= 4.5, f"Text/background contrast ratio {text_bg_contrast:.2f} is below WCAG AA standard (4.5:1)"
        
        # Test primary color on dark background (should be visible)
        primary_bg_contrast = contrast_ratio(ColorScheme.PRIMARY, ColorScheme.BG_DARK)
        assert primary_bg_contrast >= 3.0, f"Primary/background contrast ratio {primary_bg_contrast:.2f} is too low"


class TestComponentStyling:
    """Test component styling definitions"""
    
    def test_card_styling_exists(self):
        """Test that card styling is defined"""
        css_path = Path(__file__).parent.parent / 'src' / 'dashboard' / 'styles' / 'custom_theme.css'
        
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert '.card' in content, "Card styling not found"
        assert 'border-radius' in content, "Border radius not defined"
        assert 'box-shadow' in content, "Box shadow not defined"
    
    def test_button_styling_exists(self):
        """Test that button styling is defined"""
        css_path = Path(__file__).parent.parent / 'src' / 'dashboard' / 'styles' / 'custom_theme.css'
        
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert '.stButton' in content or 'button' in content, "Button styling not found"
        assert 'gradient' in content.lower(), "Gradient not used in styling"
    
    def test_metric_styling_exists(self):
        """Test that metric card styling is defined"""
        css_path = Path(__file__).parent.parent / 'src' / 'dashboard' / 'styles' / 'custom_theme.css'
        
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert '.metric-container' in content, "Metric container styling not found"
        assert '.metric-value' in content, "Metric value styling not found"
        assert '.metric-delta' in content, "Metric delta styling not found"
    
    def test_hover_animations_exist(self):
        """Test that hover animations are defined"""
        css_path = Path(__file__).parent.parent / 'src' / 'dashboard' / 'styles' / 'custom_theme.css'
        
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert ':hover' in content, "Hover pseudo-class not found"
        assert 'transition' in content.lower(), "Transitions not defined"
        
        # Check for specific hover effects
        hover_effects = ['transform', 'box-shadow', 'opacity']
        found_effects = [effect for effect in hover_effects if effect in content.lower()]
        assert len(found_effects) >= 2, f"Not enough hover effects found. Expected at least 2 of {hover_effects}, found {found_effects}"


class TestIconsAndHelpText:
    """Test Icons and HelpText classes"""
    
    def test_icons_class_exists(self):
        """Test that Icons class is defined with required icons"""
        required_icons = ['EXCELLENT', 'HEALTHY', 'MODERATE', 'STRESSED', 'CRITICAL']
        
        for icon in required_icons:
            assert hasattr(Icons, icon), f"Icons class missing {icon}"
    
    def test_help_text_class_exists(self):
        """Test that HelpText class is defined with required help text"""
        assert hasattr(HelpText, 'VEGETATION_INDICES'), "HelpText missing VEGETATION_INDICES"
        assert hasattr(HelpText, 'ALERT_SEVERITIES'), "HelpText missing ALERT_SEVERITIES"
        assert hasattr(HelpText, 'FEATURES'), "HelpText missing FEATURES"
    
    def test_vegetation_indices_help_complete(self):
        """Test that all vegetation indices have help text"""
        required_indices = ['NDVI', 'SAVI', 'EVI', 'NDWI']
        
        for index in required_indices:
            assert index in HelpText.VEGETATION_INDICES, f"Missing help text for {index}"
            help_text = HelpText.VEGETATION_INDICES[index]
            assert len(help_text) > 50, f"Help text for {index} is too short"
            assert 'Range:' in help_text, f"Help text for {index} missing range information"


class TestThemeLoaderFunction:
    """Test the apply_custom_theme function"""
    
    def test_apply_custom_theme_callable(self):
        """Test that apply_custom_theme function exists and is callable"""
        assert callable(apply_custom_theme), "apply_custom_theme is not callable"
    
    def test_load_custom_fonts_callable(self):
        """Test that load_custom_fonts function exists and is callable"""
        assert callable(load_custom_fonts), "load_custom_fonts is not callable"


class TestUtilityClasses:
    """Test utility classes in CSS"""
    
    def test_text_alignment_classes(self):
        """Test that text alignment utility classes exist"""
        css_path = Path(__file__).parent.parent / 'src' / 'dashboard' / 'styles' / 'custom_theme.css'
        
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert '.text-center' in content, "text-center utility class not found"
        assert '.text-left' in content, "text-left utility class not found"
        assert '.text-right' in content, "text-right utility class not found"
    
    def test_spacing_classes(self):
        """Test that spacing utility classes exist"""
        css_path = Path(__file__).parent.parent / 'src' / 'dashboard' / 'styles' / 'custom_theme.css'
        
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for margin and padding utilities
        assert '.mt-' in content or '.mb-' in content, "Margin utility classes not found"
        assert '.p-' in content, "Padding utility classes not found"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
