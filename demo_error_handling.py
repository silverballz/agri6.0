#!/usr/bin/env python3
"""
Demonstration of AgriFlux Error Handling Framework
Shows how the error handling decorators work in practice
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.error_handler import safe_page, safe_operation, handle_data_loading, logger
from utils.dependency_checker import DependencyChecker


def demo_safe_page_success():
    """Demo: Successful page function"""
    print("\n" + "="*60)
    print("DEMO 1: Successful Page Function")
    print("="*60)
    
    @safe_page
    def successful_page():
        print("✅ Page loaded successfully!")
        return "Page content"
    
    result = successful_page()
    print(f"Result: {result}")


def demo_safe_page_file_error():
    """Demo: Page function with FileNotFoundError"""
    print("\n" + "="*60)
    print("DEMO 2: Page Function with FileNotFoundError")
    print("="*60)
    
    @safe_page
    def page_with_file_error():
        print("Attempting to load data file...")
        raise FileNotFoundError("data/missing_file.csv")
    
    result = page_with_file_error()
    print(f"Result: {result} (None indicates error was handled)")


def demo_safe_page_import_error():
    """Demo: Page function with ImportError"""
    print("\n" + "="*60)
    print("DEMO 3: Page Function with ImportError")
    print("="*60)
    
    @safe_page
    def page_with_import_error():
        print("Attempting to import missing package...")
        raise ImportError("No module named 'missing_package'")
    
    result = page_with_import_error()
    print(f"Result: {result} (None indicates error was handled)")


def demo_safe_operation():
    """Demo: Safe operation decorator"""
    print("\n" + "="*60)
    print("DEMO 4: Safe Operation Decorator")
    print("="*60)
    
    @safe_operation("Data processing")
    def process_data():
        print("Processing data...")
        raise ValueError("Invalid data format")
    
    result = process_data()
    print(f"Result: {result} (None indicates error was handled)")


def demo_handle_data_loading():
    """Demo: Data loading decorator"""
    print("\n" + "="*60)
    print("DEMO 5: Data Loading Decorator")
    print("="*60)
    
    @handle_data_loading
    def load_data():
        print("Loading data from database...")
        return {"records": 100, "status": "success"}
    
    result = load_data()
    print(f"Result: {result}")


def demo_dependency_checker():
    """Demo: Dependency checker"""
    print("\n" + "="*60)
    print("DEMO 6: Dependency Checker")
    print("="*60)
    
    checker = DependencyChecker()
    results = checker.check_all()
    
    print(f"\nOverall Status: {results['overall_status'].upper()}")
    print(f"Status Message: {results.get('status_message', 'N/A')}")
    
    print("\nRequired Packages:")
    for pkg_key, info in results['required_packages'].items():
        status_icon = "✅" if info['status'] == 'installed' else "❌"
        print(f"  {status_icon} {info['display_name']}: {info['status']}")
    
    print("\nOptional Packages:")
    for pkg_key, info in results['optional_packages'].items():
        status_icon = "✅" if info['status'] == 'installed' else "⚪"
        print(f"  {status_icon} {info['display_name']}: {info['status']}")
    
    print("\nCritical Paths:")
    for path_key, info in results['critical_paths'].items():
        status_icon = "✅" if info['status'] == 'exists' else "❌"
        print(f"  {status_icon} {info['path']}: {info['status']}")


def main():
    """Run all demonstrations"""
    print("\n" + "="*60)
    print("AgriFlux Error Handling Framework Demonstration")
    print("="*60)
    print("\nThis demo shows how the error handling framework")
    print("gracefully handles various error conditions.")
    
    # Run demos
    demo_safe_page_success()
    demo_safe_page_file_error()
    demo_safe_page_import_error()
    demo_safe_operation()
    demo_handle_data_loading()
    demo_dependency_checker()
    
    print("\n" + "="*60)
    print("Demonstration Complete!")
    print("="*60)
    print("\nKey Takeaways:")
    print("✅ All errors are caught and handled gracefully")
    print("✅ User-friendly error messages are displayed")
    print("✅ Functions return None on error (no crashes)")
    print("✅ Detailed logging for debugging")
    print("✅ System health monitoring on startup")
    print("\nThe dashboard will never crash during demos! 🎉")


if __name__ == "__main__":
    main()
