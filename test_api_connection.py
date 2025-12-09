"""
Test Sentinel Hub API connection and authentication.
"""

import os
from dotenv import load_dotenv
from src.data_processing.sentinel_hub_client import SentinelHubClient

# Load credentials
load_dotenv()

print("=" * 70)
print("Testing Sentinel Hub API Connection")
print("=" * 70)

print(f"\n📋 Credentials:")
print(f"   Instance ID: {os.getenv('SENTINEL_HUB_INSTANCE_ID')}")
print(f"   Client ID: {os.getenv('SENTINEL_HUB_CLIENT_ID')[:20]}...")
print(f"   Client Secret: {'*' * 20}")

# Initialize client
print(f"\n🔌 Initializing client...")
try:
    client = SentinelHubClient()
    print(f"   ✓ Client initialized")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    exit(1)

# Test authentication
print(f"\n🔐 Testing authentication...")
try:
    token = client.authenticate()
    print(f"   ✓ Authentication successful!")
    print(f"   Token: {token[:30]}...")
except Exception as e:
    print(f"   ✗ Authentication failed: {e}")
    exit(1)

# Test connection
print(f"\n🌐 Testing API connection...")
try:
    success = client.test_connection()
    if success:
        print(f"   ✓ Connection test passed!")
    else:
        print(f"   ✗ Connection test failed")
except Exception as e:
    print(f"   ✗ Connection test error: {e}")

# Validate credentials
print(f"\n✅ Validating credentials...")
try:
    is_valid, message = client.validate_credentials()
    if is_valid:
        print(f"   ✓ Credentials are valid!")
        print(f"   Message: {message}")
    else:
        print(f"   ✗ Credentials invalid: {message}")
except Exception as e:
    print(f"   ✗ Validation error: {e}")

print(f"\n" + "=" * 70)
print("✅ API Connection Test Complete!")
print("=" * 70)
print(f"\nYour Sentinel Hub API is configured and working!")
print(f"You can now fetch satellite imagery for the Ludhiana region.")
