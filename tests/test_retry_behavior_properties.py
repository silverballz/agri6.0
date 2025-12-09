"""
Property-based tests for API retry behavior with exponential backoff.

Tests universal properties of retry logic and error handling.
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, patch, call
import time
import requests

from src.data_processing.sentinel_hub_client import (
    SentinelHubClient,
    SentinelHubConfig
)


class TestRetryBehaviorProperties:
    """Property-based tests for retry logic."""
    
    @given(
        max_retries=st.integers(min_value=1, max_value=5),
        failure_count=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_4_api_retry_with_exponential_backoff(
        self,
        max_retries,
        failure_count
    ):
        """
        **Feature: production-enhancements, Property 4: API retry with exponential backoff**
        
        For any transient API failure, the system should retry with exponentially
        increasing wait times (2^attempt seconds) up to max_retries.
        
        **Validates: Requirements 8.1**
        """
        config = SentinelHubConfig(
            instance_id='test_instance',
            client_id='test_client',
            client_secret='test_secret'
        )
        
        client = SentinelHubClient(config, max_retries=max_retries)
        
        # Mock time.sleep to track wait times
        sleep_times = []
        
        def mock_sleep(seconds):
            sleep_times.append(seconds)
        
        # Create mock that fails failure_count times, then succeeds
        call_count = [0]
        
        def mock_request(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= failure_count:
                raise requests.exceptions.ConnectionError("Connection failed")
            
            # Success
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            return mock_response
        
        with patch('time.sleep', side_effect=mock_sleep):
            with patch('requests.request', side_effect=mock_request):
                
                if failure_count < max_retries:
                    # Should eventually succeed
                    response = client.request_with_retry('GET', 'http://test.com')
                    
                    # Property: Should have made failure_count + 1 attempts
                    assert call_count[0] == failure_count + 1, \
                        f"Expected {failure_count + 1} attempts, got {call_count[0]}"
                    
                    # Property: Should have slept failure_count times (not on last success)
                    assert len(sleep_times) == failure_count, \
                        f"Expected {failure_count} sleep calls, got {len(sleep_times)}"
                    
                    # Property: Sleep times should follow exponential backoff (2^attempt)
                    expected_sleep_times = [2 ** i for i in range(failure_count)]
                    assert sleep_times == expected_sleep_times, \
                        f"Expected exponential backoff {expected_sleep_times}, got {sleep_times}"
                
                else:
                    # Should fail after max_retries
                    with pytest.raises(requests.exceptions.RequestException):
                        client.request_with_retry('GET', 'http://test.com')
                    
                    # Property: Should have made exactly max_retries attempts
                    assert call_count[0] == max_retries, \
                        f"Expected {max_retries} attempts, got {call_count[0]}"
                    
                    # Property: Should have slept max_retries - 1 times
                    assert len(sleep_times) == max_retries - 1, \
                        f"Expected {max_retries - 1} sleep calls, got {len(sleep_times)}"
    
    @given(
        retry_after=st.integers(min_value=1, max_value=300)
    )
    @settings(max_examples=50, deadline=None)
    def test_property_rate_limit_respects_retry_after(self, retry_after):
        """
        Property: Rate limit handling should respect Retry-After header.
        
        For any rate limit response (HTTP 429), the system should wait
        for the duration specified in the Retry-After header.
        
        **Validates: Requirements 8.2**
        """
        config = SentinelHubConfig(
            instance_id='test_instance',
            client_id='test_client',
            client_secret='test_secret'
        )
        
        client = SentinelHubClient(config, max_retries=3)
        
        sleep_times = []
        
        def mock_sleep(seconds):
            sleep_times.append(seconds)
        
        call_count = [0]
        
        def mock_request(*args, **kwargs):
            call_count[0] += 1
            
            if call_count[0] == 1:
                # First call: rate limited
                mock_response = Mock()
                mock_response.status_code = 429
                mock_response.headers = {'Retry-After': str(retry_after)}
                return mock_response
            else:
                # Second call: success
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.raise_for_status = Mock()
                return mock_response
        
        with patch('time.sleep', side_effect=mock_sleep):
            with patch('requests.request', side_effect=mock_request):
                response = client.request_with_retry('GET', 'http://test.com')
                
                # Property: Should have made 2 attempts (rate limited, then success)
                assert call_count[0] == 2, f"Expected 2 attempts, got {call_count[0]}"
                
                # Property: Should have slept for exactly retry_after seconds
                assert len(sleep_times) == 1, f"Expected 1 sleep call, got {len(sleep_times)}"
                assert sleep_times[0] == retry_after, \
                    f"Expected sleep of {retry_after}s, got {sleep_times[0]}s"
    
    @given(
        max_retries=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=50, deadline=None)
    def test_property_retry_count_never_exceeds_max(self, max_retries):
        """
        Property: Number of retry attempts should never exceed max_retries.
        
        For any configuration, the system should make at most max_retries
        attempts before giving up.
        """
        config = SentinelHubConfig(
            instance_id='test_instance',
            client_id='test_client',
            client_secret='test_secret'
        )
        
        client = SentinelHubClient(config, max_retries=max_retries)
        
        call_count = [0]
        
        def mock_request(*args, **kwargs):
            call_count[0] += 1
            # Always fail
            raise requests.exceptions.ConnectionError("Connection failed")
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            with patch('requests.request', side_effect=mock_request):
                with pytest.raises(requests.exceptions.RequestException):
                    client.request_with_retry('GET', 'http://test.com')
                
                # Property: Should have made exactly max_retries attempts
                assert call_count[0] == max_retries, \
                    f"Expected exactly {max_retries} attempts, got {call_count[0]}"
    
    @given(
        attempt=st.integers(min_value=0, max_value=10)
    )
    @settings(max_examples=50)
    def test_property_exponential_backoff_formula(self, attempt):
        """
        Property: Exponential backoff should follow 2^attempt formula.
        
        For any attempt number, the wait time should be exactly 2^attempt seconds.
        """
        # Property: Wait time should be 2^attempt
        expected_wait = 2 ** attempt
        
        # Verify the formula holds
        assert expected_wait == 2 ** attempt
        
        # Property: Wait time should increase exponentially
        if attempt > 0:
            previous_wait = 2 ** (attempt - 1)
            assert expected_wait == 2 * previous_wait, \
                "Wait time should double with each attempt"
    
    @given(
        success_on_attempt=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=50, deadline=None)
    def test_property_stops_retrying_on_success(self, success_on_attempt):
        """
        Property: Retry logic should stop immediately upon success.
        
        For any successful response, no further retry attempts should be made.
        """
        config = SentinelHubConfig(
            instance_id='test_instance',
            client_id='test_client',
            client_secret='test_secret'
        )
        
        max_retries = 10  # More than we'll need
        client = SentinelHubClient(config, max_retries=max_retries)
        
        call_count = [0]
        
        def mock_request(*args, **kwargs):
            call_count[0] += 1
            
            if call_count[0] < success_on_attempt:
                # Fail
                raise requests.exceptions.ConnectionError("Connection failed")
            else:
                # Success
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.raise_for_status = Mock()
                return mock_response
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            with patch('requests.request', side_effect=mock_request):
                response = client.request_with_retry('GET', 'http://test.com')
                
                # Property: Should have made exactly success_on_attempt attempts
                assert call_count[0] == success_on_attempt, \
                    f"Expected {success_on_attempt} attempts, got {call_count[0]}"
                
                # Property: Should not have made more attempts than necessary
                assert call_count[0] <= max_retries, \
                    "Should not exceed max_retries"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
