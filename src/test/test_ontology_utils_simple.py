"""Simple tests for the ontology utility module that don't require external dependencies."""
import os
import sys

# Add the src directory to the path so we can import pts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_module_import():
    """Test that the ontology module can be imported."""
    try:
        from pts.utils.ontology import ONTOMA_MAX_ATTEMPTS, _ontoma_udf, _simple_retry, add_efo_mapping
        assert ONTOMA_MAX_ATTEMPTS == 3
        assert callable(_simple_retry)
        assert callable(_ontoma_udf)
        assert callable(add_efo_mapping)
        print("✓ All ontology module functions imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_retry_function_basic():
    """Test basic functionality of _simple_retry without external dependencies."""
    try:
        from pts.utils.ontology import _simple_retry
        
        # Test successful function
        def success_func(x):
            return x * 2
        
        result = _simple_retry(success_func, x=5)
        assert result == 10
        print("✓ _simple_retry works with successful functions")
        
        # Test failing function (should return empty list after max attempts)
        def fail_func():
            raise Exception("Test failure")
        
        result = _simple_retry(fail_func)
        assert result == []
        print("✓ _simple_retry handles failures correctly")
        
        return True
    except Exception as e:
        print(f"✗ _simple_retry test failed: {e}")
        return False

def test_ontoma_udf_basic():
    """Test basic functionality of _ontoma_udf without external dependencies."""
    try:
        from pts.utils.ontology import _ontoma_udf
        
        # Test with empty row
        row = {'diseaseFromSource': None, 'diseaseFromSourceId': None}
        result = _ontoma_udf(row, None)
        assert result == []
        print("✓ _ontoma_udf handles empty rows correctly")
        
        return True
    except Exception as e:
        print(f"✗ _ontoma_udf test failed: {e}")
        return False

def run_simple_tests():
    """Run all simple tests."""
    print("Running simple ontology utility tests...")
    print("=" * 50)
    
    tests = [
        test_module_import,
        test_retry_function_basic,
        test_ontoma_udf_basic,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All simple tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False

if __name__ == "__main__":
    success = run_simple_tests()
    sys.exit(0 if success else 1)
