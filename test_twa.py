#!/usr/bin/env python3
"""
Test script to verify the TWA package structure and imports
"""

def test_imports():
    """Test that all main modules can be imported successfully"""
    try:
        import twa
        print("✅ twa package imported successfully")
        
        from twa.config import Config
        print("✅ Config imported successfully")
        
        from twa.modules.modules import SlotBasedVideoModel, SlotEncoder, SlotDecoder, TripleWiseAttention
        print("✅ Main model classes imported successfully")
        
        from twa.modules.rope import Sinusoidal2DPositionEmbed
        print("✅ Position embedding classes imported successfully")
        
        from twa.modules.utils import LatentDataset, init_slot_prototype
        print("✅ Utility functions imported successfully")
        
        print("\n🎉 All imports successful! The TWA package is properly structured.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_config():
    """Test that configuration is accessible"""
    try:
        from twa.config import Config
        print(f"✅ Config loaded - batch_size: {Config.batch_size}, slot_dim: {Config.slot_dim}")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_triple_wise_attention():
    """Test that triple-wise attention can be instantiated and used"""
    try:
        from twa.modules.modules import TripleWiseAttention
        import torch
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping triple-wise attention test (requires flash attention)")
            return True
        
        # Test triple-wise attention
        dim = 256
        num_heads = 4
        batch_size = 2
        seq_len = 10
        
        attn = TripleWiseAttention(dim=dim, num_heads=num_heads).cuda().half()
        x = torch.randn(batch_size, seq_len, dim).cuda().half()
        
        output = attn(x)
        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
        
        print(f"✅ TripleWiseAttention created and tested successfully on CUDA - input/output shape: {x.shape}")
        return True
    except Exception as e:
        print(f"❌ Triple-wise attention test failed: {e}")
        return False

def test_config_triple_wise():
    """Test that triple-wise attention config is accessible"""
    try:
        from twa.config import Config
        print(f"✅ Triple-wise attention config: use_triple_wise_attention = {Config.use_triple_wise_attention}")
        return True
    except Exception as e:
        print(f"❌ Triple-wise config test failed: {e}")
        return False

def test_model_creation():
    """Test that models can be instantiated"""
    try:
        from twa.modules.modules import SlotBasedVideoModel
        from twa.config import Config
        
        model = SlotBasedVideoModel(Config, pre_trained=True)
        print(f"✅ SlotBasedVideoModel created successfully - {sum(p.numel() for p in model.parameters())} parameters")
        return True
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing TWA Package Structure\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Config Test", test_config),
        ("Triple-wise Config Test", test_config_triple_wise),
        ("Triple-wise Attention Test", test_triple_wise_attention),
        ("Model Creation Test", test_model_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The TWA package is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
