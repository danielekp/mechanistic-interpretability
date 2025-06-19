#!/usr/bin/env python3

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from circuit_tracer import ReplacementModel

def test_transcoder_interface():
    """Test the transcoder interface to understand how to access semantic tokens."""
    
    print("Testing circuit-tracer transcoder interface...")
    
    try:
        # Load model with transcoder
        model = ReplacementModel.from_pretrained(
            'google/gemma-2-2b',
            'gemma',
            dtype=torch.bfloat16,
            device='cpu'  # Use CPU for testing
        )
        
        print("✓ Model loaded successfully")
        
        # Check if there's a transcoders attribute
        if hasattr(model, 'transcoders'):
            print(f"\n✓ Transcoders found: {type(model.transcoders)}")
            
            # Explore transcoders attributes
            print("\nTranscoders attributes:")
            for attr in dir(model.transcoders):
                if not attr.startswith('_'):
                    print(f"  - {attr}")
        
        # Check for any methods that might give us token data
        transcoder_methods = []
        for attr in dir(model):
            if not attr.startswith('_') and callable(getattr(model, attr, None)):
                if 'token' in attr.lower():
                    transcoder_methods.append(attr)
        
        if transcoder_methods:
            print(f"\nPotential token-related methods: {transcoder_methods}")
        
        # Try to find feature-to-token mapping methods
        feature_methods = []
        for attr in dir(model):
            if not attr.startswith('_') and callable(getattr(model, attr, None)):
                if 'feature' in attr.lower() or 'transcoder' in attr.lower():
                    feature_methods.append(attr)
        
        if feature_methods:
            print(f"\nPotential feature-related methods: {feature_methods}")
        
        # Test if we can access layer information
        if hasattr(model, 'cfg'):
            print(f"\nModel config: {model.cfg}")
        
        # Try to access transcoder data directly
        if hasattr(model, 'transcoders'):
            print("\nExploring transcoder data...")
            
            # Check if transcoders has any methods for getting tokens
            transcoder_methods = []
            for attr in dir(model.transcoders):
                if not attr.startswith('_') and callable(getattr(model.transcoders, attr, None)):
                    transcoder_methods.append(attr)
            
            print(f"Transcoder methods: {transcoder_methods}")
            
            # Try to get some basic info about the transcoder
            if hasattr(model.transcoders, '__len__'):
                print(f"Number of transcoders: {len(model.transcoders)}")
            
            # Check if we can access individual transcoders
            if hasattr(model.transcoders, '__getitem__'):
                try:
                    first_transcoder = model.transcoders[0]
                    print(f"First transcoder type: {type(first_transcoder)}")
                    
                    # Explore first transcoder
                    transcoder_attrs = []
                    for attr in dir(first_transcoder):
                        if not attr.startswith('_'):
                            transcoder_attrs.append(attr)
                    
                    print(f"First transcoder attributes: {transcoder_attrs[:10]}...")  # Show first 10
                    
                except Exception as e:
                    print(f"Could not access first transcoder: {e}")
        
        print("\n✓ Transcoder interface test completed")
        
    except Exception as e:
        print(f"Error testing transcoder interface: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_transcoder_interface() 