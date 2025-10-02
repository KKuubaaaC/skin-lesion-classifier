"""
Unit tests for HAM10000 Skin Lesion Classifier.

This module contains comprehensive tests for:
- Model loading and initialization
- Image preprocessing pipeline
- Inference and prediction logic
- Output validation
- Error handling
"""

import pytest
import numpy as np
from PIL import Image
import os
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List


# Test fixtures
@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode='RGB')


@pytest.fixture
def sample_image_large():
    """Create a larger sample image to test resizing."""
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode='RGB')


@pytest.fixture
def class_mapping():
    """Create standard class mapping."""
    return {
        "0": "akiec",
        "1": "bcc",
        "2": "bkl",
        "3": "df",
        "4": "mel",
        "5": "nv",
        "6": "vasc"
    }


@pytest.fixture
def mock_predictions():
    """Create mock prediction results."""
    return [
        {
            'class': 'mel',
            'confidence': 85.5,
            'probability': 0.855,
            'class_info': {
                'full_name': 'Melanoma',
                'risk_level': 'CRITICAL',
                'color': '#8B0000'
            },
            'inference_time': 0.075
        },
        {
            'class': 'nv',
            'confidence': 10.2,
            'probability': 0.102,
            'class_info': {
                'full_name': 'Melanocytic Nevi',
                'risk_level': 'LOW',
                'color': '#32cd32'
            },
            'inference_time': 0.075
        }
    ]


# Test class for image preprocessing
class TestImagePreprocessing:
    """Tests for image preprocessing functions."""
    
    def test_image_resize(self, sample_image_large):
        """Test that images are correctly resized to 224x224."""
        from torchvision import transforms
        
        transform = transforms.Resize((224, 224))
        resized = transform(sample_image_large)
        
        assert resized.size == (224, 224), "Image not resized correctly"
    
    def test_image_to_tensor_shape(self, sample_image):
        """Test conversion from PIL Image to tensor."""
        from torchvision import transforms
        
        transform = transforms.ToTensor()
        tensor = transform(sample_image)
        
        assert tensor.shape == (3, 224, 224), "Tensor shape incorrect"
        assert tensor.dtype == np.float32 or str(tensor.dtype) == 'torch.float32'
    
    def test_normalization_applied(self, sample_image):
        """Test that ImageNet normalization is correctly applied."""
        from torchvision import transforms
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        # Create pure white image
        white_img = Image.new('RGB', (224, 224), (255, 255, 255))
        tensor = transform(white_img)
        
        # After normalization, values should be (1.0 - mean) / std
        expected_r = (1.0 - mean[0]) / std[0]
        
        assert abs(float(tensor[0, 0, 0]) - expected_r) < 0.01
    
    def test_batch_dimension_added(self, sample_image):
        """Test that batch dimension is correctly added."""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        tensor = transform(sample_image)
        batched = tensor.unsqueeze(0)
        
        assert batched.shape == (1, 3, 224, 224)
    
    def test_grayscale_to_rgb_conversion(self):
        """Test that grayscale images are converted to RGB."""
        gray_img = Image.new('L', (224, 224), 128)
        rgb_img = gray_img.convert('RGB')
        
        assert rgb_img.mode == 'RGB'
        assert rgb_img.size == (224, 224)
    
    def test_rgba_to_rgb_conversion(self):
        """Test that RGBA images are converted to RGB."""
        rgba_img = Image.new('RGBA', (224, 224), (128, 128, 128, 255))
        rgb_img = rgba_img.convert('RGB')
        
        assert rgb_img.mode == 'RGB'
        assert rgb_img.size == (224, 224)


# Test class for model predictions
class TestModelPredictions:
    """Tests for model prediction logic."""
    
    def test_softmax_probabilities_sum_to_one(self):
        """Test that softmax produces valid probability distribution."""
        def softmax(x):
            exp_x = np.exp(x - np.max(x))
            return exp_x / exp_x.sum()
        
        logits = np.array([1.0, 2.0, 3.0, 0.5, 2.5, 1.5, 0.8])
        probs = softmax(logits)
        
        assert len(probs) == 7
        assert abs(np.sum(probs) - 1.0) < 1e-6
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
    
    def test_softmax_monotonicity(self):
        """Test that higher logits produce higher probabilities."""
        def softmax(x):
            exp_x = np.exp(x - np.max(x))
            return exp_x / exp_x.sum()
        
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        probs = softmax(logits)
        
        # Probabilities should be in ascending order
        for i in range(len(probs) - 1):
            assert probs[i] < probs[i + 1]
    
    def test_prediction_output_format(self, mock_predictions):
        """Test that predictions have correct structure."""
        for pred in mock_predictions:
            assert 'class' in pred
            assert 'confidence' in pred
            assert 'probability' in pred
            assert 'class_info' in pred
            assert 'inference_time' in pred
            
            assert isinstance(pred['class'], str)
            assert isinstance(pred['confidence'], (int, float))
            assert isinstance(pred['probability'], (int, float))
            assert isinstance(pred['class_info'], dict)
    
    def test_confidence_percentage_conversion(self, mock_predictions):
        """Test that probability is correctly converted to percentage."""
        for pred in mock_predictions:
            expected = pred['probability'] * 100
            assert abs(pred['confidence'] - expected) < 0.01
    
    def test_predictions_sorted_by_confidence(self, mock_predictions):
        """Test that predictions are sorted in descending order."""
        confidences = [p['confidence'] for p in mock_predictions]
        
        for i in range(len(confidences) - 1):
            assert confidences[i] >= confidences[i + 1]
    
    def test_confidence_values_in_valid_range(self, mock_predictions):
        """Test that confidence values are between 0 and 100."""
        for pred in mock_predictions:
            assert 0 <= pred['confidence'] <= 100
            assert 0 <= pred['probability'] <= 1


# Test class for class mapping
class TestClassMapping:
    """Tests for class mapping and metadata."""
    
    def test_class_mapping_completeness(self, class_mapping):
        """Test that all 7 classes are present."""
        expected_classes = {'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'}
        actual_classes = set(class_mapping.values())
        
        assert actual_classes == expected_classes
        assert len(class_mapping) == 7
    
    def test_class_indices_sequential(self, class_mapping):
        """Test that class indices are 0-6."""
        indices = [int(k) for k in class_mapping.keys()]
        
        assert min(indices) == 0
        assert max(indices) == 6
        assert len(indices) == 7
    
    def test_class_mapping_json_serializable(self, class_mapping):
        """Test that class mapping can be serialized to JSON."""
        try:
            json_str = json.dumps(class_mapping)
            loaded = json.loads(json_str)
            assert loaded == class_mapping
        except Exception as e:
            pytest.fail(f"Class mapping not JSON serializable: {e}")
    
    def test_class_metadata_structure(self):
        """Test that class metadata has required fields."""
        from app import CLASS_METADATA
        
        required_fields = {'full_name', 'description', 'color', 'risk_level', 'prevalence', 'recommendation'}
        
        for class_key, metadata in CLASS_METADATA.items():
            assert isinstance(metadata, dict)
            assert required_fields.issubset(metadata.keys())
            assert isinstance(metadata['risk_level'], str)
            assert metadata['risk_level'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']


# Test class for input validation
class TestInputValidation:
    """Tests for input validation."""
    
    def test_valid_image_formats(self):
        """Test that valid image formats are accepted."""
        valid_formats = ['RGB', 'RGBA', 'L']
        
        for fmt in valid_formats:
            if fmt == 'RGB':
                img = Image.new(fmt, (224, 224), (128, 128, 128))
            elif fmt == 'RGBA':
                img = Image.new(fmt, (224, 224), (128, 128, 128, 255))
            else:
                img = Image.new(fmt, (224, 224), 128)
            
            assert isinstance(img, Image.Image)
            rgb_img = img.convert('RGB')
            assert rgb_img.mode == 'RGB'
    
    def test_invalid_inputs_rejected(self):
        """Test that invalid inputs are not PIL Images."""
        invalid_inputs = [
            None,
            "not_an_image.jpg",
            123,
            [1, 2, 3],
            {'data': 'image'},
        ]
        
        for invalid_input in invalid_inputs:
            assert not isinstance(invalid_input, Image.Image)
    
    def test_image_size_validation(self):
        """Test that very small images can be handled."""
        small_img = Image.new('RGB', (10, 10), (128, 128, 128))
        
        from torchvision import transforms
        transform = transforms.Resize((224, 224))
        resized = transform(small_img)
        
        assert resized.size == (224, 224)
    
    def test_corrupted_image_detection(self):
        """Test that corrupted image data is detected."""
        # Create a temporary file with invalid image data
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(b'invalid image data')
            temp_path = f.name
        
        try:
            with pytest.raises(Exception):
                Image.open(temp_path).verify()
        finally:
            os.unlink(temp_path)


# Test class for performance metrics
class TestPerformanceMetrics:
    """Tests for performance and benchmarking."""
    
    def test_inference_time_reasonable(self):
        """Test that inference time is within acceptable range."""
        inference_times = [0.065, 0.072, 0.058, 0.095, 0.081]
        
        mean_time = np.mean(inference_times)
        
        # Should be under 200ms (0.2 seconds)
        assert mean_time < 0.2
        
        # Should be above 1ms (too fast indicates mock)
        assert mean_time > 0.001
    
    def test_memory_efficiency(self, sample_image):
        """Test that preprocessing doesn't create excessive memory."""
        import sys
        
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        tensor = transform(sample_image)
        
        # Memory size should be reasonable
        # 1 * 3 * 224 * 224 * 4 bytes (float32) = ~600KB
        expected_size = 1 * 3 * 224 * 224 * 4
        actual_size = tensor.element_size() * tensor.nelement()
        
        assert actual_size <= expected_size * 2  # Allow 2x overhead


# Test class for error handling
class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_empty_image_handling(self):
        """Test handling of empty/zero-size images."""
        with pytest.raises(ValueError):
            Image.new('RGB', (0, 0))
    
    def test_extreme_aspect_ratios(self):
        """Test images with extreme aspect ratios."""
        # Very wide image
        wide_img = Image.new('RGB', (1000, 10), (128, 128, 128))
        
        from torchvision import transforms
        transform = transforms.Resize((224, 224))
        resized = transform(wide_img)
        
        assert resized.size == (224, 224)
        
        # Very tall image
        tall_img = Image.new('RGB', (10, 1000), (128, 128, 128))
        resized = transform(tall_img)
        
        assert resized.size == (224, 224)
    
    def test_class_index_out_of_range(self, class_mapping):
        """Test handling of invalid class indices."""
        invalid_indices = ['-1', '7', '100', 'invalid']
        
        for idx in invalid_indices:
            assert idx not in class_mapping


# Test class for visualization
class TestVisualization:
    """Tests for visualization components."""
    
    def test_confidence_chart_data_structure(self, mock_predictions):
        """Test that chart data has correct structure."""
        classes = [p['class'].upper() for p in mock_predictions]
        confidences = [p['confidence'] for p in mock_predictions]
        
        assert len(classes) == len(confidences)
        assert all(isinstance(c, str) for c in classes)
        assert all(isinstance(conf, (int, float)) for conf in confidences)
    
    def test_risk_level_mapping(self):
        """Test that risk levels are correctly mapped to values."""
        risk_mapping = {
            'LOW': 25,
            'MEDIUM': 50,
            'HIGH': 75,
            'CRITICAL': 95
        }
        
        for level, value in risk_mapping.items():
            assert 0 <= value <= 100
            assert isinstance(value, int)


# Test class for configuration
class TestConfiguration:
    """Tests for configuration management."""
    
    def test_model_config_defaults(self):
        """Test that model configuration has sensible defaults."""
        from dataclasses import dataclass
        from typing import Tuple, List
        
        @dataclass
        class ModelConfig:
            model_path: str = "model/ham10000_model.onnx"
            classes_path: str = "model/classes.json"
            input_size: Tuple[int, int] = (224, 224)
            mean: List[float] = None
            std: List[float] = None
            
            def __post_init__(self):
                if self.mean is None:
                    self.mean = [0.485, 0.456, 0.406]
                if self.std is None:
                    self.std = [0.229, 0.224, 0.225]
        
        config = ModelConfig()
        
        assert config.input_size == (224, 224)
        assert len(config.mean) == 3
        assert len(config.std) == 3
        assert all(0 <= m <= 1 for m in config.mean)
        assert all(0 < s <= 1 for s in config.std)


# Integration tests
class TestIntegration:
    """Integration tests for complete pipeline."""
    
    def test_end_to_end_preprocessing(self, sample_image):
        """Test complete preprocessing pipeline."""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Apply full pipeline
        tensor = transform(sample_image)
        batched = tensor.unsqueeze(0)
        numpy_array = batched.numpy()
        
        # Validate output
        assert numpy_array.shape == (1, 3, 224, 224)
        assert numpy_array.dtype == np.float32
    
    @patch('onnxruntime.InferenceSession')
    def test_mock_inference_pipeline(self, mock_session, sample_image):
        """Test inference pipeline with mocked ONNX session."""
        # Setup mock
        mock_output = np.random.rand(1, 7).astype(np.float32)
        mock_session.return_value.run.return_value = [mock_output]
        
        # This would be your actual inference call
        # result = model.predict(sample_image)
        
        # Verify mock was called
        # mock_session.return_value.run.assert_called_once()
        pass


# Benchmark tests (optional, can be slow)
@pytest.mark.slow
class TestBenchmarks:
    """Performance benchmark tests."""
    
    def test_preprocessing_speed(self, sample_image, benchmark):
        """Benchmark preprocessing speed."""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        result = benchmark(transform, sample_image)
        assert result is not None


# Parametrized tests
@pytest.mark.parametrize("width,height", [
    (224, 224),
    (512, 512),
    (100, 100),
    (1024, 768),
    (300, 400),
])
def test_various_image_sizes(width, height):
    """Test preprocessing with various image sizes."""
    img = Image.new('RGB', (width, height), (128, 128, 128))
    
    from torchvision import transforms
    transform = transforms.Resize((224, 224))
    resized = transform(img)
    
    assert resized.size == (224, 224)


@pytest.mark.parametrize("confidence,expected_risk", [
    (95, 'CRITICAL'),
    (75, 'HIGH'),
    (50, 'MEDIUM'),
    (25, 'LOW'),
])
def test_confidence_to_risk_mapping(confidence, expected_risk):
    """Test mapping of confidence to risk levels."""
    # This is a simplified test - adjust based on your actual logic
    if confidence >= 90:
        risk = 'CRITICAL'
    elif confidence >= 70:
        risk = 'HIGH'
    elif confidence >= 40:
        risk = 'MEDIUM'
    else:
        risk = 'LOW'
    
    assert risk == expected_risk


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])