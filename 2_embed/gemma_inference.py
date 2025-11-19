"""
vLLM Inference for Google Embedding Gemma 300M
Provides efficient batch embedding generation using vLLM
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List
import numpy as np

class EmbeddingGemmaModel:
    """Wrapper for Google Embedding Gemma 300M with vLLM-style batching"""
    
    def __init__(self, model_name: str = "google/embeddinggemma-300m", batch_size: int = 32):
        """
        Initialize Embedding Gemma model
        
        Args:
            model_name: HuggingFace model identifier
            batch_size: Number of texts to process at once
        """
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading {model_name}...")
        print(f"Device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        print(f"✅ Model loaded on {self.device}")
        print(f"   Batch size: {batch_size}")
    
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of texts to embeddings
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of embeddings (batch_size, embedding_dim)
        """
        # Tokenize
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Mean pooling
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convert to numpy
        return embeddings.cpu().numpy()
    
    def encode(self, texts: List[str], show_progress: bool = True) -> List[np.ndarray]:
        """
        Encode texts with automatic batching
        
        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar
            
        Returns:
            List of embeddings (one per input text)
        """
        all_embeddings = []
        
        # Process in batches
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(texts), self.batch_size), desc="Encoding batches")
        else:
            iterator = range(0, len(texts), self.batch_size)
        
        for i in iterator:
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.encode_batch(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings"""
        # Test with dummy input
        test_embedding = self.encode_batch(["test"])
        return test_embedding.shape[1]


# Singleton instance
_model_instance = None


def get_embedding_model(batch_size: int = 32) -> EmbeddingGemmaModel:
    """
    Get or create embedding model instance (singleton pattern)
    
    Args:
        batch_size: Batch size for processing
        
    Returns:
        EmbeddingGemmaModel instance
    """
    global _model_instance
    
    if _model_instance is None:
        _model_instance = EmbeddingGemmaModel(batch_size=batch_size)
    
    return _model_instance


def embed_texts(texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
    """
    Convenience function to embed texts
    
    Args:
        texts: List of text strings
        batch_size: Batch size for processing
        
    Returns:
        List of embeddings
    """
    model = get_embedding_model(batch_size=batch_size)
    return model.encode(texts, show_progress=False)


if __name__ == "__main__":
    # Test the model
    print("="*70)
    print("Testing Embedding Gemma Model")
    print("="*70)
    
    # Initialize model
    model = get_embedding_model(batch_size=8)
    
    # Test texts
    test_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand text."
    ]
    
    print(f"\n🧪 Testing with {len(test_texts)} texts...")
    embeddings = model.encode(test_texts)
    
    print(f"\n✅ Generated {len(embeddings)} embeddings")
    print(f"   Embedding dimension: {embeddings[0].shape[0]}")
    print(f"   Shape of first embedding: {embeddings[0].shape}")
    
    # Test similarity
    print(f"\n📊 Cosine similarities:")
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            similarity = np.dot(embeddings[i], embeddings[j])
            print(f"   Text {i+1} <-> Text {j+1}: {similarity:.4f}")
    
    print("\n✅ Model test complete!")