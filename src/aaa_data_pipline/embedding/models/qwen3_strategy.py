# -*- coding: utf-8 -*-
"""
qwen3_strategy.py

This module implements the embedding strategy for Qwen3-Embedding models.
Qwen3-Embedding models are compatible with sentence-transformers but require
special handling to extract token-level embeddings for chunking methods.
"""

import time
import logging
import numpy as np
import torch
from typing import Optional, List
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from .base_strategy import AbstractEmbeddingStrategy

logger = logging.getLogger(__name__)


class Qwen3EmbeddingStrategy(AbstractEmbeddingStrategy):
    """
    Embedding strategy implementation for Qwen3-Embedding models.
    
    This strategy uses sentence-transformers for loading and encoding,
    but also provides token-level embeddings (colbert_vecs) by accessing
    the underlying transformer model's hidden states.
    """
    
    def __init__(self, model_name: str, target_devices: list[str]):
        """
        Initialize the Qwen3 embedding strategy.
        
        Args:
            model_name (str): The name of the model to load (e.g., 'Qwen/Qwen3-Embedding-4B').
            target_devices (list[str]): List of CUDA device identifiers.
        """
        super().__init__(model_name, target_devices)
        self.model: Optional[SentenceTransformer] = None
        self.pool = None
        self._enable_token_embeddings = True  # Enable token-level embeddings by default
        # For token extraction, we'll use transformers directly
        self._transformer_model = None
        self._transformer_tokenizer = None
    
    def set_token_embeddings_enabled(self, enabled: bool):
        """
        Enable or disable token-level embedding extraction.
        
        This can be used to save memory and time for methods that only need dense embeddings.
        
        Args:
            enabled (bool): If True, extract token-level embeddings. If False, only extract dense embeddings.
        """
        self._enable_token_embeddings = enabled
        if not enabled:
            # Clean up transformer model if it exists
            self.cleanup_transformer_model()
    
    def load_model(self) -> None:
        """
        Load the Qwen3-Embedding model using sentence-transformers.
        
        Simple single GPU usage only.
        """
        logger.info(f"Loading Qwen3-Embedding model: {self.model_name}")
        try:
            # Qwen3-Embedding models work with sentence-transformers
            # Requires transformers>=4.51.0 and sentence-transformers>=2.7.0
            
            # Always use single GPU (first device only)
            primary_device = self.target_devices[0] if self.target_devices else 'cuda:0'
            
            if primary_device.startswith('cuda'):
                logger.info(f"Loading model on {primary_device} (single GPU only)")
                self.model = SentenceTransformer(self.model_name, device=primary_device)
            else:
                logger.info("Loading model on CPU")
                self.model = SentenceTransformer(self.model_name, device='cpu')
            
            if hasattr(self.model, 'get_max_seq_length'):
                model_max_len = self.model.get_max_seq_length()
                logger.info(f"Model max sequence length: {model_max_len} tokens")
            
            logger.info("Qwen3-Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Qwen3-Embedding model: {e}")
            raise
    
    def _extract_token_embeddings(self, sentences: List[str], batch_size: int) -> List[np.ndarray]:
        """
        Extract token-level embeddings (hidden states) from the model.
        
        This method uses transformers library directly to avoid device mismatch issues
        with sentence-transformers wrapper.
        
        Args:
            sentences (List[str]): List of sentences to encode.
            batch_size (int): Batch size for encoding.
        
        Returns:
            List[np.ndarray]: List of token-level embeddings, one per sentence.
                            Each array has shape [seq_len, hidden_dim].
        """
        token_embeddings = []
        
        # Determine the device
        device = self.target_devices[0] if self.target_devices else 'cpu'
        
        # Try to get tokenizer from sentence-transformers model first
        tokenizer = None
        if hasattr(self.model, 'tokenizer'):
            tokenizer = self.model.tokenizer
        
        # If not available, load it
        if tokenizer is None:
            try:
                self._transformer_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                tokenizer = self._transformer_tokenizer
            except Exception as e:
                logger.error(f"Failed to load tokenizer: {e}")
                return []
        
        # Get the underlying transformer model from sentence-transformers
        # sentence-transformers wraps the model, we need to access the actual AutoModel
        transformer_model = None
        
        # Try different ways to access the underlying model
        if hasattr(self.model, '_modules'):
            modules_list = list(self.model._modules.values())
            # The first module might be a wrapper, try to get the actual model
            for module in modules_list:
                if hasattr(module, 'auto_model'):
                    transformer_model = module.auto_model
                    break
                elif hasattr(module, 'model'):
                    transformer_model = module.model
                    break
                # Sometimes the module itself is the model
                elif hasattr(module, 'forward') and hasattr(module, 'config'):
                    transformer_model = module
                    break
        
        # If still not found, try direct attributes
        if transformer_model is None:
            if hasattr(self.model, 'auto_model'):
                transformer_model = self.model.auto_model
            elif hasattr(self.model, 'model'):
                transformer_model = self.model.model
        
        # If we still can't get it, try to reuse the sentence-transformers model's underlying model
        # Only load a separate model if absolutely necessary (to save memory)
        if transformer_model is None:
            # Last resort: load directly (uses more memory but works)
            if self._transformer_model is None:
                try:
                    logger.warning(f"Could not access model from sentence-transformers. Loading separate model for token extraction (this uses extra memory)")
                    self._transformer_model = AutoModel.from_pretrained(self.model_name)
                    self._transformer_model = self._transformer_model.to(device)
                    self._transformer_model.eval()
                except Exception as e:
                    logger.error(f"Failed to load transformer model directly: {e}")
                    return []
            transformer_model = self._transformer_model
        else:
            # Use the model from sentence-transformers, ensure it's on the right device
            # Don't move it if it's already on a GPU (to avoid memory issues)
            try:
                current_device = None
                if hasattr(transformer_model, 'device'):
                    current_device = str(transformer_model.device)
                elif hasattr(transformer_model, 'parameters'):
                    first_param = next(transformer_model.parameters(), None)
                    if first_param is not None:
                        current_device = str(first_param.device)
                
                # Only move if not already on a CUDA device
                if current_device and not current_device.startswith('cuda'):
                    transformer_model = transformer_model.to(device)
                elif not current_device:
                    transformer_model = transformer_model.to(device)
                transformer_model.eval()
            except Exception as e:
                logger.warning(f"Could not move model to device: {e}")
        
        # Get embedding dimension from model config for consistent empty array shapes
        emb_dim = 1024  # Default fallback
        try:
            if transformer_model is not None and hasattr(transformer_model, 'config'):
                emb_dim = getattr(transformer_model.config, 'hidden_size', 1024)
                logger.debug(f"Using embedding dimension: {emb_dim}")
        except Exception as e:
            logger.debug(f"Could not get embedding dimension from model config: {e}")
        
        # Process in batches
        # Use larger batches for better GPU utilization
        total_batches = (len(sentences) + batch_size - 1) // batch_size
        logger.info(f"Extracting token embeddings for {len(sentences)} sentences in {total_batches} batches of {batch_size}")
        
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            # Log progress for large batches
            if total_batches > 10 and batch_num % max(1, total_batches // 10) == 0:
                logger.info(f"  Token extraction progress: {batch_num}/{total_batches} batches ({batch_num * 100 // total_batches}%)")
            
            # Tokenize
            try:
                encoded = tokenizer(
                    batch_sentences,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=32768  # Qwen3 supports 32k
                )
            except Exception as e:
                logger.warning(f"Error tokenizing batch {i}: {e}")
                # Add empty arrays with correct shape (using emb_dim from above)
                for _ in batch_sentences:
                    token_embeddings.append(np.zeros((0, emb_dim)))  # Consistent shape
                continue
            
            # Move all tensors to device
            encoded = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in encoded.items()}
            
            # Get hidden states
            with torch.no_grad():
                try:
                    outputs = transformer_model(**encoded, output_hidden_states=True)
                    
                    # Get the last hidden state
                    if hasattr(outputs, 'last_hidden_state'):
                        hidden_states = outputs.last_hidden_state
                    elif hasattr(outputs, 'hidden_states') and len(outputs.hidden_states) > 0:
                        hidden_states = outputs.hidden_states[-1]
                    elif isinstance(outputs, tuple) and len(outputs) > 0:
                        hidden_states = outputs[0]
                    else:
                        logger.warning("Could not extract hidden states from model output")
                        # Add empty arrays as fallback
                        for _ in batch_sentences:
                            token_embeddings.append(np.array([]))
                        continue
                    
                    # Convert to numpy and store
                    # Remove padding tokens (where attention_mask is 0)
                    if isinstance(encoded, dict) and 'attention_mask' in encoded:
                        attention_mask = encoded['attention_mask']
                        # Ensure attention_mask is on same device as hidden_states
                        if attention_mask.device != hidden_states.device:
                            attention_mask = attention_mask.to(hidden_states.device)
                        
                        # Process on GPU, then move to CPU for numpy conversion
                        for j in range(hidden_states.shape[0]):
                            # Get mask for this sample (keep on GPU for computation)
                            mask = attention_mask[j]  # This is on the same device as hidden_states
                            # Get actual sequence length (excluding padding) - compute on GPU
                            seq_len = int(mask.sum().item())
                            if seq_len > 0:
                                # Slice on GPU, then move to CPU
                                token_emb = hidden_states[j, :seq_len].cpu().numpy()
                                token_embeddings.append(token_emb)
                            else:
                                # Empty sequence - create zero vector with correct dimension
                                emb_dim = hidden_states.shape[-1]
                                token_embeddings.append(np.zeros((0, emb_dim)))
                        
                        # Clear intermediate tensors to free memory
                        del attention_mask, hidden_states, outputs
                    else:
                        # No attention mask, use all tokens
                        for j in range(hidden_states.shape[0]):
                            token_emb = hidden_states[j].cpu().numpy()
                            token_embeddings.append(token_emb)
                        
                        # Clear intermediate tensors
                        del hidden_states, outputs
                    
                    # Clear encoded tensors
                    del encoded
                    
                    # Clear CUDA cache after each batch to prevent OOM
                    # When using DataParallel, be even more aggressive
                    if device.startswith('cuda'):
                        torch.cuda.empty_cache()
                        # Every few batches, do a more thorough cleanup
                        if (i // batch_size) % 10 == 0:
                            torch.cuda.synchronize()
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError as e:
                    logger.warning(f"CUDA OOM error extracting token embeddings for batch {i}: {e}")
                    # Add empty arrays as fallback with correct shape (using emb_dim from above)
                    for _ in batch_sentences:
                        token_embeddings.append(np.zeros((0, emb_dim)))  # Consistent shape
                    
                    # Aggressively clear CUDA cache after OOM error
                    if device.startswith('cuda'):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        # Try to free up more memory
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Error extracting token embeddings for batch {i}: {e}")
                    # Add empty arrays as fallback with correct shape (using emb_dim from above)
                    for _ in batch_sentences:
                        token_embeddings.append(np.zeros((0, emb_dim)))  # Consistent shape
                    
                    # Clear CUDA cache after error to free memory
                    if device.startswith('cuda'):
                        torch.cuda.empty_cache()
        
        return token_embeddings
    
    def encode(self, corpus: list[str], batch_size: int) -> dict[str, np.ndarray]:
        """
        Encode a corpus of documents using Qwen3-Embedding.
        
        This method provides both dense embeddings (for CLSToken, ChunkFirstEmbed)
        and token-level embeddings (for MeanPooling, LateChunking).
        
        Args:
            corpus (list[str]): List of documents to encode.
            batch_size (int): Batch size for encoding.
        
        Returns:
            dict[str, np.ndarray]: Dictionary with 'dense' and optionally 'colbert_vecs' keys.
        """
        if not corpus:
            logger.warning("Input corpus is empty. Returning empty dictionary.")
            return {'dense': np.array([])}
        
        if not self.target_devices:
            logger.error("No target devices specified for parallel encoding.")
            raise ValueError("target_devices list cannot be empty.")
        
        logger.info(f"Starting Qwen3-Embedding encoding on {len(self.target_devices)} devices")
        
        # Check available GPU memory and adjust batch size if needed
        if self.target_devices and self.target_devices[0].startswith('cuda'):
            try:
                import torch
                if torch.cuda.is_available():
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0) - torch.cuda.memory_reserved(0)
                    free_gb = free_memory / (1024**3)
                    
                    # If less than 2GB free, reduce batch size
                    if free_gb < 2.0 and batch_size > 8:
                        old_batch_size = batch_size
                        batch_size = max(8, batch_size // 2)
                        logger.warning(f"Low GPU memory ({free_gb:.2f} GB free). Reducing batch size from {old_batch_size} to {batch_size}")
            except Exception as e:
                logger.debug(f"Could not check GPU memory: {e}")
        
        logger.info(f"Processing {len(corpus)} documents with batch size {batch_size}")
        
        try:
            # Determine device for encoding - simple single GPU
            device = self.target_devices[0] if self.target_devices else 'cpu'
            
            # For token-level embeddings, we need to use the model directly
            if self._enable_token_embeddings:
                logger.info("Encoding with token-level extraction enabled...")
                
                # Simple single GPU encoding
                logger.info(f"Using direct encoding on {device}")
                start_time = time.time()
                dense_embeddings = self.model.encode(
                    sentences=corpus,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                
                # Extract token-level embeddings - simple single GPU
                # Use smaller batch size for token extraction to avoid OOM
                token_batch_size = batch_size  # Start with same batch size
                if device.startswith('cuda'):
                    try:
                        import torch
                        # Check memory on GPU
                        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0) - torch.cuda.memory_reserved(0)
                        free_gb = free_memory / (1024**3)
                        
                        # Adjust batch size based on available memory
                        if free_gb < 4.0:
                            token_batch_size = max(1, min(batch_size, 4))  # Very small batches
                            logger.warning(f"Low GPU memory ({free_gb:.2f} GB free). Using small token extraction batch size: {token_batch_size}")
                        elif free_gb < 8.0:
                            token_batch_size = max(2, min(batch_size, 8))  # Small batches
                            logger.info(f"Moderate GPU memory ({free_gb:.2f} GB free). Using token extraction batch size: {token_batch_size}")
                        else:
                            token_batch_size = max(batch_size, 4)  # Can use larger batches
                    except Exception as e:
                        logger.debug(f"Could not check GPU memory for token extraction: {e}")
                        token_batch_size = max(1, batch_size // 2)
                
                logger.info(f"Extracting token-level embeddings with batch size {token_batch_size} (this requires a second forward pass on first GPU)...")
                token_start = time.time()
                colbert_vecs = self._extract_token_embeddings(corpus, token_batch_size)
                token_time = time.time() - token_start
                logger.info(f"Token extraction took {token_time:.2f} seconds ({len(corpus)/token_time:.1f} docs/sec)")
                
                # Clear the separate transformer model after token extraction to free memory
                self.cleanup_transformer_model()
                
                end_time = time.time()
                duration = end_time - start_time
                docs_per_sec = len(corpus) / duration if duration > 0 else 0
                
                logger.info(f"Encoding complete. Time taken: {duration:.2f} seconds")
                logger.info(f"Throughput: {docs_per_sec:.2f} docs/sec")
                
                result = {'dense': dense_embeddings}
                if colbert_vecs and len(colbert_vecs) == len(corpus):
                    # Check if any embeddings are non-empty
                    non_empty = sum(1 for emb in colbert_vecs if isinstance(emb, np.ndarray) and emb.size > 0)
                    if non_empty > 0:
                        result['colbert_vecs'] = colbert_vecs
                        logger.info(f"Extracted token-level embeddings for {non_empty}/{len(corpus)} documents")
                    else:
                        logger.warning(f"Token-level extraction returned empty arrays for all documents. Falling back to dense-only mode.")
                else:
                    logger.warning(f"Token-level extraction incomplete: got {len(colbert_vecs) if colbert_vecs else 0} embeddings for {len(corpus)} documents")
                
                return result
            else:
                # Dense-only encoding - simple single GPU
                logger.info(f"Using direct encoding on {device}")
                start_time = time.time()
                dense_embeddings = self.model.encode(
                    sentences=corpus,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                
                end_time = time.time()
                duration = end_time - start_time
                docs_per_sec = len(corpus) / duration if duration > 0 else 0
                
                logger.info(f"Dense encoding complete. Time taken: {duration:.2f} seconds")
                logger.info(f"Throughput: {docs_per_sec:.2f} docs/sec")
                
                return {'dense': dense_embeddings}
        
        except Exception as e:
            logger.error(f"An error occurred during Qwen3-Embedding encoding: {e}")
            raise
        finally:
            # Always ensure the pool is stopped
            if self.pool:
                try:
                    logger.info("Stopping multi-process pool...")
                    self.model.stop_multi_process_pool(self.pool)
                    self.pool = None
                    logger.info("Pool stopped.")
                except Exception as e:
                    logger.error(f"Error stopping multiprocessing pool: {e}")
                    self.pool = None
                
                # Clear CUDA cache after token extraction
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
    
    def cleanup_transformer_model(self):
        """Free the separate transformer model if it was loaded."""
        if self._transformer_model is not None:
            del self._transformer_model
            self._transformer_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Cleaned up separate transformer model")
    
