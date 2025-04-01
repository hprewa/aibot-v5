"""
Schema cache implementation with in-memory caching and optional GCS persistence.
Focuses on fast in-memory access with graceful fallback if GCS is not available.
"""

import json
import os
import time
from typing import Dict, Any, Optional
from google.cloud import storage
from threading import Lock

class SchemaCache:
    # Class-level variables for in-memory caching
    _memory_cache: Dict[str, Any] = {}
    _last_refresh: float = 0
    _refresh_interval: int = 300  # 5 minutes
    _lock = Lock()
    _gcs_enabled = False  # Track if GCS is available
    
    def __init__(self):
        self.bucket_name = "text-to-sql-dev.appspot.com"
        self.blob_name = "schema_cache/cache.json"
        self._storage_client = None
        self._bucket = None
        
    @property
    def storage_client(self):
        """Lazy initialization of storage client"""
        if self._storage_client is None and self._gcs_enabled:
            try:
                self._storage_client = storage.Client()
                print(f"[GCS] Initialized client for project: {self._storage_client.project}")
            except Exception as e:
                print(f"[GCS] Storage client initialization failed, falling back to memory-only cache")
                self._gcs_enabled = False
        return self._storage_client
        
    @property
    def bucket(self):
        """Lazy initialization of bucket"""
        if self._bucket is None and self._gcs_enabled:
            try:
                self._bucket = self.storage_client.bucket(self.bucket_name)
                # Test bucket access
                if not self._bucket.exists():
                    print(f"[GCS] Bucket {self.bucket_name} not found, falling back to memory-only cache")
                    self._gcs_enabled = False
                else:
                    print(f"[GCS] Using bucket: {self.bucket_name}")
            except Exception as e:
                print(f"[GCS] Bucket access failed, falling back to memory-only cache")
                self._gcs_enabled = False
        return self._bucket if self._gcs_enabled else None
    
    @classmethod
    def load(cls) -> Dict[str, Any]:
        """Load schema cache with two-level caching strategy"""
        current_time = time.time()
        
        # Check if memory cache is fresh enough
        with cls._lock:
            if current_time - cls._last_refresh < cls._refresh_interval:
                print("[CACHE] Using in-memory schema cache")
                return cls._memory_cache
        
        # Try to load from Cloud Storage if enabled
        if cls._gcs_enabled:
            try:
                instance = cls()
                if instance.bucket:
                    blob = instance.bucket.blob(instance.blob_name)
                    if blob.exists():
                        cache_data = json.loads(blob.download_as_string())
                        print(f"[GCS] Loaded cache with {len(cache_data)} entries")
                        
                        # Update memory cache
                        with cls._lock:
                            cls._memory_cache = cache_data
                            cls._last_refresh = current_time
                        
                        return cache_data
            except Exception as e:
                print(f"[GCS] Cache load failed, using memory cache")
                cls._gcs_enabled = False
        
        # Return memory cache or initialize new one
        with cls._lock:
            if cls._memory_cache:
                print("[CACHE] Using existing memory cache")
                return cls._memory_cache
            return cls._init_empty_cache()
    
    @classmethod
    def save(cls, schemas: Dict[str, Any]) -> bool:
        """Save schema cache to memory and optionally to Cloud Storage"""
        # Always update memory cache
        with cls._lock:
            cls._memory_cache = schemas.copy()
            cls._last_refresh = time.time()
            print(f"[CACHE] Updated memory cache with {len(schemas)} entries")
        
        # Try to save to Cloud Storage if enabled
        if cls._gcs_enabled:
            try:
                instance = cls()
                if instance.bucket:
                    blob = instance.bucket.blob(instance.blob_name)
                    cache_data = json.dumps(schemas, indent=2)
                    blob.upload_from_string(
                        cache_data,
                        content_type='application/json',
                        timeout=30
                    )
                    print(f"[GCS] Saved cache with {len(schemas)} entries")
                    return True
            except Exception as e:
                print(f"[GCS] Cache save failed, operating in memory-only mode")
                cls._gcs_enabled = False
        
        return True  # Memory cache was updated successfully
    
    @classmethod
    def _init_empty_cache(cls) -> Dict[str, Any]:
        """Initialize an empty cache"""
        with cls._lock:
            cls._memory_cache = {}
            cls._last_refresh = time.time()
            print("[CACHE] Initialized empty cache")
            return cls._memory_cache
            
    @classmethod
    def clear_memory_cache(cls):
        """Clear the in-memory cache"""
        with cls._lock:
            cls._memory_cache = {}
            cls._last_refresh = 0
            print("[CACHE] Cleared memory cache") 