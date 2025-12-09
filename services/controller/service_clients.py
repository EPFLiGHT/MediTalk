"""
Service Clients: HTTP communication with model services.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class ServiceClient:
    """Base HTTP client for communicating with model services."""
    
    def __init__(
        self,
        base_url: str,
        service_name: str,
        timeout: float = 30.0, # seconds
        max_retries: int = 2,
        verbose: bool = False
    ):
        self.base_url = base_url.rstrip('/')
        self.service_name = service_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.verbose = verbose
        
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True
        )
        
        if self.verbose:
            logger.info(f"Initialized {service_name} client: {base_url}")
    
    async def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        POST request to service endpoint.
        
        Args:
            endpoint: Endpoint path (e.g., "/generate")
            data: JSON payload
        
        Returns:
            Response JSON as dict
        
        Raises:
            httpx.HTTPError: On request failure after retries
        """
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                
                response = await self.client.post(
                    url,
                    json=data,
                    **kwargs
                )
                
                response.raise_for_status()
                
                elapsed_ms = (time.time() - start_time) * 1000
                
                if self.verbose:
                    logger.info(
                        f"{self.service_name} {endpoint} succeeded "
                        f"in {elapsed_ms:.0f}ms (attempt {attempt + 1})"
                    )
                elif elapsed_ms > 1000:
                    logger.warning(
                        f"{self.service_name} {endpoint} slow response: {elapsed_ms:.0f}ms"
                    )
                
                return response.json()
            
            except httpx.HTTPError as e:
                # retry logic
                is_last_attempt = (attempt == self.max_retries)
                
                if is_last_attempt:
                    logger.error(
                        f"{self.service_name} {endpoint} failed after "
                        f"{self.max_retries + 1} attempts: {e}"
                    )
                    raise
                else:
                    logger.warning(
                        f"{self.service_name} {endpoint} failed "
                        f"(attempt {attempt + 1}), retrying: {e}"
                    )
                    await asyncio.sleep(1.0)
    
    async def get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """GET request to service endpoint"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = await self.client.get(url, **kwargs)
            response.raise_for_status()
            return response.json()
        
        except httpx.HTTPError as e:
            logger.error(f"{self.service_name} GET {endpoint} failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if service is healthy.
        
        Returns:
            {"status": "healthy" | "unhealthy", "response_time_ms": float, ...}
        """
        try:
            start_time = time.time()
            response = await self.client.get(
                f"{self.base_url}/health",
                timeout=5.0
            )
            elapsed_ms = (time.time() - start_time) * 1000
            
            health_status = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time_ms": elapsed_ms,
                "status_code": response.status_code
            }
            
            if self.verbose:
                logger.info(
                    f"{self.service_name} health check: {health_status['status']} "
                    f"({elapsed_ms:.0f}ms)"
                )
            
            return health_status
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


class ServiceRegistry:
    """Registry of all model services."""
    
    def __init__(self, config: Dict[str, str], verbose: bool = False):
        """
        Initialize service registry.
        
        Args:
            config: Dict mapping service name to base URL
                Example: {
                    "multimeditron": "http://localhost:5000",
                    "whisper": "http://localhost:5007",
                    ...
                }
        """
        self.clients: Dict[str, ServiceClient] = {}
        self.verbose = verbose
        
        for service_name, base_url in config.items():
            self.clients[service_name] = ServiceClient(
                base_url=base_url,
                service_name=service_name,
                verbose=verbose
            )
        
        if verbose:
            logger.info(f"Initialized {len(self.clients)} service clients: {list(self.clients.keys())}")
        else:
            logger.info(f"Initialized {len(self.clients)} service clients")
    
    def get(self, service_name: str) -> Optional[ServiceClient]:
        """Get client for a service"""
        return self.clients.get(service_name)
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all registered services"""
        tasks = {
            name: client.health_check()
            for name, client in self.clients.items()
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        return {
            name: result if not isinstance(result, Exception) else {
                "status": "error",
                "error": str(result)
            }
            for name, result in zip(tasks.keys(), results)
        }
    
    async def close_all(self):
        """Close all HTTP clients"""
        await asyncio.gather(*[
            client.close() 
            for client in self.clients.values()
        ])


# ============================================================================
# Helper functions for common service interactions
# ============================================================================


async def call_multimeditron(
    client: ServiceClient,
    conversation_json_path: str,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Call MultiMeditron service."""
    if verbose:
        logger.info(f"Calling MultiMeditron with conversation file: {conversation_json_path}")
    
    return await client.post(
        "/generate",
        data={
            "conversation_path": conversation_json_path,
            **kwargs
        }
    )


async def call_stt(
    client: ServiceClient,
    audio_path: str,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Call Speech-to-Text service (Whisper only for now)."""
    if verbose:
        logger.info(f"Calling STT with audio: {audio_path}")
    

    return await client.post(
        "/transcribe",
        audio=audio_path,
        **kwargs
    )


async def call_tts(
    client: ServiceClient,
    target_text: str,
    conversation_json_path: Optional[str] = None,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Call Text-to-Speech service (Orpheus, Bark, CSM, Qwen3-Omni)."""
    if verbose:
        logger.info(f"Calling TTS with target_text (length={len(target_text)}): {target_text[:50]}...")
        if conversation_json_path:
            logger.info(f"  Conversation file: {conversation_json_path}")
    
    # Build data payload - conversation_path is now the primary method
    data_payload = {
        "conversation_path": conversation_json_path,
        **kwargs
    }
    
    # Keep target_text for backward compatibility with services not yet updated
    if not conversation_json_path and target_text:
        data_payload["text"] = target_text
    
    return await client.post(
        "/synthesize",
        data=data_payload
    )
