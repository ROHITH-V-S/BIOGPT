import time
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from app.logging_config import request_id_var

logger = logging.getLogger(__name__)

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = str(uuid.uuid4())
        token = request_id_var.set(req_id)
        
        logger.info(f"Request started: {request.method} {request.url.path}")
        start_time = time.monotonic()
        
        try:
            response = await call_next(request)
            process_time = time.monotonic() - start_time
            response.headers["X-Request-ID"] = req_id
            logger.info(f"Request completed: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s")
            return response
        except Exception as e:
            process_time = time.monotonic() - start_time
            logger.exception(f"Request failed: {request.method} {request.url.path} - Time: {process_time:.4f}s")
            raise
        finally:
            request_id_var.reset(token)
