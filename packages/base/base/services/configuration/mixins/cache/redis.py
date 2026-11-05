from pydantic import BaseModel


class RedisSettingsMixin(BaseModel):
    redis_url: str = ""
    redis_ssl_cert_reqs: str = ""
    redis_max_connections: int = 50
    redis_socket_timeout: float = 5.0
    redis_socket_connect_timeout: float = 5.0
    redis_retry_on_timeout: bool = True
    redis_health_check_interval: int = 30
    redis_decode_responses: bool = True
