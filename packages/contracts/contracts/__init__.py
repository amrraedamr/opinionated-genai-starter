"""Shared payloads and event schemas.

Depends on pydantic only, never on base or a service, so any package can
import it without pulling in infrastructure.

Versioning: the namespace is versioned, not the class. Within a version only
additive changes are allowed; a breaking change goes in the next version and
both are served until consumers migrate.
"""
