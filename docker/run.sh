#!/bin/sh

set -e

SERVICE="${SERVICE:?SERVICE must be set}"
VENV_BIN="${VENV_BIN:-/app/.venv/bin}"

cd "/app/services/$SERVICE"

# An image supports only the backends its extras installed, but SQL_BACKEND and
# BROKER are set long after the build. Fail here, with the supported values
# named, rather than inside a client on the first call.
"$VENV_BIN/python" -m base.preflight

# Only SQL services carry migrations. Prefer a pre-deploy job over this flag
# when running more than one replica.
if [ -n "$RUN_MIGRATIONS" ] && [ -f alembic.ini ]; then
    echo "Running migrations for $SERVICE"
    "$VENV_BIN/alembic" upgrade head
fi

# Run from the service directory so its own .env and ../../.env.shared resolve.
echo "Starting $SERVICE"
exec "$VENV_BIN/serve"
