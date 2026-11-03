#!/bin/sh

set -e

SERVICE="${SERVICE:?SERVICE must be set}"
VENV_BIN="${VENV_BIN:-/app/.venv/bin}"

cd "/app/services/$SERVICE"

# Only SQL services carry migrations. Prefer a pre-deploy job over this flag
# when running more than one replica.
if [ -n "$RUN_MIGRATIONS" ] && [ -f alembic.ini ]; then
    echo "Running migrations for $SERVICE"
    "$VENV_BIN/alembic" upgrade head
fi

# Run from the service directory so its own .env and ../../.env.shared resolve.
echo "Starting $SERVICE"
exec "$VENV_BIN/serve"
