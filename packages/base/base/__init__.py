"""Opinionated FastAPI GenAI foundation library.

`base` is a library, not an application. It has no entrypoint of its own —
a service composes one with `base.app.create_app` and serves it with
`base.tools.webserver.run_webserver`.
"""
