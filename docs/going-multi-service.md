# Outgrowing one app

This starter is one deployable. One database, one `.env`, one dependency set,
one thing to release. That is the right shape far longer than people expect,
and most products never need anything else.

It stops being the right shape when the single process starts carrying
conflicting requirements.

## The signs

- **Release cadence splits.** The chat path ships daily; billing ships when
  finance says so. One pipeline cannot serve both without one of them waiting.
- **Scaling splits.** Ingestion needs memory and no traffic; the API needs
  traffic and little memory. One container means paying for the maximum of
  both, everywhere.
- **Dependencies conflict.** One part wants a library another cannot take, or
  a worker drags the whole web stack into its image because they share a
  manifest.
- **Ownership splits.** Two teams merging into one `Settings` class, one
  migration history and one test suite is a queue, not a workflow.
- **Blast radius matters.** A bad deploy to a reporting endpoint should not be
  able to take down login.

One of these is usually worth absorbing. Three at once is the signal.

## What the other repo does differently

[opinionated-genai-microservices-starter](https://github.com/amrraedamr/opinionated-genai-microservices-starter)
keeps everything here and changes the shape around it: a uv workspace with a
shared `base` package plus independently deployable services.

- **Each service owns its database.** No shared tables, no foreign key across a
  boundary. A reference to another service's row is an opaque id.
- **Each service owns its settings.** `CoreSettings` holds only what every
  service genuinely shares; everything else is a mixin a service opts into, so
  a worker has no signing key and the chat API cannot mint tokens.
- **Each service owns its dependencies.** Extras are declared per service, so a
  worker's image contains no web server.
- **Identity is its own service.** Users, registration, OAuth and token issuing
  live in one place; everything else only verifies.
- **Services talk through events.** A versioned contracts package and a broker
  abstraction, with Redis Streams behind it.
- **The boundaries are enforced.** A lint fails CI when base imports a service,
  a service imports another, or a foreign key crosses a boundary.

## Moving across

You do not have to port by hand. That repository tracks this one and ships a
tool that rewrites a commit's paths into the shared package or the owning
service, and tells you what needs a human instead of guessing:

```bash
uv run python tools/port_from_starter.py <sha> --check
```

So a feature can land here and follow across, rather than the two drifting.

## The honest trade

Splitting buys independence and costs you a network between things that used
to be a function call: partial failure, eventual consistency, more to run
locally, and a deploy story per service.

If you can name which of the signs above you are hitting, it is worth it. If
you cannot, stay here.
