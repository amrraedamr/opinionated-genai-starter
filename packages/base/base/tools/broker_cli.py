import argparse
import asyncio

from base.services.broker.redis_streams import RedisStreamsBroker


def main() -> None:
    parser = argparse.ArgumentParser(description="Broker operational commands")
    subcommands = parser.add_subparsers(dest="command", required=True)
    replay = subcommands.add_parser("replay-dlq")
    replay.add_argument("topic")
    replay.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    async def run() -> None:
        broker = RedisStreamsBroker()
        try:
            count = await broker.areplay_dlq(args.topic, limit=args.limit)
            print(f"replayed={count}")
        finally:
            await broker.astop()

    asyncio.run(run())


if __name__ == "__main__":
    main()
