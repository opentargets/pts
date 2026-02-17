"""Core module for the PTS application."""

import asyncio

from otter import Runner


def main() -> None:
    """Main entry point for the PTS application."""
    runner = Runner(name='pts')
    runner.start()
    runner.register_tasks('pts.tasks')
    asyncio.run(runner.run())
