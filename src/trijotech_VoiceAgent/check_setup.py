"""Smoke test for Trija's backend. Run it BEFORE starting the voice bot:

    uv run python -m gym_support.check_setup

Checks, in order: Postgres connection + checkpointer, the knowledge search
(embedding model must match your stored vectors), and appointment availability.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv(override=True)

from . import db, tools  # noqa: E402


async def main() -> None:
    print("1) Postgres + checkpointer ...")
    await asyncio.to_thread(db.get_checkpointer)
    rows = await asyncio.to_thread(db.query, "select count(*) as n from company_documents")
    print(f"   OK - company_documents has {rows[0]['n']} rows")

    print("2) Knowledge search ...")
    out = await tools.search_company_knowledge.ainvoke({"query": "What SAP services do you offer?"})
    print("  ", out[:600].replace("\n", " "))

    print("3) Appointment availability ...")
    tomorrow = (datetime.now(tools.get_tz()) + timedelta(days=1)).date().isoformat()
    print("  ", await tools.check_appointment.ainvoke({"date": tomorrow}))

    print("4) Schedule an appointment ...")
    out = await tools.schedule_appointment.ainvoke(
        {
            "customer_name": "Alice Smith",
            "company_name": "Acme Corp",
            "contact_details": "123-456-7890",
            "customer_need": "SAP S/4HANA migration services",
            "conversation_summary": "Alice is interested in our SAP S/4HANA migration services.",
            "date": tomorrow,
            "start_time": "14:30",
        }
    )
    print("  ", out)

    print("3) Appointment availability ...")
    tomorrow = (datetime.now(tools.get_tz()) + timedelta(days=1)).date().isoformat()
    print("  ", await tools.check_appointment.ainvoke({"date": tomorrow}))


if __name__ == "__main__":
    asyncio.run(main())