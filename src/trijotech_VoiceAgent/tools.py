"""Tools for Trija, Trijotech's sales voice agent.

* ``search_company_knowledge`` - RAG over the Supabase vector table.
* ``talk_to_human``            - mock transfer; records a row in ``handoff_requests``.
* ``check_appointment``        - free 30-minute slots (12:00 PM - 5:00 PM) on a day.
* ``schedule_appointment``     - books a slot in ``appointments``.

Database access is synchronous psycopg run through ``asyncio.to_thread`` (see
``db.py``). Tools return plain strings the model can paraphrase aloud; failures
come back as a message instead of an exception so the call never crashes.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import functools
import os
from zoneinfo import ZoneInfo

from langchain.tools import tool
from loguru import logger

from . import db
from .knowledge import get_vector_store

SLOT_MINUTES = 30
OPEN_HOUR = 12   # first slot starts at 12:00
CLOSE_HOUR = 17  # last slot ends at 17:00 (so it starts at 16:30)

TOOL_FAILED = (
    "That action failed because of a technical problem. Apologise briefly and "
    "offer to connect the customer with a human representative."
)


@functools.lru_cache(maxsize=1)
def get_tz() -> ZoneInfo:
    """Appointment timezone (``APPOINTMENT_TIMEZONE``, default Asia/Kolkata)."""
    return ZoneInfo(os.getenv("APPOINTMENT_TIMEZONE", "Asia/Kolkata"))


# --- Helpers -----------------------------------------------------------------


def _fmt_time(t: dt.datetime) -> str:
    return t.strftime("%I:%M %p").lstrip("0")


def _fmt_day(d: dt.date) -> str:
    return f"{d:%A}, {d.day} {d:%B %Y}"


def _slots_for(day: dt.date) -> list[dt.datetime]:
    tz = get_tz()
    cur = dt.datetime.combine(day, dt.time(OPEN_HOUR), tzinfo=tz)
    end = dt.datetime.combine(day, dt.time(CLOSE_HOUR), tzinfo=tz)
    slots = []
    while cur < end:
        slots.append(cur)
        cur += dt.timedelta(minutes=SLOT_MINUTES)
    return slots


def _free_slots(day: dt.date) -> list[dt.datetime]:
    """Slots on ``day`` that are not booked and not already in the past (blocking)."""
    slots = _slots_for(day)
    rows = db.query(
        "SELECT start_time FROM appointments "
        "WHERE start_time >= %(lo)s AND start_time < %(hi)s AND status <> 'cancelled'",
        {"lo": slots[0], "hi": slots[-1] + dt.timedelta(minutes=SLOT_MINUTES)},
    )
    booked = {r["start_time"] for r in rows}
    now = dt.datetime.now(get_tz())
    return [s for s in slots if s not in booked and s > now]


# --- Tools -------------------------------------------------------------------


@tool
async def search_company_knowledge(query: str) -> str:
    """Search Trijotech's knowledge base (services, products, SAP expertise, process).

    Use this before answering any question about what Trijotech offers. Pass a
    short, specific search query in plain English, e.g. "SAP S/4HANA migration
    services". Answer only from what it returns.
    """
    try:
        store = await get_vector_store()
        docs = await store.asimilarity_search(query, k=4)
    except Exception:  # noqa: BLE001
        logger.exception("search_company_knowledge failed")
        return TOOL_FAILED
    if not docs:
        return "Nothing relevant was found in the knowledge base for that query."
    return "\n\n---\n\n".join(d.page_content[:1500] for d in docs)


@tool
async def talk_to_human(
    customer_name: str,
    company_name: str,
    contact_details: str,
    customer_need: str,
    escalation_reason: str,
    conversation_summary: str,
) -> str:
    """Transfer the customer to a human Trijotech representative (mock transfer).

    Use when you cannot answer the customer's question after searching, or when
    the customer asks to speak to a person. Before calling, make sure you have
    the customer's name, company name and contact details (phone or email); ask
    for anything missing and never invent values.
    """
    try:
        rows = await asyncio.to_thread(
            db.query,
            "INSERT INTO handoff_requests "
            "(id, customer_name, company_name, contact_details, customer_need, "
            " escalation_reason, conversation_summary, status, created_at) "
            "VALUES (gen_random_uuid(), %(name)s, %(company)s, %(contact)s, %(need)s, "
            "        %(reason)s, %(summary)s, 'pending', now()) "
            "RETURNING id",
            {
                "name": customer_name,
                "company": company_name,
                "contact": contact_details,
                "need": customer_need,
                "reason": escalation_reason,
                "summary": conversation_summary,
            },
        )
    except Exception:  # noqa: BLE001
        logger.exception("talk_to_human failed")
        return TOOL_FAILED
    logger.info(f"MOCK TRANSFER to human: {customer_name} ({company_name}) - {escalation_reason}")
    return (
        f"Transfer requested (reference {str(rows[0]['id'])[:8]}). Tell the customer a "
        f"Trijotech representative will contact them shortly on {contact_details}."
    )


@tool
async def check_appointment(date: str) -> str:
    """Check which 30-minute appointment slots are free on a given day.

    Slots run from 12:00 PM to 5:00 PM. `date` must be YYYY-MM-DD; work it out
    from today's date when the customer says "tomorrow", "next Monday", etc.
    """
    try:
        day = dt.date.fromisoformat(date.strip())
    except ValueError:
        return "That date is not valid. Use the format YYYY-MM-DD."
    if day < dt.datetime.now(get_tz()).date():
        return f"{_fmt_day(day)} is in the past. Ask the customer for a date from today onwards."
    try:
        free = await asyncio.to_thread(_free_slots, day)
    except Exception:  # noqa: BLE001
        logger.exception("check_appointment failed")
        return TOOL_FAILED
    if not free:
        return f"There are no free slots on {_fmt_day(day)}. Offer another day."
    times = ", ".join(_fmt_time(s) for s in free)
    return f"Free 30-minute slots on {_fmt_day(day)} ({get_tz().key}): {times}."


_INSERT_APPOINTMENT = """
INSERT INTO appointments
    (id, customer_name, company_name, contact_details, customer_need,
     conversation_summary, start_time, end_time, timezone, status, created_at)
SELECT gen_random_uuid(), %(name)s::text, %(company)s::text, %(contact)s::text,
       %(need)s::text, %(summary)s::text, %(start)s::timestamptz, %(end)s::timestamptz,
       %(tz)s::text, 'confirmed', now()
WHERE NOT EXISTS (
    SELECT 1 FROM appointments
    WHERE start_time = %(start)s::timestamptz AND status <> 'cancelled'
)
RETURNING id
"""


@tool
async def schedule_appointment(
    customer_name: str,
    company_name: str,
    contact_details: str,
    customer_need: str,
    conversation_summary: str,
    date: str,
    start_time: str,
) -> str:
    """Book a 30-minute appointment for the customer.

    Call only after check_appointment showed the slot is free AND the customer
    has confirmed the date, time and contact details. `date` is YYYY-MM-DD and
    `start_time` is 24-hour HH:MM (e.g. "14:30"). Never invent customer details.
    """
    try:
        day = dt.date.fromisoformat(date.strip())
        hh, mm = (int(p) for p in start_time.strip().split(":"))
        start = dt.datetime.combine(day, dt.time(hh, mm), tzinfo=get_tz())
    except ValueError:
        return "The date or time is not valid. Use YYYY-MM-DD and 24-hour HH:MM."
    if start not in _slots_for(day):
        return "That is not a bookable time. Slots start every 30 minutes from 12:00 PM to 4:30 PM."
    if start <= dt.datetime.now(get_tz()):
        return "That time has already passed. Ask for a later slot."

    try:
        rows = await asyncio.to_thread(
            db.query,
            _INSERT_APPOINTMENT,
            {
                "name": customer_name,
                "company": company_name,
                "contact": contact_details,
                "need": customer_need,
                "summary": conversation_summary,
                "start": start,
                "end": start + dt.timedelta(minutes=SLOT_MINUTES),
                "tz": get_tz().key,
            },
        )
    except Exception:  # noqa: BLE001
        logger.exception("schedule_appointment failed")
        return TOOL_FAILED
    if not rows:
        return "That slot was just taken. Apologise, call check_appointment again and offer other times."
    return (
        f"Appointment confirmed for {_fmt_day(day)} at {_fmt_time(start)} "
        f"({get_tz().key}) for {customer_name} from {company_name}."
    )