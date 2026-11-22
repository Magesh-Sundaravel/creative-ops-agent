import json

from mcp_server.server import SEED_DIR

NEW_TASK = {
    "title": "Spring Linen Edit: Pinterest pins",
    "brand": "loomhouse",
    "campaign_name": "Spring Linen Edit",
    "description": "Three 2:3 pins from the feed shoot.",
}


async def test_create_task_returns_task_with_next_id_after_seed_and_status_briefed(connect):
    async with connect() as client:
        result = await client.call_tool(
            "tracker_create_task", {**NEW_TASK, "due_date": "2031-04-01"}
        )

    assert not result.is_error
    assert result.structured_content == {
        "id": "TSK-004",
        **NEW_TASK,
        "due_date": "2031-04-01",
        "status": "briefed",
    }


async def test_create_task_ids_are_sequential_and_due_date_is_optional(connect):
    async with connect() as client:
        first = await client.call_tool("tracker_create_task", NEW_TASK)
        second = await client.call_tool("tracker_create_task", NEW_TASK)

    assert first.structured_content["id"] == "TSK-004"
    assert first.structured_content["due_date"] is None
    assert second.structured_content["id"] == "TSK-005"


async def test_create_task_caller_cannot_set_starting_status(connect):
    async with connect() as client:
        tools = await client.list_tools()
        result = await client.call_tool("tracker_create_task", {**NEW_TASK, "status": "delivered"})

    [tool] = [t for t in tools.tools if t.name == "tracker_create_task"]
    assert "status" not in tool.input_schema["properties"]
    assert "status" in tool.output_schema["properties"]
    assert result.structured_content["status"] == "briefed"


async def test_create_task_unknown_brand_returns_not_found_tool_error(connect):
    async with connect() as client:
        result = await client.call_tool("tracker_create_task", {**NEW_TASK, "brand": "loomhous"})

    assert result.is_error
    error = json.loads(result.content[0].text)
    assert error["type"] == "not_found"
    assert "loomhous" in error["message"]
    assert error["details"] == {"brand": "loomhous"}


async def test_update_status_returns_updated_task(connect):
    async with connect() as client:
        result = await client.call_tool(
            "tracker_update_status", {"task_id": "TSK-002", "status": "revisions_requested"}
        )

    assert not result.is_error
    assert result.structured_content == {
        "id": "TSK-002",
        "title": "City Miles: TikTok launch cut",
        "brand": "stridewell",
        "campaign_name": "City Miles",
        "description": "15-second vertical video for the City Miles running shoe launch.",
        "due_date": "2031-03-01",
        "status": "revisions_requested",
    }


async def test_update_status_any_status_may_follow_any_other_on_a_created_task(connect):
    async with connect() as client:
        created = await client.call_tool("tracker_create_task", NEW_TASK)
        task_id = created.structured_content["id"]
        delivered = await client.call_tool(
            "tracker_update_status", {"task_id": task_id, "status": "delivered"}
        )
        reopened = await client.call_tool(
            "tracker_update_status", {"task_id": task_id, "status": "briefed"}
        )

    assert delivered.structured_content["status"] == "delivered"
    assert reopened.structured_content["status"] == "briefed"


async def test_update_status_unknown_task_id_returns_not_found_tool_error(connect):
    async with connect() as client:
        result = await client.call_tool(
            "tracker_update_status", {"task_id": "TSK-999", "status": "approved"}
        )

    assert result.is_error
    error = json.loads(result.content[0].text)
    assert error["type"] == "not_found"
    assert "TSK-999" in error["message"]
    assert error["details"] == {"task_id": "TSK-999"}


async def test_update_status_invalid_status_returns_invalid_input_listing_valid_statuses(connect):
    async with connect() as client:
        result = await client.call_tool(
            "tracker_update_status", {"task_id": "TSK-001", "status": "done"}
        )

    assert result.is_error
    error = json.loads(result.content[0].text)
    assert error["type"] == "invalid_input"
    assert "'done'" in error["message"]
    valid = [
        "briefed",
        "in_production",
        "in_review",
        "revisions_requested",
        "approved",
        "delivered",
    ]
    assert all(status in error["message"] for status in valid)
    assert error["details"] == {"status": "done", "valid_statuses": valid}


async def test_writes_stay_in_one_server_and_never_touch_seed_files(connect):
    seed_before = (SEED_DIR / "tasks.json").read_bytes()

    async with connect() as client:
        await client.call_tool("tracker_create_task", NEW_TASK)
        await client.call_tool("tracker_create_task", NEW_TASK)  # TSK-005
        await client.call_tool(
            "tracker_update_status", {"task_id": "TSK-001", "status": "delivered"}
        )

    async with connect() as fresh:
        unknown = await fresh.call_tool(
            "tracker_update_status", {"task_id": "TSK-005", "status": "in_review"}
        )
        created = await fresh.call_tool("tracker_create_task", NEW_TASK)

    assert unknown.is_error
    assert created.structured_content["id"] == "TSK-004"
    assert (SEED_DIR / "tasks.json").read_bytes() == seed_before


async def test_create_task_brand_ignores_case_and_is_stored_as_slug(connect):
    async with connect() as client:
        result = await client.call_tool("tracker_create_task", {**NEW_TASK, "brand": "LoomHouse"})

    assert not result.is_error
    assert result.structured_content["brand"] == "loomhouse"


async def test_update_status_ignores_case(connect):
    async with connect() as client:
        result = await client.call_tool(
            "tracker_update_status", {"task_id": "TSK-001", "status": "Approved"}
        )

    assert not result.is_error
    assert result.structured_content["status"] == "approved"
