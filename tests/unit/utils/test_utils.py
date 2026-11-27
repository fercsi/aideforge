import pytest
from datetime import datetime, timezone
from uuid import UUID

from aideforge.utils import deep_merge, remove_nones, create_id, id_to_timestamp, IdType


class TestDeepMerge:
    def test_basic_merge(self):
        primary = {"a": 1, "b": 2}
        secondary = {"b": 3, "c": 4}
        result = deep_merge(primary, secondary)
        assert result == {"a": 1, "b": 2, "c": 4}

    def test_primary_takes_precedence(self):
        primary = {"a": 1, "b": 2}
        secondary = {"a": 99, "c": 4}
        result = deep_merge(primary, secondary)
        assert result == {"a": 1, "b": 2, "c": 4}

    def test_nested_dict_merge(self):
        primary = {"a": {"x": 1}, "b": 2}
        secondary = {"a": {"y": 3}, "c": 4}
        result = deep_merge(primary, secondary)
        assert result == {"a": {"x": 1, "y": 3}, "b": 2, "c": 4}

    def test_nested_dict_primary_precedence(self):
        primary = {"a": {"x": 1, "z": 99}}
        secondary = {"a": {"x": 2, "y": 3}}
        result = deep_merge(primary, secondary)
        assert result == {"a": {"x": 1, "z": 99, "y": 3}}

    def test_list_override(self):
        primary = {"a": [1, 2]}
        secondary = {"a": [3, 4], "b": 5}
        result = deep_merge(primary, secondary)
        assert result == {"a": [1, 2], "b": 5}

    def test_none_values(self):
        primary = {"a": None}
        secondary = {"a": 1, "b": 2}
        result = deep_merge(primary, secondary)
        assert result == {"a": None, "b": 2}

    def test_empty_primary(self):
        primary = {}
        secondary = {"a": 1, "b": 2}
        result = deep_merge(primary, secondary)
        assert result == {"a": 1, "b": 2}

    def test_empty_secondary(self):
        primary = {"a": 1, "b": 2}
        secondary = {}
        result = deep_merge(primary, secondary)
        assert result == {"a": 1, "b": 2}

    def test_both_empty(self):
        result = deep_merge({}, {})
        assert result == {}

    def test_deep_nesting(self):
        primary = {"a": {"b": {"c": 1}}}
        secondary = {"a": {"b": {"d": 2}, "e": 3}, "f": 4}
        result = deep_merge(primary, secondary)
        assert result == {"a": {"b": {"c": 1, "d": 2}, "e": 3}, "f": 4}

    def test_original_dicts_unchanged(self):
        primary = {"a": 1}
        secondary = {"b": 2}
        deep_merge(primary, secondary)
        assert primary == {"a": 1}
        assert secondary == {"b": 2}


class TestRemoveNones:
    def test_remove_none_from_dict(self):
        data = {"a": 1, "b": None, "c": 3}
        result = remove_nones(data)
        assert result == {"a": 1, "c": 3}

    def test_remove_none_from_list(self):
        data = [1, None, 3, None, 5]
        result = remove_nones(data)
        assert result == [1, 3, 5]

    def test_nested_dict(self):
        data = {"a": {"b": None, "c": 2}, "d": None}
        result = remove_nones(data)
        assert result == {"a": {"c": 2}}

    def test_nested_list(self):
        data = [{"a": None, "b": 1}, [None, 2], None]
        result = remove_nones(data)
        assert result == [{"b": 1}, [2]]

    def test_no_nones(self):
        data = {"a": 1, "b": 2}
        result = remove_nones(data)
        assert result == {"a": 1, "b": 2}

    def test_all_nones_dict(self):
        data = {"a": None, "b": None}
        result = remove_nones(data)
        assert result == {}

    def test_all_nones_list(self):
        data = [None, None]
        result = remove_nones(data)
        assert result == []

    def test_non_container(self):
        data = "string"
        result = remove_nones(data)
        assert result == "string"

    def test_integer(self):
        data = 42
        result = remove_nones(data)
        assert result == 42

    def test_empty_dict(self):
        data = {}
        result = remove_nones(data)
        assert result == {}

    def test_empty_list(self):
        data = []
        result = remove_nones(data)
        assert result == []

    def test_mixed_structure(self):
        data = {
            "keep": "value",
            "remove": None,
            "nested": {
                "keep": 1,
                "remove": None,
                "list": [1, None, 3]
            },
            "list": [{"a": None}, {"b": 2}]
        }
        result = remove_nones(data)
        expected = {
            "keep": "value",
            "nested": {
                "keep": 1,
                "list": [1, 3]
            },
            "list": [{}, {"b": 2}]
        }
        assert result == expected


class TestCreateId:
    def test_returns_uuid(self):
        id_val = create_id()
        assert isinstance(id_val, UUID)

    def test_returns_uuid7(self):
        id_val = create_id()
        assert id_val.version == 7

    def test_unique_ids(self):
        ids = [create_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_with_data_type(self):
        id_val = create_id("event")
        assert isinstance(id_val, UUID)

    def test_with_none_data_type(self):
        id_val = create_id(None)
        assert isinstance(id_val, UUID)


class TestIdToTimestamp:
    def test_returns_utc_datetime(self):
        id_val = create_id()
        ts = id_to_timestamp(id_val)
        assert isinstance(ts, datetime)
        assert ts.tzinfo == timezone.utc

    def test_timestamp_is_recent(self):
        id_val = create_id()
        ts = id_to_timestamp(id_val)
        now = datetime.now(timezone.utc)
        diff = abs((now - ts).total_seconds())
        assert diff < 10