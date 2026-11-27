import pytest
from datetime import datetime
from pydantic import ValidationError

from aideforge.structs.abc import (
    BaseStruct,
    BaseEvent,
    BaseIncrementalStruct,
    ChangeProperty,
    PropertyChangeType,
    StructProxy,
)
from aideforge.utils import create_id, IdType


class TestBaseStruct:
    def test_creation_with_required_fields(self):
        class MyStruct(BaseStruct):
            kind: str = "test"
            version: int = 1

        s = MyStruct(kind="test", version=1)
        assert s.kind == "test"
        assert s.version == 1

    def test_missing_kind_raises(self):
        class MyStruct(BaseStruct):
            pass

        with pytest.raises(ValidationError):
            MyStruct(version=1)

    def test_missing_version_raises(self):
        class MyStruct(BaseStruct):
            pass

        with pytest.raises(ValidationError):
            MyStruct(kind="test")

    def test_class_metadata(self):
        class MyStruct(BaseStruct):
            kind: str = "test"
            version: int = 1

        assert MyStruct.scope == "user"
        assert MyStruct.storage_name == "{struct.kind}"
        assert MyStruct.storage_mode == "snapshot"
        assert MyStruct.kinds is None
        assert MyStruct.type_fields == ["kind", "version"]

    def test_created_at_raises_without_id(self):
        class MyStruct(BaseStruct):
            kind: str = "test"
            version: int = 1

        s = MyStruct(kind="test", version=1)
        with pytest.raises(AttributeError, match="does not support.*created_at"):
            _ = s.created_at

    def test_created_at_with_event_id(self):
        class MyStruct(BaseStruct):
            kind: str = "test"
            version: int = 1
            test_id: IdType = create_id("event")

        s = MyStruct(kind="test", version=1, test_id=create_id("event"))
        assert isinstance(s.created_at, datetime)


class TestBaseEvent:
    def test_creation(self):
        event = BaseEvent(event_type="test_event")
        assert event.kind == "event"
        assert event.version == 1
        assert event.event_type == "test_event"
        assert event.event_id is not None
        assert event.event_id.version == 7

    def test_metadata(self):
        assert BaseEvent.storage_name == "{struct.event_id}"
        assert BaseEvent.kinds == "events"

    def test_missing_event_type_raises(self):
        with pytest.raises(ValidationError):
            BaseEvent()


class TestBaseIncrementalStruct:
    def test_creation(self):
        class MyIncStruct(BaseIncrementalStruct):
            kind: str = "test"
            version: int = 1

        s = MyIncStruct(kind="test", version=1)
        assert s.events == []
        assert MyIncStruct.storage_mode == "incremental"

    def test_add_event(self):
        class MyIncStruct(BaseIncrementalStruct):
            kind: str = "test"
            version: int = 1

        s = MyIncStruct(kind="test", version=1)
        event = BaseEvent(event_type="test")
        s.events.append(event)
        assert len(s.events) == 1
        assert s.events[0].event_type == "test"


class TestPropertyChangeType:
    def test_valid_types(self):
        assert "replace" in PropertyChangeType.__args__
        assert "append" in PropertyChangeType.__args__
        assert "extend" in PropertyChangeType.__args__
        assert "update" in PropertyChangeType.__args__
        assert "add" in PropertyChangeType.__args__


class TestChangeProperty:
    def test_creation(self):
        event = ChangeProperty(
            property_name="test_prop",
            new_value="new_value",
            change_type="replace"
        )
        assert event.event_type == "change-property"
        assert event.change_type == "replace"
        assert event.property_name == "test_prop"
        assert event.old_value is None
        assert event.new_value == "new_value"

    def test_change_replace(self):
        class MyIncStruct(BaseIncrementalStruct):
            kind: str = "test"
            version: int = 1
            test_prop: str = "old"

        s = MyIncStruct(kind="test", version=1)
        event = ChangeProperty(
            property_name="test_prop",
            new_value="new",
            change_type="replace"
        )
        event.change(s)
        assert s.test_prop == "new"

    def test_change_append(self):
        class MyIncStruct(BaseIncrementalStruct):
            kind: str = "test"
            version: int = 1
            test_list: list[str] = ["a", "b"]

        s = MyIncStruct(kind="test", version=1)
        event = ChangeProperty(
            property_name="test_list",
            new_value="c",
            change_type="append"
        )
        event.change(s)
        assert s.test_list == ["a", "b", "c"]

    def test_change_extend(self):
        class MyIncStruct(BaseIncrementalStruct):
            kind: str = "test"
            version: int = 1
            test_list: list[str] = ["a", "b"]

        s = MyIncStruct(kind="test", version=1)
        event = ChangeProperty(
            property_name="test_list",
            new_value=["c", "d"],
            change_type="extend"
        )
        event.change(s)
        assert s.test_list == ["a", "b", "c", "d"]

    def test_change_update(self):
        class MyIncStruct(BaseIncrementalStruct):
            kind: str = "test"
            version: int = 1
            test_dict: dict[str, str] = {"a": "1"}

        s = MyIncStruct(kind="test", version=1)
        event = ChangeProperty(
            property_name="test_dict",
            new_value={"b": "2"},
            change_type="update"
        )
        event.change(s)
        assert s.test_dict == {"a": "1", "b": "2"}

    def test_change_add(self):
        class MyIncStruct(BaseIncrementalStruct):
            kind: str = "test"
            version: int = 1
            test_set: set[str] = {"a", "b"}

        s = MyIncStruct(kind="test", version=1)
        event = ChangeProperty(
            property_name="test_set",
            new_value="c",
            change_type="add"
        )
        event.change(s)
        assert s.test_set == {"a", "b", "c"}

    def test_change_invalid_type_raises(self):
        class MyIncStruct(BaseIncrementalStruct):
            kind: str = "test"
            version: int = 1
            test_prop: str = "old"

        s = MyIncStruct(kind="test", version=1)
        with pytest.raises(ValidationError):
            ChangeProperty(
                property_name="test_prop",
                new_value="new",
                change_type="invalid"  # type: ignore
            )

    def test_from_changing(self):
        class MyIncStruct(BaseIncrementalStruct):
            kind: str = "test"
            version: int = 1
            test_prop: str = "old"

        s = MyIncStruct(kind="test", version=1)
        event = ChangeProperty.from_changing(s, "test_prop", "new", change_type="replace")
        assert event.property_name == "test_prop"
        assert event.old_value == "old"
        assert event.new_value == "new"
        assert event.change_type == "replace"
        assert s.test_prop == "new"
        assert len(s.events) == 1
        assert s.events[0] is event

    def test_from_changing_missing_property_raises(self):
        class MyIncStruct(BaseIncrementalStruct):
            kind: str = "test"
            version: int = 1

        s = MyIncStruct(kind="test", version=1)
        with pytest.raises(AttributeError):
            ChangeProperty.from_changing(s, "nonexistent", "new")


class TestStructProxy:
    def test_getattr_class_meta(self):
        class MyStruct(BaseStruct):
            kind: str = "test"
            version: int = 1

        proxy = StructProxy(MyStruct)
        assert proxy.kind == "test"
        assert proxy.version == 1
        assert proxy.scope == "user"
        assert proxy.storage_name == "{struct.kind}"

    def test_getattr_field_default(self):
        class MyStruct(BaseStruct):
            kind: str = "test"
            version: int = 1
            name: str = "default_name"

        proxy = StructProxy(MyStruct)
        assert proxy.name == "default_name"

    def test_getattr_missing_raises(self):
        class MyStruct(BaseStruct):
            kind: str = "test"
            version: int = 1

        proxy = StructProxy(MyStruct)
        with pytest.raises(AttributeError, match="has no attribute 'missing'"):
            _ = proxy.missing

    def test_getattr_none_default(self):
        class MyStruct(BaseStruct):
            kind: str = "test"
            version: int = 1
            optional_field: str | None = None

        proxy = StructProxy(MyStruct)
        assert proxy.optional_field is None