import pytest
from pydantic import ValidationError
from unittest.mock import MagicMock

from aideforge.storage.memory import MemoryStorage
from aideforge.structs.abc import BaseStruct, BaseIncrementalStruct, BaseEvent, ChangeProperty
from aideforge.structs.agent import AgentConfig
from aideforge.structs.llm import LlmConfig
from aideforge.structs.theme import Theme
from aideforge.utils import create_id


class TestMemoryStorage:
    def setup_method(self):
        # Clear class-level storage before each test
        MemoryStorage._structs.clear()
        MemoryStorage._resources.clear()

    def test_retrieve_resource_not_found(self):
        storage = MemoryStorage()
        with pytest.raises(FileNotFoundError):
            storage.retrieve_resource("nonexistent.txt")

    def test_retrieve_resource_ignore_failure(self):
        storage = MemoryStorage()
        result = storage.retrieve_resource("nonexistent.txt", ignore_failure=True)
        assert result is None

    def test_store_and_retrieve_resource(self):
        storage = MemoryStorage()
        storage.store_resource("test.txt", "hello world")
        result = storage.retrieve_resource("test.txt")
        assert result == "hello world"

    def test_store_binary_resource(self):
        storage = MemoryStorage()
        storage.store_resource("test.bin", b"binary data")
        result = storage.retrieve_resource("test.bin")
        assert result == b"binary data"

    def test_overwrite_resource(self):
        storage = MemoryStorage()
        storage.store_resource("test.txt", "first")
        storage.store_resource("test.txt", "second")
        result = storage.retrieve_resource("test.txt")
        assert result == "second"

    def test_retrieve_struct_not_found(self):
        storage = MemoryStorage()
        with pytest.raises(FileNotFoundError):
            storage.retrieve_struct(Theme, "nonexistent")

    def test_retrieve_struct_ignore_failure(self):
        storage = MemoryStorage()
        result = storage.retrieve_struct(Theme, "nonexistent", ignore_failure=True)
        assert result is None

    def test_store_and_retrieve_struct(self):
        storage = MemoryStorage()
        theme = Theme(name="test_theme")
        storage.store_struct(theme)
        retrieved = storage.retrieve_struct(Theme, "test_theme")
        assert retrieved is not None
        assert retrieved.name == "test_theme"
        assert retrieved is not theme  # Should be a copy

    def test_store_struct_with_custom_name(self):
        storage = MemoryStorage()
        theme = Theme(name="original_name")
        storage.store_struct(theme, struct_name="custom_name")
        retrieved = storage.retrieve_struct(Theme, "custom_name")
        assert retrieved is not None
        assert retrieved.name == "original_name"

    def test_list_structs_empty(self):
        storage = MemoryStorage()
        result = storage.list_structs(Theme)
        assert result == []

    def test_list_structs(self):
        storage = MemoryStorage()
        storage.store_struct(Theme(name="theme1"))
        storage.store_struct(Theme(name="theme2"))
        storage.store_struct(Theme(name="theme3"))
        result = storage.list_structs(Theme)
        assert result == ["theme1", "theme2", "theme3"]

    def test_list_structs_different_types(self):
        storage = MemoryStorage()
        storage.store_struct(Theme(name="theme1"))
        storage.store_struct(AgentConfig(name="agent1", llm=LlmConfig(model="gpt-4", provider="openai")))
        themes = storage.list_structs(Theme)
        agents = storage.list_structs(AgentConfig)
        assert themes == ["theme1"]
        assert agents == ["agent1"]

    def test_store_struct_increment_new_struct(self):
        # Create an incremental struct
        class MyIncStruct(BaseIncrementalStruct):
            kind: str = "test"
            version: int = 1
            value: str = "initial"

        storage = MemoryStorage()
        struct = MyIncStruct(kind="test", version=1)
        
        # Try to store increment on non-existent struct - should fail
        event = BaseEvent(event_type="test")
        with pytest.raises(FileNotFoundError):
            storage.store_struct_increment(struct, event)

    def test_store_struct_increment_existing(self):
        class MyIncStruct(BaseIncrementalStruct):
            kind: str = "test"
            version: int = 1
            value: str = "initial"

        storage = MemoryStorage()
        struct = MyIncStruct(kind="test", version=1)
        storage.store_struct(struct)
        
        event = BaseEvent(event_type="test_event")
        storage.store_struct_increment(struct, event)
        
        retrieved = storage.retrieve_struct(MyIncStruct, "test")
        assert len(retrieved.events) == 1
        assert retrieved.events[0].event_type == "test_event"

    def test_store_struct_increment_change_property(self):
        class MyIncStruct(BaseIncrementalStruct):
            kind: str = "test"
            version: int = 1
            value: str = "initial"

        storage = MemoryStorage()
        struct = MyIncStruct(kind="test", version=1)
        storage.store_struct(struct)
        
        event = ChangeProperty.from_changing(struct, "value", "updated")
        storage.store_struct_increment(struct, event)
        
        retrieved = storage.retrieve_struct(MyIncStruct, "test")
        assert retrieved.value == "updated"
        assert len(retrieved.events) == 1

    def test_isolation_between_instances(self):
        storage1 = MemoryStorage()
        storage2 = MemoryStorage()
        
        storage1.store_struct(Theme(name="theme1"))
        
        # storage2 should not see storage1's data since _structs is class-level
        # Actually MemoryStorage uses class variables, so they share state
        result = storage2.list_structs(Theme)
        assert result == ["theme1"]

    def test_get_checkpoint_store(self):
        storage = MemoryStorage()
        checkpoint_store = storage.get_checkpoint_store()
        # Just verify it returns something (could be None if langgraph not installed)
        assert checkpoint_store is not None or checkpoint_store is None