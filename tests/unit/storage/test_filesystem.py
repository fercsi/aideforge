import pytest
import tempfile
import os
from pathlib import Path

from aideforge.storage.filesystem import FileSystemStorage
from aideforge.structs.abc import BaseStruct, BaseIncrementalStruct, BaseEvent, ChangeProperty
from aideforge.structs.agent import AgentConfig
from aideforge.structs.llm import LlmConfig
from aideforge.structs.theme import Theme
from aideforge.utils import create_id


class TestFileSystemStorage:
    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage = FileSystemStorage(struct_paths=[self.temp_dir.name], struct_format="json")

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_creation(self):
        assert self.storage.struct_paths == [self.temp_dir.name]
        assert self.storage.struct_format == "json"

    def test_creation_with_multiple_paths(self):
        with tempfile.TemporaryDirectory() as dir2:
            storage = FileSystemStorage(struct_paths=[self.temp_dir.name, dir2], struct_format="yaml")
            assert len(storage.struct_paths) == 2

    def test_retrieve_resource_not_found(self):
        with pytest.raises(FileNotFoundError):
            self.storage.retrieve_resource("nonexistent.txt")

    def test_retrieve_resource_ignore_failure(self):
        result = self.storage.retrieve_resource("nonexistent.json", ignore_failure=True)
        assert result is None

    def test_store_and_retrieve_resource(self):
        self.storage.store_resource("test.json", "hello world")
        result = self.storage.retrieve_resource("test.json")
        assert result == "hello world"

    def test_store_binary_resource(self):
        self.storage.store_resource("test.json", b"binary data")
        result = self.storage.retrieve_resource("test.json", mode="b")
        assert result == b"binary data"

    def test_overwrite_resource(self):
        self.storage.store_resource("test.json", "first")
        self.storage.store_resource("test.json", "second")
        result = self.storage.retrieve_resource("test.json")
        assert result == "second"

    def test_retrieve_struct_not_found(self):
        with pytest.raises(FileNotFoundError):
            self.storage.retrieve_struct(Theme, "nonexistent")

    def test_retrieve_struct_ignore_failure(self):
        result = self.storage.retrieve_struct(Theme, "nonexistent", ignore_failure=True)
        assert result is None

    def test_store_and_retrieve_struct(self):
        theme = Theme(name="test_theme")
        self.storage.store_struct(theme)
        retrieved = self.storage.retrieve_struct(Theme, "test_theme")
        assert retrieved is not None
        assert retrieved.name == "test_theme"

    def test_store_struct_with_custom_name(self):
        theme = Theme(name="original_name")
        self.storage.store_struct(theme, struct_name="custom_name")
        retrieved = self.storage.retrieve_struct(Theme, "custom_name")
        assert retrieved is not None
        assert retrieved.name == "original_name"

    def test_list_structs_empty(self):
        result = self.storage.list_structs(Theme)
        assert result == []

    def test_list_structs(self):
        self.storage.store_struct(Theme(name="theme1"))
        self.storage.store_struct(Theme(name="theme2"))
        self.storage.store_struct(Theme(name="theme3"))
        result = self.storage.list_structs(Theme)
        assert result == ["theme1", "theme2", "theme3"]

    def test_list_structs_different_types(self):
        llm = LlmConfig(model="gpt-4", provider="openai")
        self.storage.store_struct(Theme(name="theme1"))
        self.storage.store_struct(AgentConfig(name="agent1", llm=llm))
        themes = self.storage.list_structs(Theme)
        agents = self.storage.list_structs(AgentConfig)
        assert themes == ["theme1"]
        assert agents == ["agent1"]

    def test_yaml_format(self):
        with tempfile.TemporaryDirectory() as dir2:
            storage = FileSystemStorage(struct_paths=[dir2], struct_format="yaml")
            theme = Theme(name="yaml_theme")
            storage.store_struct(theme)
            retrieved = storage.retrieve_struct(Theme, "yaml_theme")
            assert retrieved.name == "yaml_theme"

    def test_yml_format(self):
        with tempfile.TemporaryDirectory() as dir2:
            storage = FileSystemStorage(struct_paths=[dir2], struct_format="yml")
            theme = Theme(name="yml_theme")
            storage.store_struct(theme)
            retrieved = storage.retrieve_struct(Theme, "yml_theme")
            assert retrieved.name == "yml_theme"

    def test_invalid_format_raises(self):
        with tempfile.TemporaryDirectory() as dir2:
            storage = FileSystemStorage(struct_paths=[dir2], struct_format="xml")
            theme = Theme(name="test")
            with pytest.raises(ValueError, match="Unsupported format"):
                storage.store_struct(theme)

    def test_multiple_paths_merge(self):
        with tempfile.TemporaryDirectory() as dir1, tempfile.TemporaryDirectory() as dir2:
            # Create theme in dir1 with welcome_message
            theme1_path = Path(dir1) / "themes" / "merge_theme.json"
            theme1_path.parent.mkdir(parents=True)
            theme1_path.write_text('{"kind": "theme", "version": 1, "name": "merge_theme", "text_styles": {"welcome_message": {"color": "red"}}}')
            
            # Create theme in dir2 with response style (should be merged)
            theme2_path = Path(dir2) / "themes" / "merge_theme.json"
            theme2_path.parent.mkdir(parents=True)
            theme2_path.write_text('{"kind": "theme", "version": 1, "name": "merge_theme", "text_styles": {"response": {"color": "blue"}}}')
            
            storage = FileSystemStorage(struct_paths=[dir1, dir2])
            retrieved = storage.retrieve_struct(Theme, "merge_theme")
            
            # First path (dir1) has precedence
            assert retrieved.text_styles.welcome_message.color == "red"
            # dir2's response should be merged in
            assert retrieved.text_styles.response.color == "blue"

    def test_expand_user_path(self):
        with tempfile.TemporaryDirectory() as dir2:
            # Use ~ in path
            home_dir = Path(dir2).expanduser()
            storage = FileSystemStorage(struct_paths=[str(home_dir)])
            theme = Theme(name="expand_theme")
            storage.store_struct(theme)
            retrieved = storage.retrieve_struct(Theme, "expand_theme")
            assert retrieved.name == "expand_theme"

    def test_store_struct_increment_new_struct(self):
        class MyIncStruct(BaseIncrementalStruct):
            kind: str = "test"
            version: int = 1
            value: str = "initial"

        struct = MyIncStruct(kind="test", version=1)
        event = BaseEvent(event_type="test_event")
        
        # Should fail if struct doesn't exist
        with pytest.raises(FileNotFoundError):
            self.storage.store_struct_increment(struct, event)

    def test_store_struct_increment_existing(self):
        class MyIncStruct(BaseIncrementalStruct):
            kind: str = "test"
            version: int = 1
            value: str = "initial"

        struct = MyIncStruct(kind="test", version=1)
        self.storage.store_struct(struct)
        
        event = BaseEvent(event_type="test_event")
        self.storage.store_struct_increment(struct, event)
        
        retrieved = self.storage.retrieve_struct(MyIncStruct, "test")
        assert len(retrieved.events) == 1
        assert retrieved.events[0].event_type == "test_event"

    def test_store_struct_increment_change_property(self):
        class MyIncStruct(BaseIncrementalStruct):
            kind: str = "test"
            version: int = 1
            value: str = "initial"

        struct = MyIncStruct(kind="test", version=1)
        self.storage.store_struct(struct)
        
        event = ChangeProperty.from_changing(struct, "value", "updated")
        self.storage.store_struct_increment(struct, event)
        
        retrieved = self.storage.retrieve_struct(MyIncStruct, "test")
        assert retrieved.value == "updated"
        assert len(retrieved.events) == 1

    def test_prepare_cleanup(self):
        from aideforge.storage.filesystem import LANGGRAPH_SUPPORTED
        if not LANGGRAPH_SUPPORTED:
            pytest.skip("langgraph not supported")
        self.storage.prepare()
        assert self.storage._checkpoint_store is not None
        self.storage.cleanup()
        assert self.storage._conn is None

    def test_get_checkpoint_store(self):
        from aideforge.storage.filesystem import LANGGRAPH_SUPPORTED
        if not LANGGRAPH_SUPPORTED:
            pytest.skip("langgraph not supported")
        checkpoint_store = self.storage.get_checkpoint_store()
        assert checkpoint_store is not None


class TestFileSystemStorageIncrementalJsonl:
    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage = FileSystemStorage(struct_paths=[self.temp_dir.name], struct_format="json")

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_incremental_uses_jsonl(self):
        class MyIncStruct(BaseIncrementalStruct):
            kind: str = "test"
            version: int = 1
            value: str = "initial"

        struct = MyIncStruct(kind="test", version=1)
        self.storage.store_struct(struct)
        
        event = BaseEvent(event_type="test_event")
        self.storage.store_struct_increment(struct, event)
        
        # Check that .jsonl file was created
        files = list(Path(self.temp_dir.name).rglob("*.jsonl"))
        assert len(files) == 1
        
        # Check content - JSONL has no spaces after colons
        content = files[0].read_text()
        assert 'event_type":"test_event"' in content


class TestFileSystemStorageYamlIncremental:
    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage = FileSystemStorage(struct_paths=[self.temp_dir.name], struct_format="yaml")

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_incremental_yaml_uses_document_separator(self):
        class MyIncStruct(BaseIncrementalStruct):
            kind: str = "test"
            version: int = 1
            value: str = "initial"

        struct = MyIncStruct(kind="test", version=1)
        self.storage.store_struct(struct)
        
        event = BaseEvent(event_type="test_event")
        self.storage.store_struct_increment(struct, event)
        
        files = list(Path(self.temp_dir.name).rglob("*.yaml"))
        assert len(files) == 1
        
        content = files[0].read_text()
        assert "---" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])