import pytest
from unittest.mock import MagicMock, patch

from aideforge.tools import (
    PreconfiguredTool,
    ToolPolicy,
    ToolType,
    ToolBlueprintType,
    BaseAideForgeTool,
    AideForgeStructuredTool,
    BaseAsyncTool,
    default_tools,
)
from aideforge.abc.tool import AgentToolProtocol
from langchain.tools import BaseTool
from langchain_core.tools import StructuredTool


class TestPreconfiguredTool:
    def test_creation(self):
        class MockTool:
            pass
        
        tool = PreconfiguredTool(tool=MockTool)
        assert tool.tool == MockTool
        assert tool.config is None

    def test_creation_with_config(self):
        class MockTool:
            pass
        
        config = {"setting": "value"}
        tool = PreconfiguredTool(tool=MockTool, config=config)
        assert tool.tool == MockTool
        assert tool.config == config


class TestToolPolicy:
    def test_values(self):
        assert ToolPolicy.INHERIT == "inherit"
        assert ToolPolicy.ALLOW == "allow"
        assert ToolPolicy.ASK == "ask"
        assert ToolPolicy.DENY == "deny"


class TestToolTypeAliases:
    def test_tool_type(self):
        # ToolType is a type alias, just verify it's defined
        assert ToolType is not None

    def test_tool_blueprint_type(self):
        # ToolBlueprintType is a type alias, just verify it's defined
        assert ToolBlueprintType is not None


class TestBaseAideForgeTool:
    def test_creation(self):
        class TestTool(BaseAideForgeTool):
            name: str = "test_tool"
            description: str = "Test tool"
            
            def _run(self, *args, **kwargs):
                return "test result"
        
        tool = TestTool()
        assert tool.name == "test_tool"
        assert tool.description == "Test tool"

    def test_update_description(self):
        class TestTool(BaseAideForgeTool):
            name: str = "test_tool"
            description: str = "Test tool with {name}"
            
            def _run(self, *args, **kwargs):
                return "test result"
        
        tool = TestTool()
        tool.update_description()
        assert tool.description == "Test tool with test_tool"


class TestAideForgeStructuredTool:
    def test_creation(self):
        from pydantic import BaseModel
        
        class TestInput(BaseModel):
            query: str
        
        class TestTool(AideForgeStructuredTool):
            name: str = "structured_tool"
            description: str = "Structured tool"
            args_schema: type = TestInput
            
            def _run(self, query: str, **kwargs):
                return f"structured result for {query}"
        
        tool = TestTool()
        assert tool.name == "structured_tool"
        assert tool.description == "Structured tool"
        assert tool.args_schema == TestInput


class TestBaseAsyncTool:
    def test_creation(self):
        class TestTool(BaseAsyncTool):
            name: str = "async_tool"
            description: str = "Async tool"
            
            async def _arun(self, *args, **kwargs):
                return "async result"
            
            def _run(self, *args, **kwargs):
                return "sync result"
        
        tool = TestTool()
        assert tool.name == "async_tool"
        assert tool.description == "Async tool"

    @pytest.mark.asyncio
    async def test_arun(self):
        class TestTool(BaseAsyncTool):
            name: str = "async_tool"
            description: str = "Async tool"
            
            async def _arun(self, *args, **kwargs):
                return "async result"
            
            def _run(self, *args, **kwargs):
                return "sync result"
        
        tool = TestTool()
        result = await tool._arun()
        assert result == "async result"


class TestDefaultTools:
    def test_default_tools_is_list(self):
        assert isinstance(default_tools, list)

    def test_default_tools_contains_base_tool_subclasses(self):
        for tool in default_tools:
            assert issubclass(tool, BaseTool)
            assert tool is not BaseTool


if __name__ == "__main__":
    pytest.main([__file__, "-v"])