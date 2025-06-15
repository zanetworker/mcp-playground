# MCP Playground Test Suite

This directory contains comprehensive tests for the MCP playground project, organized into unit tests and integration tests.

## Test Structure

```
tests/
├── conftest.py                          # pytest configuration & fixtures
├── unit/                                # Fast unit tests (isolated, mocked)
│   ├── test_message_duplication_fix.py  # Message duplication fix validation
│   ├── test_tool_calling_workflow.py    # Tool calling & final response tests
│   └── test_llm_bridge_methods.py       # LLM bridge method tests
├── integration/                         # Integration tests (real implementations)
│   └── test_message_fix_integration.py  # Cross-bridge integration tests
├── mocks/                              # Reusable mock classes
│   └── mock_components.py              # Mock LLM bridges, clients, results
└── fixtures/                           # Test data and fixtures
    └── test_data.py                    # Sample conversations, tool calls, responses
```

## Running Tests

### Run All Tests
```bash
# Run all unit and integration tests
pytest tests/unit/ tests/integration/ -v
```

### Run Unit Tests Only (Fast)
```bash
# Run only unit tests (recommended for development)
pytest tests/unit/ -v

# Run specific unit test file
pytest tests/unit/test_message_duplication_fix.py -v
```

### Run Integration Tests Only
```bash
# Run integration tests (may require API keys)
pytest tests/integration/ -v
```

### Run Tests by Marker
```bash
# Run only unit tests
pytest -m unit -v

# Run only integration tests
pytest -m integration -v

# Run only slow tests
pytest -m slow -v
```

## Test Categories

### Unit Tests (Fast, Isolated)
- **Message Duplication Fix Tests**: Verify that the new `process_messages()` method eliminates message duplication
- **Tool Calling Workflow Tests**: Ensure tool calling generates final LLM responses
- **LLM Bridge Method Tests**: Test new methods and backward compatibility

### Integration Tests (Real Implementations)
- **Cross-Bridge Tests**: Verify message fix works across all LLM bridge implementations
- **End-to-End Tests**: Test complete workflows with mocked LLM responses

## Test Coverage

The test suite covers:

✅ **Message Duplication Fix**
- No duplicate messages in conversation
- Conversation history integrity
- Original query extraction for tool processing

✅ **Tool Calling Workflow**
- Tool execution generates final responses
- Successful and failed tool call handling
- No-tool-call scenarios

✅ **LLM Bridge Methods**
- New `process_messages()` and `submit_messages()` methods
- Backward compatibility with old methods
- Error handling and edge cases

✅ **Integration Testing**
- OpenAI, Ollama, Anthropic, OpenRouter bridges
- Consistent behavior across all implementations
- Tool calling integration

## Environment Variables

Some integration tests may require API keys:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENROUTER_API_KEY="your-openrouter-key"
```

Tests will be skipped if the required API keys are not available.

## Test Development

### Adding New Tests

1. **Unit Tests**: Add to `tests/unit/` for isolated functionality testing
2. **Integration Tests**: Add to `tests/integration/` for cross-component testing
3. **Mock Objects**: Extend `tests/mocks/mock_components.py` for reusable mocks
4. **Test Data**: Add to `tests/fixtures/test_data.py` for sample data

### Test Fixtures

Use the provided fixtures in your tests:

```python
def test_example(sample_conversation, mock_mcp_client):
    # Use pre-configured test data and mocks
    pass
```

### Test Markers

Mark your tests appropriately:

```python
@pytest.mark.unit
def test_fast_unit_test():
    pass

@pytest.mark.integration
def test_integration_test():
    pass
```

## Continuous Integration

The test suite is designed for CI/CD:

- **Unit tests** run on every commit (fast, no external dependencies)
- **Integration tests** run on PR/merge (may require secrets for API keys)
- All tests use mocking to avoid external API calls during development

## Performance

- **Unit tests**: ~0.1 seconds (33 tests)
- **Integration tests**: ~0.1 seconds (6 tests, mocked)
- **Total runtime**: < 1 second for full test suite