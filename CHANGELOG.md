# Changelog

## [0.1.1] - 2024-09-28

### Fixed
- **Connected actual model inference to API endpoints** - No more placeholder responses
- **Fixed streaming token generation** - Real-time token streaming now works
- **Improved model loading** - Proper engine initialization and model management
- **Enhanced error handling** - Better timeout handling and error messages

### Added
- **Test script** (`test_server.py`) - Comprehensive testing of all endpoints
- **Requirements.txt** - Simplified dependency installation
- **Better logging** - More detailed server logs and model loading status

### Changed
- **Server initialization** - Improved model loading workflow
- **API responses** - Now return actual model outputs instead of placeholders
- **Documentation** - Updated with working examples and troubleshooting

## [0.1.0] - 2024-09-28

### Added
- Initial release of MLX-vLLM
- OpenAI-compatible API server
- MLX model loading and management
- Continuous batching scheduler
- CLI interface with model recommendations
- Comprehensive documentation
