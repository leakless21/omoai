# OmoAI Documentation Hub

Welcome to the documentation for OmoAI, a modern speech recognition and processing pipeline.

## 📖 Essential Documentation

- **[Migration Guide](./migration_guide.md)** - Migrating from legacy scripts to new pipeline modules
- **[Final Summary](./final_summary.md)** - Complete refactor summary and achievements

## 🚀 API Documentation

- **[API Reference](./api/reference.md)** - Complete API documentation with endpoints and examples
- **[Pipeline Endpoint Guide](./api/pipeline_endpoint.md)** - Detailed guide for the /pipeline/ endpoint with example responses
- **[Configuration Guide](./user_guide/configuration.md)** - API and system configuration

## 🏗️ Architecture & Development

- **[Architecture Overview](./architecture/index.md)** - System architecture and design
- **[Development Guide](./development/best_practices.md)** - Contributing and development setup
- **[Testing Report](./development/testing_report.md)** - Test coverage and quality metrics

## 📋 Project Status

- **[COMPLETED.md](../COMPLETED.md)** - Project completion status and achievements
- **[Requirements](./project/requirements.md)** - Functional and non-functional requirements
- **[Gap Analysis](./project/gap_analysis.md)** - Current gaps and known issues

---

## 📂 Documentation Structure

```
docs/
├── README.md                    # This documentation hub
├── migration_guide.md           # Migration from legacy scripts
├── final_summary.md             # Complete refactor summary
├── api/
│   ├── reference.md            # Complete API documentation
│   └── pipeline_endpoint.md    # Detailed /pipeline/ endpoint guide
├── architecture/
│   ├── index.md               # System architecture
│   ├── analysis_report.md     # Architecture analysis
│   └── punctuation.md         # Punctuation system details
├── development/
│   ├── best_practices.md      # Development setup
│   └── testing_report.md       # Test coverage
├── project/
│   ├── requirements.md        # Project requirements
│   └── gap_analysis.md        # Gaps and known issues
└── user_guide/
    └── configuration.md       # Configuration guide
```

## 🎯 Key Achievements

✅ **Modern Pipeline Architecture** - Clean separation of concerns with modular design
✅ **Configuration Standardization** - Unified OmoAIConfig across all modules
✅ **Custom Exception Handling** - Comprehensive error handling system
✅ **Legacy Script Archival** - Clean migration from legacy implementations
✅ **Test Suite Modernization** - Updated tests for new architecture
✅ **Performance Optimization** - Centralized memory management and debugging
✅ **Documentation Consolidation** - Streamlined, organized documentation
✅ **RESTful API** - Full-featured API with automatic fallback between high-performance and robust modes

## 🚀 Quick Start

### Starting the API Server

```bash
# Start the API server with uv
uv run litestar --app src.omoai.api.app:app run --host 0.0.0.0 --port 8000

# The API will be available at http://localhost:8000
# OpenAPI documentation at http://localhost:8000/schema
```

### Basic API Usage

```bash
# Process an audio file using the pipeline endpoint
curl -X POST "http://localhost:8000/pipeline" \
  -F "audio_file=@path/to/your/audio.mp3" \
  -H "Accept: application/json"
```

## 📞 Support

For questions or issues:

- Check the [Migration Guide](./migration_guide.md) for transition help
- Review the [API Reference](./api/reference.md) for endpoint details
- See [Development Guide](./development/best_practices.md) for contribution guidelines
- Consult the [Configuration Guide](./user_guide/configuration.md) for setup options
