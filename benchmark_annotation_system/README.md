# Benchmark Annotation System

Interactive visualization annotation tool using your project's real ToolExecutor.

## Quick Start

### 1. Place in your project directory

```
your-project/
├── tools/
│   ├── tool_executor.py
│   └── tool_registry.py
├── config/
│   └── chart_types.py
├── benchmark_annotation_system/   <-- Put this folder here
│   ├── backend/
│   │   ├── main.py
│   │   └── specs/
│   └── frontend/
```

### 2. Install dependencies

```bash
cd benchmark_annotation_system
pip install -r requirements.txt
```

### 3. Start the server

```bash
cd backend
python -m uvicorn main:app --reload --port 8002
```

### 4. Check the console output

You should see:
```
✓ Project ToolExecutor loaded successfully
✓ Tool registry loaded with XX tools
Loaded: 01_scatter_cars.json
Loaded: 02_bar_sales.json
...
Total specs loaded: 4
```

### 5. Open the browser

Visit http://localhost:8000

## Troubleshooting

### "Could not import ToolExecutor"

Edit `backend/main.py` line 20 to set the correct path:

```python
# If your structure is:
# /home/user/my-project/benchmark_annotation_system/backend/main.py
# And tools are at:
# /home/user/my-project/tools/

PROJECT_ROOT = BACKEND_DIR.parent.parent  # Goes up two levels
```

Or use an absolute path:
```python
PROJECT_ROOT = Path("/home/user/my-project")
```

### "Tool not found" errors

Make sure your project has:
- `tools/tool_executor.py` with `get_tool_executor()` function
- `tools/tool_registry.py` with `tool_registry` instance
- `config/chart_types.py` with `ChartType` enum

## Adding Your Own Specs

Add JSON files to `backend/specs/` folder:

### Format 1: Full format with metadata
```json
{
  "spec": { ... vega-lite spec ... },
  "meta": {
    "chart_type": "scatter",
    "title": "My Chart"
  }
}
```

### Format 2: Direct Vega/Vega-Lite spec
```json
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "title": "My Chart",
  ...
}
```

Call `POST /api/reload` to reload specs without restarting.

## Workflow

1. **Load spec** - Use Prev/Next to navigate
2. **Enter question** - What you want to explore
3. **Enter answer** - Ground truth answer (optional)
4. **Select tool** - Click on a tool button
5. **Set parameters** - Fill in required parameters
6. **Execute** - Click Execute button
7. **Record iteration** - Add insights and reasoning
8. **Repeat** - Do more iterations if needed
9. **Finish Question** - Generate benchmark JSON
10. **Export** - Download all benchmarks

## API Endpoints

- `GET /api/specs` - List all specs
- `GET /api/spec/{index}` - Get spec by index
- `POST /api/execute_tool` - Execute a tool
- `POST /api/finish` - Generate benchmark
- `POST /api/reload` - Reload specs
- `GET /api/tools` - List available tools
- `GET /api/tool/{name}` - Get tool details

## Sample Specs Included

- `01_scatter_cars.json` - Scatter plot (Horsepower vs MPG)
- `02_bar_sales.json` - Bar chart (Sales by Region)
- `03_line_stocks.json` - Line chart (Stock prices)
- `04_heatmap_temp.json` - Heatmap (Temperature)
