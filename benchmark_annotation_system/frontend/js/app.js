/**
 * Benchmark Annotation System - Main App
 * With proper Sankey rendering, sampling notice, and sortable support
 */

const State = {
    specList: [],
    currentIndex: 0,
    currentSpec: null,
    originalSpec: null,
    specHistory: [],
    chartMeta: {},
    chartType: 'scatter',
    vegaView: null,
    currentQuestion: '',
    currentAnswer: '',
    currentAnswerType: 'categorical',
    currentAnswerConfig: {},
    activeTool: null,
    pendingParams: {},
    iterations: [],
    completedBenchmarks: []
};

// Logger
const Log = {
    el: null,
    init() { this.el = document.getElementById('realtime-log'); },
    add(type, msg, data = null) {
        if (!this.el) return;
        const time = new Date().toLocaleTimeString();
        const icons = { tool: '[TOOL]', param: '[PARAM]', execute: '[EXEC]', result: '[OK]', error: '[ERR]', info: '[INFO]' };
        let html = `<div class="log-entry log-${type}"><span class="log-time">${time}</span><span class="log-icon">${icons[type] || '[LOG]'}</span> ${msg}`;
        if (data) html += `<pre class="log-data">${JSON.stringify(data, null, 2)}</pre>`;
        html += '</div>';
        this.el.innerHTML += html;
        this.el.scrollTop = this.el.scrollHeight;
        console.log(`[${type}] ${msg}`, data || '');
    },
    clear() { if (this.el) this.el.innerHTML = ''; }
};

// API
const API = {
    async getSpecs() {
        const r = await fetch('/api/specs');
        return r.json();
    },
    async getSpec(i) {
        const r = await fetch(`/api/spec/${i}`);
        return r.json();
    },
    async executeTool(name, params, spec) {
        // 调试：检查发送的 spec 是否包含 _sankey_state
        if (name === 'expand_node' || name === 'collapse_nodes') {
            const hasState = spec && '_sankey_state' in spec;
            const stateKeys = hasState ? Object.keys(spec._sankey_state || {}) : [];
            console.log(`[DEBUG] executeTool(${name}): sending spec with _sankey_state=${hasState}, keys=${stateKeys.join(',')}`);
        }
        const r = await fetch('/api/execute_tool', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                tool_name: name, 
                params, 
                vega_spec: spec,
                spec_index: State.currentIndex  // 传递索引用于重采样
            })
        });
        if (!r.ok) {
            const e = await r.json();
            throw new Error(e.detail || 'Failed');
        }
        const result = await r.json();
        
        // 显示重采样信息
        if (result.result?.sampling_info) {
            Log.add('info', result.result.sampling_info.message);
        }
        
        return result;
    },
    async finish(data) {
        const r = await fetch('/api/finish', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return r.json();
    }
};

// Data extraction
const DataEx = {
    get(spec) {
        if (spec?.data?.values) return spec.data.values;
        if (Array.isArray(spec?.data)) {
            const d = spec.data.find(x => x.values);
            return d?.values || [];
        }
        return [];
    },
    getXCats(spec) {
        const d = this.get(spec), f = spec?.encoding?.x?.field;
        return f ? [...new Set(d.map(x => x[f]))].filter(Boolean) : [];
    },
    getYCats(spec) {
        const d = this.get(spec), f = spec?.encoding?.y?.field;
        return f ? [...new Set(d.map(x => x[f]))].filter(Boolean) : [];
    },
    getColorCats(spec) {
        const d = this.get(spec), f = spec?.encoding?.color?.field;
        return f ? [...new Set(d.map(x => x[f]))].filter(Boolean) : [];
    },
    getSeries(spec) {
        const d = this.get(spec);
        const f = spec?.encoding?.color?.field || spec?.encoding?.strokeDash?.field;
        return f ? [...new Set(d.map(x => x[f]))].filter(Boolean) : [];
    },
    getAllFields(spec) {
        const d = this.get(spec);
        return d.length ? Object.keys(d[0]) : [];
    },
    getCategoricalFields(spec) {
        const d = this.get(spec);
        if (!d.length) return [];
        const sample = d[0];
        return Object.keys(sample).filter(k => typeof sample[k] === 'string');
    },
    getTimeValues(spec) {
        // Prefer x scale domain if available; fallback to data values
        const getEnc = () => {
            if (spec?.layer && spec.layer.length > 0) {
                return spec.layer[0].encoding || {};
            }
            return spec?.encoding || {};
        };
        const enc = getEnc();
        const xEnc = enc?.x || {};
        const xField = xEnc?.field;
        if (!xField) return [];
        
        const domain = xEnc?.scale?.domain;
        let values = [];
        if (Array.isArray(domain) && domain.length > 0) {
            values = domain.slice();
        } else {
            const d = this.get(spec);
            values = d.map(row => row?.[xField]).filter(v => v !== undefined && v !== null);
        }
        
        // 去重
        const uniq = [...new Set(values)];
        const looksLikeDate = (v) => typeof v === 'string' && /^\d{4}-\d{2}-\d{2}/.test(v);
        const allDateLike = uniq.length > 0 && uniq.every(looksLikeDate);
        const allNumber = uniq.length > 0 && uniq.every(v => typeof v === 'number');
        
        if (allDateLike) {
            return uniq.sort((a, b) => new Date(a).getTime() - new Date(b).getTime());
        }
        if (allNumber) {
            return uniq.sort((a, b) => a - b);
        }
        return uniq.sort((a, b) => String(a).localeCompare(String(b)));
    },
    getAllCategories(spec) {
        const d = this.get(spec);
        if (!d.length) return [];
        
        const colorField = spec?.encoding?.color?.field;
        const xField = spec?.encoding?.x?.field;
        const xType = spec?.encoding?.x?.type;
        
        let cats = [];
        
        if (colorField) {
            cats = [...new Set(d.map(x => x[colorField]))].filter(Boolean);
        }
        
        if (xField && (xType === 'nominal' || xType === 'ordinal')) {
            const xCats = [...new Set(d.map(x => x[xField]))].filter(Boolean);
            cats = [...new Set([...cats, ...xCats])];
        }
        
        if (!cats.length) {
            const sample = d[0];
            for (const [field, value] of Object.entries(sample)) {
                if (typeof value === 'string') {
                    cats = [...new Set(d.map(x => x[field]))].filter(Boolean);
                    break;
                }
            }
        }
        
        return cats;
    },
    getDimensions(spec) {
        // 方法1: 从 fold transform 获取 (平行坐标图标准格式)
        const transforms = spec?.transform || [];
        for (const t of transforms) {
            if (t.fold && Array.isArray(t.fold)) {
                return t.fold;
            }
        }
        
        // 方法2: 从 repeat.column 获取
        if (spec?.repeat?.column) {
            return spec.repeat.column;
        }
        
        // 方法3: 从 x encoding 的 sort 数组获取 (预归一化长格式)
        const getXSort = (enc) => enc?.x?.sort;
        if (spec?.layer) {
            for (const layer of spec.layer) {
                const sort = getXSort(layer.encoding);
                if (Array.isArray(sort) && sort.length > 0) {
                    return sort;
                }
            }
        }
        if (Array.isArray(getXSort(spec?.encoding))) {
            return getXSort(spec.encoding);
        }
        
        // 方法4: 从 x encoding 的 scale.domain 获取
        const getXDomain = (enc) => enc?.x?.scale?.domain;
        if (spec?.layer) {
            for (const layer of spec.layer) {
                const domain = getXDomain(layer.encoding);
                if (Array.isArray(domain) && domain.length > 0) {
                    return domain;
                }
            }
        }
        if (Array.isArray(getXDomain(spec?.encoding))) {
            return getXDomain(spec.encoding);
        }
        
        // 方法5: 从数据的 dimension 字段获取唯一值 (预归一化长格式)
        const data = this.get(spec);
        if (data.length > 0 && data[0].dimension !== undefined) {
            const dims = [...new Set(data.map(d => d.dimension).filter(Boolean))];
            if (dims.length > 0) {
                return dims;
            }
        }
        
        // 方法6: 从 layer 的 y encoding 获取
        if (spec?.layer) {
            const dims = spec.layer
                .map(l => l.encoding?.y?.field)
                .filter(Boolean);
            if (dims.length > 0) {
                return [...new Set(dims)];
            }
        }
        
        // 方法7: 从数据中获取数值字段 (最后手段)
        if (data.length > 0) {
            const sample = data[0];
            return Object.keys(sample).filter(k => typeof sample[k] === 'number');
        }
        
        return [];
    },
    getNodes(spec) {
        if (!Array.isArray(spec?.data)) return [];
        // Format 1: nodes + links
        const n = spec.data.find(d => d.name === 'nodes');
        if (n?.values?.length) return n.values.map(x => x.name || x.id).filter(Boolean);
        // Format 2: rawLinks + nodeConfig (sankey_tools)
        const nc = spec.data.find(d => d.name === 'nodeConfig');
        if (nc?.values?.length) return nc.values.map(x => x.name || x.id).filter(Boolean);
        // Format 3: derive from links/rawLinks
        const linksData = spec.data.find(d => d.name === 'links' || d.name === 'rawLinks');
        if (linksData?.values?.length) {
            const names = new Set();
            linksData.values.forEach(l => { names.add(l.source); names.add(l.target); });
            return [...names].filter(Boolean);
        }
        return [];
    },
    getTemporalField(spec) {
        const enc = spec?.layer?.[0]?.encoding || spec?.encoding || {};
        const x = enc?.x || {};
        if (x.type === 'temporal' && x.field) return x.field;
        if (x.field) return x.field;
        const d = this.get(spec);
        if (!d.length) return null;
        const sample = d[0];
        for (const k of Object.keys(sample)) {
            const v = sample[k];
            if (typeof v === 'string' && /^\d{4}-\d{2}/.test(v)) return k;
        }
        return null;
    },
    getDrilldownYears(spec) {
        const field = this.getTemporalField(spec);
        const d = this.get(spec);
        if (!field || !d.length) return [];
        const years = new Set();
        for (const row of d) {
            const v = row[field];
            if (v == null) continue;
            let y = NaN;
            if (typeof v === 'number') y = Math.floor(v);
            else if (typeof v === 'string') {
                const m = v.match(/^(\d{4})/);
                if (m) y = parseInt(m[1], 10);
            }
            if (Number.isFinite(y)) years.add(y);
        }
        return [...years].sort((a, b) => a - b);
    },
    getDrilldownMonths(spec) {
        const d = this.get(spec);
        const field = this.getTemporalField(spec);
        if (!field || !d.length) return [1,2,3,4,5,6,7,8,9,10,11,12];
        const months = new Set();
        for (const row of d) {
            const v = row[field];
            if (v == null) continue;
            let m = NaN;
            if (typeof v === 'number') m = v;
            else if (typeof v === 'string') {
                const ms = v.match(/-0?(\d{1,2})[-T]/) || v.match(/-(\d{2})$/);
                if (ms) m = parseInt(ms[1], 10);
            }
            if (m >= 1 && m <= 12) months.add(m);
        }
        const arr = [...months].sort((a, b) => a - b);
        return arr.length ? arr : [1,2,3,4,5,6,7,8,9,10,11,12];
    },
    getAggregateNodes(spec) {
        // 优先从 _sankey_state.collapsed_groups 读取折叠聚合（标准方式，与 sankey_tools 一致）
        if (spec?._sankey_state?.collapsed_groups) {
            return Object.keys(spec._sankey_state.collapsed_groups);
        }
        // Fallback：按名称过滤（兼容旧数据或没有 _sankey_state 的情况）
        const nodes = this.getNodes(spec);
        return nodes.filter(n => n.includes('Other') || n.includes('Others'));
    },
    getSankeyFields(spec) {
        if (!Array.isArray(spec?.data)) return [];
        const hasNodes = spec.data.some(d => d?.name === 'nodes');
        const hasLinks = spec.data.some(d => d?.name === 'links');
        if (!hasNodes || !hasLinks) return [];
        return ['name', 'depth', 'source', 'target', 'value'];
    },
    getCatFields(spec) {
        const d = this.get(spec);
        return d.length ? Object.keys(d[0]).filter(k => typeof d[0][k] === 'string') : [];
    },
    getCatValues(spec, field) {
        const d = this.get(spec);
        return field ? [...new Set(d.map(x => x[field]))].filter(Boolean) : [];
    },
    // 获取 series（折线名等），排除 x 轴时间字段；支持 layer 折线图
    getSeriesOrCategorical(spec) {
        const d = this.get(spec);
        if (!d.length) return [];
        
        const xField = spec?.encoding?.x?.field;
        const yField = spec?.encoding?.y?.field;
        const enc = spec?.layer?.[0]?.encoding || spec?.encoding || {};
        
        // 优先 color（折线/系列），再 strokeDash
        let field = enc?.color?.field || enc?.strokeDash?.field;
        if (field && field !== xField && field !== yField) {
            return [...new Set(d.map(x => x[field]))].filter(Boolean);
        }
        // detail 仅当非 index / 非时间时用
        field = enc?.detail?.field;
        if (field && field !== xField && field !== yField) {
            const vals = [...new Set(d.map(x => x[field]))].filter(Boolean);
            const looksLikeDate = (v) => typeof v === 'string' && /^\d{4}-\d{2}-\d{2}/.test(v);
            if (vals.length && !vals.slice(0, 5).every(looksLikeDate)) {
                return vals;
            }
        }
        // 回退：第一个非 x/y 的 categorical，且排除纯日期
        const sample = d[0];
        const categoricalFields = Object.keys(sample).filter(k => typeof sample[k] === 'string' && k !== xField && k !== yField);
        for (const k of categoricalFields) {
            const vals = [...new Set(d.map(x => x[k]))].filter(Boolean);
            const looksLikeDate = (v) => typeof v === 'string' && /^\d{4}-\d{2}-\d{2}/.test(v);
            if (vals.length && !vals.slice(0, 5).every(looksLikeDate)) return vals;
        }
        return [];
    },
    getOpts(source, spec) {
        const map = {
            'x_categories': () => this.getXCats(spec),
            'y_categories': () => this.getYCats(spec),
            'color_categories': () => this.getColorCats(spec),
            'all_categories': () => this.getAllCategories(spec),
            'series': () => this.getSeries(spec),
            'series_or_categorical': () => this.getSeriesOrCategorical(spec),
            'all_fields': () => this.getAllFields(spec),
            'categorical_fields': () => this.getCategoricalFields(spec),
            'time_values': () => this.getTimeValues(spec),
            'dimensions': () => this.getDimensions(spec),
            'nodes': () => this.getNodes(spec),
            'aggregate_nodes': () => this.getAggregateNodes(spec),
            'sankey_fields': () => this.getSankeyFields(spec),
            'category_fields': () => this.getCatFields(spec),
            'category_values': () => this.getCatValues(spec, State.pendingParams?.field),
            'drilldown_years': () => this.getDrilldownYears(spec),
            'drilldown_months': () => this.getDrilldownMonths(spec)
        };
        return (map[source] || (() => []))();
    }
};

// UI
const UI = {
    buildTools(type) {
        const tools = getToolsForChartType(type);
        const common = document.getElementById('common-tools');
        const specific = document.getElementById('specific-tools');
        
        if (common) {
            common.innerHTML = '';
            Object.entries(tools.common).forEach(([n, c]) => {
                common.appendChild(this.toolBtn(n, c));
            });
        }
        
        if (specific) {
            specific.innerHTML = '';
            Object.entries(tools.specific).forEach(([n, c]) => {
                specific.appendChild(this.toolBtn(n, c));
            });
        }
    },
    
    toolBtn(name, config) {
        const b = document.createElement('button');
        b.className = 'tool-btn';
        b.dataset.tool = name;
        b.title = config.description;
        b.textContent = name.replace(/_/g, ' ');
        b.onclick = () => Tools.select(name, config);
        return b;
    },
    
    buildParams(name, config) {
        const c = document.getElementById('param-panel');
        if (!c) return;
        
        c.innerHTML = `<h4>${name.replace(/_/g, ' ')}</h4>`;
        const form = document.createElement('div');
        form.className = 'param-form';
        
        Object.entries(config.params || {}).forEach(([pn, pc]) => {
            form.appendChild(this.paramInput(pn, pc));
        });
        
        const exec = document.createElement('button');
        exec.className = 'execute-btn';
        exec.textContent = 'Execute';
        exec.onclick = () => Tools.execute();
        form.appendChild(exec);
        
        const cancel = document.createElement('button');
        cancel.className = 'cancel-btn';
        cancel.textContent = 'Cancel';
        cancel.onclick = () => Tools.cancel();
        form.appendChild(cancel);
        
        c.appendChild(form);
        c.style.display = 'block';
    },
    
    paramInput(name, config) {
        const w = document.createElement('div');
        w.className = 'param-input-wrapper';
        w.dataset.paramName = name;
        w.dataset.source = config.source || '';
        
        const label = document.createElement('label');
        label.textContent = name.replace(/_/g, ' ') + (config.required ? ' *' : '');
        w.appendChild(label);
        
        let input;
        const opts = config.options || DataEx.getOpts(config.source, State.currentSpec);
        
        switch (config.type) {
            case 'select':
                input = UIComponents.createSelect(name, opts, {
                    placeholder: '请选择',
                    default: config.default
                });
                // 添加联动支持：监听选项变化
                if (config.source === 'categorical_fields' || config.source === 'all_fields') {
                    // 使用MutationObserver监听data-value变化
                    const observer = new MutationObserver((mutations) => {
                        mutations.forEach(m => {
                            if (m.attributeName === 'data-value') {
                                const selectedValue = input.dataset.value;
                                if (selectedValue) {
                                    State.pendingParams[name] = selectedValue;
                                    this.updateDependentParams(name, selectedValue);
                                }
                            }
                        });
                    });
                    observer.observe(input, { attributes: true, attributeFilter: ['data-value'] });
                }
                break;
                
            case 'multi-select':
                input = UIComponents.createMultiSelect(name, opts, {
                    placeholder: '请选择（可多选）'
                });
                break;
                
            case 'number':
                input = UIComponents.createNumberInput(name, config);
                break;
                
            case 'text':
                input = UIComponents.createTextInput(name, config);
                break;
                
            case 'checkbox':
                input = UIComponents.createCheckbox(name, config);
                break;
                
            case 'range-input':
                input = UIComponents.createRangeInput(name, config);
                break;
                
            case 'brush':
                input = UIComponents.createBrushDisplay(name, config);
                break;
            
            case 'sortable-list':
                input = UIComponents.createSortableList(name, opts, config);
                break;
                
            case 'path-builder':
                input = UIComponents.createPathBuilder(name, opts, config);
                break;
                
            default:
                input = UIComponents.createTextInput(name, config);
        }
        
        if (input) {
            w.appendChild(input);
        }
        
        // 添加hint说明文字
        if (config.hint) {
            const hint = document.createElement('div');
            hint.className = 'ui-hint';
            hint.textContent = config.hint;
            w.appendChild(hint);
        }
        
        return w;
    },
    
    // 更新依赖参数（联动）
    updateDependentParams(changedParam, newValue) {
        const form = document.querySelector('.param-form');
        if (!form) return;
        
        // 查找依赖 category_values 的参数（它依赖于 field 参数）
        if (changedParam === 'field') {
            const dependentWrapper = form.querySelector('[data-source="category_values"]');
            if (dependentWrapper) {
                const newOpts = DataEx.getCatValues(State.currentSpec, newValue);
                const oldMultiSelect = dependentWrapper.querySelector('.ui-multi-select');
                if (oldMultiSelect) {
                    // 重新创建多选组件
                    const paramName = dependentWrapper.dataset.paramName;
                    const newMultiSelect = UIComponents.createMultiSelect(paramName, newOpts, {
                        placeholder: '请选择（可多选）'
                    });
                    oldMultiSelect.replaceWith(newMultiSelect);
                }
            }
        }
    },
    
    hideParams() {
        const c = document.getElementById('param-panel');
        if (c) c.style.display = 'none';
    },
    
    updateMeta() {
        const m = State.chartMeta;
        const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v || '-'; };
        set('chart-type-display', m.chart_type);
        set('chart-title-display', m.title);
        set('chart-topic-display', m.topic);
        set('data-points-display', m.data_points);
        set('current-index', State.currentIndex + 1);
        set('total-specs', State.specList.length);
        const ni = document.getElementById('nav-index-input');
        if (ni) { ni.placeholder = '跳转'; ni.setAttribute('data-max', Math.max(1, State.specList.length)); }
        
        this.updateSamplingNotice();
    },
    
    updateSamplingNotice() {
        const notice = document.getElementById('sampling-notice');
        if (!notice) return;
        
        const chartType = (State.chartType || '').toLowerCase();
        const data = DataEx.get(State.currentSpec);
        const displayedCount = data.length;
        const totalCount = State.chartMeta?.total_points || State.chartMeta?.data_points || displayedCount;
        const isSampled = State.chartMeta?.is_sampled || (totalCount > displayedCount);
        
        if (chartType === 'scatter' || chartType === 'scatterplot') {
            notice.style.display = 'block';
            notice.className = 'sampling-notice scatter-notice';
            notice.innerHTML = isSampled 
                ? `<strong>Data Sampling Active:</strong> Displaying <b>${displayedCount}</b> of <b>${totalCount}</b> total points (max 500 per view).<br>
                   <small>Use <code>zoom_dense_area</code> or <code>select_region</code> to focus on specific areas and load more points.</small>`
                : `<strong>Data Info:</strong> Displaying all <b>${displayedCount}</b> data points.`;
        } else if (chartType === 'sankey' || chartType === 'sankeydiagram') {
            const aggregateNodes = DataEx.getAggregateNodes(State.currentSpec);
            notice.style.display = 'block';
            notice.className = 'sampling-notice sankey-notice';
            if (aggregateNodes.length > 0) {
                notice.innerHTML = `<strong>Auto-Collapse Active:</strong> Each layer displays up to <b>5</b> nodes by default.<br>
                    Remaining nodes are collapsed into "${aggregateNodes.join(', ')}".<br>
                    <small>Use <code>expand_node</code> to reveal hidden nodes, <code>auto_collapse_by_rank</code> to adjust.</small>`;
            } else {
                notice.innerHTML = `<strong>Sankey Info:</strong> Use <code>collapse_nodes</code> to merge nodes, 
                    <code>trace_node</code> to highlight paths, <code>filter_flow</code> to filter by threshold.`;
            }
        } else {
            notice.style.display = 'none';
        }
    },
    
    updateIterations() {
        const c = document.getElementById('iterations-list');
        if (!c) return;
        c.innerHTML = '';
        State.iterations.forEach((it, i) => {
            const d = document.createElement('div');
            d.className = 'iteration-item';
            d.innerHTML = `
                <div class="iteration-header">${i + 1}. ${it.tool_name}</div>
                <div class="iteration-params">${JSON.stringify(it.parameters)}</div>
                ${it.key_insights?.length ? `<div class="iteration-insights">Insights: ${it.key_insights.join('; ')}</div>` : ''}
                ${it.reasoning ? `<div class="iteration-reasoning">${it.reasoning}</div>` : ''}
            `;
            c.appendChild(d);
        });
    },
    
    showStatus(msg, type = 'info') {
        const s = document.getElementById('status-message');
        if (s) {
            s.textContent = msg;
            s.className = `status-message status-${type}`;
            s.style.display = 'block';
            setTimeout(() => s.style.display = 'none', 3000);
        }
    }
};

// Tools
const Tools = {
    select(name, config) {
        this.cancel();
        State.activeTool = name;
        State.pendingParams = {};
        State._activeToolConfig = config;
        
        Log.add('tool', `Selected: ${name}`);
        
        document.querySelectorAll('.tool-btn').forEach(b => {
            b.classList.toggle('active', b.dataset.tool === name);
        });
        
        if (config.interaction === 'brush') {
            Log.add('info', 'Draw rectangle on chart to select area');
            this.enableBrush();
        }
        
        UI.buildParams(name, config);
    },
    
    cancel() {
        State.activeTool = null;
        State.pendingParams = {};
        State._activeToolConfig = null;
        document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
        UI.hideParams();
        this.disableBrush();
    },
    
    enableBrush() {
        const c = document.getElementById('vega-chart');
        if (c) c.classList.add('brush-mode');
    },
    
    disableBrush() {
        const c = document.getElementById('vega-chart');
        if (c) c.classList.remove('brush-mode');
    },
    
    handleBrush(value) {
        if (!State.activeTool || !value) return;
        
        const spec = State.currentSpec;
        const xf = spec?.encoding?.x?.field || 'x';
        const yf = spec?.encoding?.y?.field || 'y';
        
        let xr = value[xf] || value.x;
        let yr = value[yf] || value.y;
        
        if (xr && yr) {
            if (Array.isArray(xr)) xr = [Math.min(...xr), Math.max(...xr)];
            if (Array.isArray(yr)) yr = [Math.min(...yr), Math.max(...yr)];
            
            State.pendingParams.x_range = xr;
            State.pendingParams.y_range = yr;
            
            Log.add('param', 'Brush captured', { x_range: xr, y_range: yr });
            
            const xd = document.getElementById('brush-x_range');
            const yd = document.getElementById('brush-y_range');
            if (xd) xd.textContent = `X: [${xr[0].toFixed(1)}, ${xr[1].toFixed(1)}]`;
            if (yd) yd.textContent = `Y: [${yr[0].toFixed(1)}, ${yr[1].toFixed(1)}]`;
        }
    },
    
    collectParams() {
        const params = { ...State.pendingParams };
        const form = document.querySelector('.param-form');
        if (!form) return params;
        
        // 处理普通input
        form.querySelectorAll('.ui-input').forEach(el => {
            const n = el.name;
            if (!n) return;
            
            if (el.type === 'number' && el.value) {
                params[n] = parseFloat(el.value);
            } else if (el.value) {
                params[n] = el.value;
            }
        });
        
        // 处理复选框
        form.querySelectorAll('.ui-checkbox-wrapper input[type="checkbox"]').forEach(el => {
            if (el.name) {
                params[el.name] = el.checked;
            }
        });
        
        // 处理单选下拉框
        form.querySelectorAll('.ui-select').forEach(select => {
            const name = select.dataset.name;
            const value = select.dataset.value;
            if (name && value) {
                params[name] = value;
            }
        });
        
        // 处理多选下拉框
        form.querySelectorAll('.ui-multi-select').forEach(select => {
            const name = select.dataset.name;
            const value = select.dataset.value;
            if (name && value) {
                try {
                    const arr = JSON.parse(value);
                    if (arr.length > 0) {
                        params[name] = arr;
                    }
                } catch(e) {}
            }
        });
        
        // 处理范围输入
        form.querySelectorAll('.ui-range-input').forEach(g => {
            const min = g.querySelector('[name$="_min"]');
            const max = g.querySelector('[name$="_max"]');
            if (min?.value && max?.value) {
                const base = min.name.replace('_min', '');
                params[base] = [parseFloat(min.value), parseFloat(max.value)];
            }
        });
        
        // 处理可排序列表
        form.querySelectorAll('.ui-sortable-list').forEach(wrapper => {
            const name = wrapper.dataset.name;
            const items = wrapper.querySelectorAll('.ui-sortable-item');
            if (name && items.length > 0) {
                params[name] = Array.from(items).map(li => li.dataset.value);
            }
        });
        
        // 处理路径构建器
        form.querySelectorAll('.ui-path-builder').forEach(wrapper => {
            const name = wrapper.dataset.name;
            const items = wrapper.querySelectorAll('.ui-path-item');
            if (name && items.length > 0) {
                params[name] = Array.from(items).map(li => li.dataset.value);
            }
        });
        
        return params;
    },
    
    async execute() {
        if (!State.activeTool) {
            Log.add('error', 'No tool selected');
            return;
        }
        
        const params = this.collectParams();
        
        // Ensure vega_spec contains metadata for reset_view and undo_view
        const specToSend = JSON.parse(JSON.stringify(State.currentSpec));
        if (!specToSend._original_spec && State.originalSpec) {
            specToSend._original_spec = JSON.parse(JSON.stringify(State.originalSpec));
        }
        if (!specToSend._spec_history) {
            specToSend._spec_history = JSON.parse(JSON.stringify(State.specHistory));
        }
        
        Log.add('execute', `${State.activeTool}`, params);
        
        try {
            // Save current spec to history before executing (except for reset/undo)
            if (State.activeTool !== 'reset_view' && State.activeTool !== 'undo_view') {
                const currentSpecCopy = JSON.parse(JSON.stringify(State.currentSpec));
                State.specHistory.push(currentSpecCopy);
                // Update metadata in specToSend to include updated history
                specToSend._spec_history = JSON.parse(JSON.stringify(State.specHistory));
            }
            
            const result = await API.executeTool(State.activeTool, params, specToSend);
            
            // Check if execution was successful
            if (result.success === false) {
                const errorMsg = result.result?.error || 'Tool execution failed';
                Log.add('error', errorMsg);
                if (result.result?.traceback) {
                    console.error('Traceback:', result.result.traceback);
                }
                UI.showStatus(errorMsg, 'error');
                // Restore history
                if (State.activeTool !== 'reset_view' && State.activeTool !== 'undo_view') {
                    State.specHistory.pop();
                }
                return;
            }
            
            Log.add('result', 'Success', result.result);
            
            if (result.new_spec) {
                // 调试：检查收到的 new_spec 是否包含 _sankey_state
                if (State.activeTool === 'expand_node' || State.activeTool === 'collapse_nodes') {
                    const hasState = result.new_spec && '_sankey_state' in result.new_spec;
                    const stateKeys = hasState ? Object.keys(result.new_spec._sankey_state || {}) : [];
                    console.log(`[DEBUG] execute result: received new_spec with _sankey_state=${hasState}, keys=${stateKeys.join(',')}`);
                }
                State.currentSpec = result.new_spec;
                // Ensure metadata is preserved in the new spec
                if (!State.currentSpec._original_spec && State.originalSpec) {
                    State.currentSpec._original_spec = JSON.parse(JSON.stringify(State.originalSpec));
                }
                // Handle spec_history synchronization
                if (State.activeTool === 'undo_view') {
                    // undo_view pops from history, so we need to sync State.specHistory
                    // The backend modifies the history (pop), so we remove the last item from State
                    if (State.specHistory.length > 0) {
                        State.specHistory.pop();
                    }
                    State.currentSpec._spec_history = JSON.parse(JSON.stringify(State.specHistory));
                } else {
                    // For other tools, sync from State to metadata
                    State.currentSpec._spec_history = JSON.parse(JSON.stringify(State.specHistory));
                }
                await Chart.render(result.new_spec);
                UI.updateSamplingNotice();
            }
            
            this.showInsightForm({ tool_name: State.activeTool, parameters: params, result: result.result });
            
        } catch (e) {
            Log.add('error', e.message);
            UI.showStatus(e.message, 'error');
            // Restore history
            if (State.activeTool !== 'reset_view' && State.activeTool !== 'undo_view') {
                State.specHistory.pop();
            }
        }
    },
    
    showInsightForm(iter) {
        const c = document.getElementById('insight-form-container');
        if (!c) return;
        
        c.innerHTML = `
            <div class="insight-form">
                <h4>Record: ${iter.tool_name}</h4>
                <div class="form-group">
                    <label>Key Insights:</label>
                    <div id="insights-list" class="insights-list">
                        <div class="insight-row" data-index="0">
                            <input type="text" class="insight-input" placeholder="Insight 1..." />
                            <button type="button" class="remove-insight-btn" onclick="Tools.removeInsight(0)" title="Remove">×</button>
                        </div>
                    </div>
                    <button type="button" class="add-insight-btn" onclick="Tools.addInsight()">+ Add Insight</button>
                </div>
                <div class="form-group">
                    <label>Reasoning:</label>
                    <textarea id="reasoning-input" rows="2"></textarea>
                </div>
                <button onclick="Tools.saveIter()">Save</button>
                <button onclick="Tools.skipIter()">Skip</button>
            </div>
        `;
        c.style.display = 'block';
        
        // Store temp iteration
        State._tempIter = iter;
        State._insightCount = 1;
    },

    addInsight() {
        const list = document.getElementById('insights-list');
        if (!list) return;
        const count = list.querySelectorAll('.insight-row').length;
        if (count >= 5) {
            Log.add('info', 'Maximum 5 insights allowed');
            return;
        }
        const idx = State._insightCount || count;
        const row = document.createElement('div');
        row.className = 'insight-row';
        row.dataset.index = idx;
        row.innerHTML = `
            <input type="text" class="insight-input" placeholder="Insight ${count + 1}..." />
            <button type="button" class="remove-insight-btn" onclick="Tools.removeInsight(${idx})" title="Remove">×</button>
        `;
        list.appendChild(row);
        State._insightCount = idx + 1;
    },

    removeInsight(idx) {
        const list = document.getElementById('insights-list');
        if (!list) return;
        const rows = list.querySelectorAll('.insight-row');
        if (rows.length <= 1) return; // Keep at least 1
        const row = list.querySelector(`.insight-row[data-index="${idx}"]`);
        if (row) row.remove();
        // Update placeholders
        list.querySelectorAll('.insight-row').forEach((r, i) => {
            r.querySelector('.insight-input').placeholder = `Insight ${i + 1}...`;
        });
    },
    
    saveIter() {
        const iter = State._tempIter;
        if (!iter) return;
        
        const inputs = document.querySelectorAll('#insights-list .insight-input');
        const insights = Array.from(inputs).map(el => el.value.trim()).filter(Boolean);
        const reasoning = document.getElementById('reasoning-input')?.value;
        
        iter.key_insights = insights;
        iter.reasoning = reasoning?.trim() || '';
        
        State.iterations.push(iter);
        Log.add('info', 'Iteration saved');
        
        document.getElementById('insight-form-container').style.display = 'none';
        UI.updateIterations();
        UI.hideParams();
        this.cancel();
        delete State._tempIter;
    },
    
    skipIter() {
        const iter = State._tempIter;
        if (iter) {
            iter.key_insights = [];
            iter.reasoning = '';
            State.iterations.push(iter);
        }
        
        document.getElementById('insight-form-container').style.display = 'none';
        UI.updateIterations();
        UI.hideParams();
        this.cancel();
        delete State._tempIter;
        delete State._insightCount;
    }
};

// Chart - Simple Vega rendering
const Chart = {
    /**
     * Detect spec type based on $schema and heuristics
     */
    detectSpecType(spec) {
        const schema = spec.$schema || '';
        
        if (schema.includes('vega-lite')) {
            return 'vega-lite';
        }
        if (schema.includes('vega/v') && !schema.includes('vega-lite')) {
            return 'vega';
        }
        
        // Heuristic detection
        if (Array.isArray(spec.data)) {
            const hasNamedData = spec.data.some(d => d && d.name);
            const hasMarks = Array.isArray(spec.marks);
            const hasSignals = Array.isArray(spec.signals);
            if (hasNamedData || hasMarks || hasSignals) {
                return 'vega';
            }
        }
        
        if (spec.mark || spec.encoding || spec.layer || spec.hconcat || spec.vconcat || spec.concat || spec.facet || spec.repeat) {
            return 'vega-lite';
        }
        
        return 'vega-lite';
    },
    
    /**
     * Check if spec is a Sankey diagram
     */
    isSankeySpec(spec) {
        if (!Array.isArray(spec.data)) return false;
        return spec.data.some(d => d.name === 'nodes' || d.name === 'links');
    },
    
    /**
     * Main render function - simple vegaEmbed call
     */
    async render(spec) {
        const container = document.getElementById('vega-chart');
        if (!container) return;
        
        container.innerHTML = '<div class="loading">Loading chart...</div>';
        
        try {
            const specType = this.detectSpecType(spec);
            const isSankey = this.isSankeySpec(spec);
            
            Log.add('info', `Rendering ${specType}${isSankey ? ' (Sankey)' : ''}`);
            
            // 尝试添加brush（如果适合的话）
            let renderSpec = spec;
            let addedBrush = false;
            if (specType === 'vega-lite' && !isSankey && this.canAddBrush(spec)) {
                renderSpec = this.addBrushSelection(spec);
                addedBrush = true;
            }
            
            let result;
            try {
                result = await vegaEmbed(container, renderSpec, {
                    mode: specType,
                    actions: {
                        export: true,
                        source: false,
                        compiled: false,
                        editor: false
                    },
                    renderer: 'svg',
                    hover: true,
                    tooltip: true
                });
            } catch (e) {
                // 如果添加brush后渲染失败，尝试用原始spec重新渲染
                if (addedBrush && e.message?.includes('brush')) {
                    console.log('Brush conflict detected, retrying without brush');
                    Log.add('info', 'Retrying render without brush selection');
                    result = await vegaEmbed(container, spec, {
                        mode: specType,
                        actions: {
                            export: true,
                            source: false,
                            compiled: false,
                            editor: false
                        },
                        renderer: 'svg',
                        hover: true,
                        tooltip: true
                    });
                    addedBrush = false;
                } else {
                    throw e;
                }
            }
            
            State.vegaView = result.view;
            
            // 只有成功添加了brush才添加监听器
            if (addedBrush) {
                try {
                    result.view.addSignalListener('brush', (name, value) => {
                        Tools.handleBrush(value);
                    });
                } catch (e) {
                    // 忽略
                }
            }
            
            Log.add('info', 'Chart rendered successfully');
            
        } catch (e) {
            const errorMsg = e.message || String(e);
            console.error('Render error:', e);
            
            container.innerHTML = `<div class="error">
                <strong>Render Error:</strong> ${errorMsg}
            </div>`;
            Log.add('error', `Render: ${errorMsg}`);
        }
    },
    
    /**
     * 检测spec是否可以安全添加brush
     */
    canAddBrush(spec) {
        // 复杂结构不添加
        if (spec.layer || spec.hconcat || spec.vconcat || spec.concat || spec.repeat || spec.facet) {
            console.log('canAddBrush: false (complex structure)', {layer: !!spec.layer});
            return false;
        }
        
        // 没有基本encoding不添加
        if (!spec.encoding || !spec.encoding.x || !spec.encoding.y) {
            console.log('canAddBrush: false (no x/y encoding)');
            return false;
        }
        
        // 已有brush相关params不添加
        if (spec.params) {
            const hasBrush = spec.params.some(p => 
                p.name === 'brush' || 
                p.name?.includes('brush') ||
                p.select?.type === 'interval'
            );
            if (hasBrush) {
                console.log('canAddBrush: false (already has brush)');
                return false;
            }
        }
        
        // 只对scatter和简单line/bar添加
        const mark = typeof spec.mark === 'string' ? spec.mark : spec.mark?.type;
        const safeMarks = ['point', 'circle', 'square', 'bar', 'line', 'area'];
        if (!safeMarks.includes(mark)) {
            console.log('canAddBrush: false (unsupported mark type)', mark);
            return false;
        }
        
        console.log('canAddBrush: true');
        return true;
    },
    
    /**
     * 添加brush selection
     */
    addBrushSelection(spec) {
        const newSpec = JSON.parse(JSON.stringify(spec));
        
        if (!newSpec.params) {
            newSpec.params = [];
        }
        
        newSpec.params.push({
            name: 'brush',
            select: {
                type: 'interval',
                encodings: ['x', 'y']
            }
        });
        
        return newSpec;
    }
};

// App
const App = {
    async init() {
        Log.init();
        Log.add('info', 'Initializing...');
        
        try {
            const data = await API.getSpecs();
            State.specList = data.specs || [];
            Log.add('info', `Loaded ${State.specList.length} specs`);
            
            if (State.specList.length > 0) {
                await this.loadSpec(0);
            } else {
                Log.add('error', 'No specs available. Add JSON files to backend/specs/');
            }
        } catch (e) {
            Log.add('error', `Init failed: ${e.message}`);
        }
        
        this.setupEvents();
    },
    
    async loadSpec(index) {
        if (index < 0 || index >= State.specList.length) return;
        
        State.currentIndex = index;
        Log.add('info', `Loading spec ${index + 1}/${State.specList.length}`);
        
        try {
            const data = await API.getSpec(index);
            
            // 调试：检查收到的 spec 是否包含 _sankey_state
            if (data.spec && Array.isArray(data.spec.data) && data.spec.data.some(d => d.name === 'nodes' || d.name === 'links')) {
                const hasState = '_sankey_state' in data.spec;
                const stateKeys = hasState ? Object.keys(data.spec._sankey_state || {}) : [];
                console.log(`[DEBUG] loadSpec(${index}): received spec with _sankey_state=${hasState}, keys=${stateKeys.join(',')}`);
            }
            
            State.currentSpec = data.spec;
            State.originalSpec = JSON.parse(JSON.stringify(data.spec));
            State.chartMeta = data.meta || {};

            // Normalize chart_type to the frontend's tool buckets.
            // We prefer title keyword detection (simplified, per requirement),
            // and also normalize backend variants like "sankey_multi_stage" -> "sankey".
            const normalizeChartType = (rawType, spec, meta) => {
                const rt = String(rawType || '').toLowerCase();
                if (rt.includes('sankey')) return 'sankey';
                if (rt.includes('heatmap')) return 'heatmap';
                if (rt.includes('parallel')) return 'parallel';
                if (rt.includes('scatter')) return 'scatter';
                if (rt.includes('bar')) return 'bar';
                if (rt.includes('line')) return 'line';

                // Title-based fallback
                const titleObj = spec?.title;
                const specTitle =
                    (typeof titleObj === 'string' ? titleObj : (titleObj?.text || '')) ||
                    meta?.title ||
                    spec?.description ||
                    '';
                const t = String(specTitle).toLowerCase();
                if (t.includes('sankey')) return 'sankey';
                if (t.includes('heatmap') || t.includes('heat map')) return 'heatmap';
                if (t.includes('parallel')) return 'parallel';
                if (t.includes('scatter')) return 'scatter';
                if (t.includes('bar')) return 'bar';
                if (t.includes('line')) return 'line';

                return String(rawType || '').trim() || 'scatter';
            };

            State.chartType = normalizeChartType(data.meta?.chart_type, data.spec, data.meta);
            State.specHistory = [];
            State.iterations = [];
            
            // Store original_spec in vega_spec metadata
            if (!State.currentSpec._original_spec) {
                State.currentSpec._original_spec = JSON.parse(JSON.stringify(State.originalSpec));
            }
            if (!State.currentSpec._spec_history) {
                State.currentSpec._spec_history = [];
            }
            
            UI.updateMeta();
            UI.buildTools(State.chartType);
            UI.updateIterations();
            
            await Chart.render(data.spec);
        } catch (e) {
            Log.add('error', `Load failed: ${e.message}`);
        }
    },
    
    setupEvents() {
        document.getElementById('prev-btn')?.addEventListener('click', () => {
            if (State.currentIndex > 0) this.loadSpec(State.currentIndex - 1);
        });
        
        document.getElementById('next-btn')?.addEventListener('click', () => {
            if (State.currentIndex < State.specList.length - 1) this.loadSpec(State.currentIndex + 1);
        });
        
        const navInput = document.getElementById('nav-index-input');
        const navGo = document.getElementById('nav-go-btn');
        const doJump = () => {
            const n = parseInt(navInput?.value, 10);
            const total = State.specList?.length || 0;
            if (!Number.isFinite(n) || total === 0) return;
            const idx = Math.max(0, Math.min(n - 1, total - 1));
            this.loadSpec(idx);
            if (navInput) navInput.value = '';
        };
        navGo?.addEventListener('click', doJump);
        navInput?.addEventListener('keydown', (e) => { if (e.key === 'Enter') doJump(); });
        
        document.getElementById('question-input')?.addEventListener('input', e => {
            State.currentQuestion = e.target.value;
        });
        
        document.getElementById('answer-input')?.addEventListener('input', e => {
            State.currentAnswer = e.target.value;
        });
        
        // Answer Type 选择和配置
        const answerTypeSelect = document.getElementById('answer-type-select');
        answerTypeSelect?.addEventListener('change', e => {
            State.currentAnswerType = e.target.value;
            this.updateAnswerTypeConfig(e.target.value);
        });
        
        // 各类型配置输入
        document.getElementById('alternatives-input')?.addEventListener('input', e => {
            const alternatives = e.target.value.split(',').map(s => s.trim()).filter(s => s);
            State.currentAnswerConfig.alternatives = alternatives;
        });
        
        document.getElementById('tolerance-input')?.addEventListener('input', e => {
            State.currentAnswerConfig.tolerance = parseFloat(e.target.value) || 0.05;
        });
        
        document.getElementById('region-threshold-input')?.addEventListener('input', e => {
            State.currentAnswerConfig.threshold = parseFloat(e.target.value) || 0.5;
        });
        
        document.getElementById('finish-btn')?.addEventListener('click', () => this.finish());
        document.getElementById('export-btn')?.addEventListener('click', () => this.export());
        document.getElementById('export-task-btn')?.addEventListener('click', () => this.exportTask());
    },
    
    updateAnswerTypeConfig(type) {
        // 隐藏所有配置
        document.querySelectorAll('.type-config').forEach(el => el.style.display = 'none');
        
        // 显示对应类型的配置
        const configId = `config-${type}`;
        const configEl = document.getElementById(configId);
        if (configEl) {
            configEl.style.display = 'block';
        }
        
        // 重置配置
        State.currentAnswerConfig = {};
    },
    
    async finish() {
        if (!State.currentQuestion) {
            UI.showStatus('Enter a question first', 'error');
            return;
        }
        if (!State.iterations.length) {
            UI.showStatus('Execute at least one tool', 'error');
            return;
        }
        
        const taskType = document.getElementById('task-type-select')?.value || 'clear_single';
        const answerType = document.getElementById('answer-type-select')?.value || 'categorical';
        
        // 收集 answer_config
        let answerConfig = {};
        if (answerType === 'categorical') {
            const altInput = document.getElementById('alternatives-input')?.value || '';
            answerConfig.alternatives = altInput.split(',').map(s => s.trim()).filter(s => s);
        } else if (answerType === 'numeric') {
            answerConfig.tolerance = parseFloat(document.getElementById('tolerance-input')?.value) || 0.05;
        } else if (answerType === 'region') {
            answerConfig.threshold = parseFloat(document.getElementById('region-threshold-input')?.value) || 0.5;
            answerConfig.metric = 'iou';
        }
        
        try {
            // 确保 meta 包含 index
            const metaWithIndex = {
                ...State.chartMeta,
                index: State.currentIndex
            };
            
            const result = await API.finish({
                question: State.currentQuestion,
                answer: State.currentAnswer || null,
                answer_type: answerType,
                answer_config: answerConfig,
                iterations: State.iterations,
                final_spec: State.currentSpec,
                original_spec: State.originalSpec,
                chart_type: State.chartType,
                task_type: taskType,
                meta: metaWithIndex
            });
            
            Log.add('result', 'Benchmark created', result.benchmark);
            State.completedBenchmarks.push(result.benchmark);
            
            UI.showStatus('Question completed!', 'success');
            
            // Reset
            State.currentQuestion = '';
            State.currentAnswer = '';
            State.currentAnswerType = 'categorical';
            State.currentAnswerConfig = {};
            State.iterations = [];
            document.getElementById('question-input').value = '';
            document.getElementById('answer-input').value = '';
            document.getElementById('answer-type-select').value = 'categorical';
            document.getElementById('alternatives-input').value = '';
            this.updateAnswerTypeConfig('categorical');
            UI.updateIterations();
            
            State.currentSpec = JSON.parse(JSON.stringify(State.originalSpec));
            await Chart.render(State.currentSpec);
        } catch (e) {
            Log.add('error', e.message);
            UI.showStatus(e.message, 'error');
        }
    },
    
    export() {
        if (!State.completedBenchmarks.length) {
            UI.showStatus('No benchmarks to export', 'error');
            return;
        }
        
        const blob = new Blob([JSON.stringify({ 
            benchmarks: State.completedBenchmarks, 
            exported_at: new Date().toISOString() 
        }, null, 2)], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `benchmarks_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(a.href);
        
        Log.add('info', `Exported ${State.completedBenchmarks.length} benchmarks`);
    },
    
    exportTask() {
        if (!State.completedBenchmarks.length) {
            UI.showStatus('No benchmarks to export', 'error');
            return;
        }
        
        // eval_weights 映射（根据 task_type 自动生成）
        const evalWeightsMap = {
            'clear_single': { 'tool_call': 0.8, 'final_state': 0.2 },
            'clear_multi': { 'tool_call': 0.7, 'final_state': 0.3 },
            'vague_single': { 'tool_call': 0.3, 'final_state': 0.7 },
            'vague_multi': { 'tool_call': 0.2, 'final_state': 0.8 }
        };
        
        // 一题一 JSON：每个 benchmark 单独导出为一个 task 文件
        const taskTypeAbbrev = { clear_single: 'cs', clear_multi: 'cm', vague_single: 'vs', vague_multi: 'vm' };
        const tasks = [];
        const chartSeqCount = {};  // 同 chart 下的题目序号
        
        for (const benchmark of State.completedBenchmarks) {
            const chartType = benchmark.chart_type || 'unknown';
            const meta = benchmark.meta || {};
            const filename = meta.filename || '';
            const chartIndex = meta.index !== undefined ? meta.index : null;
            const taskType = benchmark.task_type || 'clear_single';
            const typeAbbrev = taskTypeAbbrev[taskType] || 'qx';
            
            // chart 基础 id：序号_图表类型（如 "12_sankey"）
            let chartId;
            if (chartIndex !== null) {
                chartId = `${String(chartIndex + 1).padStart(2, '0')}_${chartType}`;
            } else {
                const baseName = filename.replace(/\.json$/, '');
                chartId = baseName || `${chartType}_${Date.now()}`;
            }
            
            chartSeqCount[chartId] = (chartSeqCount[chartId] || 0) + 1;
            const seq = chartSeqCount[chartId];
            
            // task_id 含问题类型：如 12_sankey_cs_01, 12_sankey_vm_02
            const taskId = `${chartId}_${typeAbbrev}_${String(seq).padStart(2, '0')}`;
            
            const vegaSpecPath = `benchmark_annotation_system/backend/specs/${filename}`;
            const evalWeights = evalWeightsMap[taskType] || { 'tool_call': 0.6, 'final_state': 0.4 };
            
            tasks.push({
                task_type: taskType,
                task_id: taskId,
                vega_spec_path: vegaSpecPath,
                eval_weights: evalWeights,
                questions: benchmark.questions || []
            });
        }
        
        if (tasks.length === 0) {
            UI.showStatus('No valid tasks to export', 'error');
            return;
        }
        
        // 逐个下载（一题一文件），避免浏览器拦截
        const delay = (ms) => new Promise(r => setTimeout(r, ms));
        (async () => {
            for (let i = 0; i < tasks.length; i++) {
                const task = tasks[i];
                const blob = new Blob([JSON.stringify(task, null, 2)], { type: 'application/json' });
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = `${task.task_id}.json`;
                a.click();
                URL.revokeObjectURL(a.href);
                Log.add('info', `Exported task: ${task.task_id}`);
                if (i < tasks.length - 1) await delay(300);
            }
            
            // 导出后自动刷新：清空已完成列表
            State.completedBenchmarks = [];
            UI.updateIterations?.();
            UI.showStatus(`Exported ${tasks.length} task(s), list cleared`, 'success');
        })();
    }
};

document.addEventListener('DOMContentLoaded', () => {
    // 初始化UI组件
    UIComponents.init();
    
    // 初始化应用
    App.init();
});