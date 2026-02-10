/**
 * Tool Configuration - Matching actual tool_registry.py
 * 
 * Note: vega_spec is automatically added by the backend, 
 * so we only define user-facing parameters here.
 */

const TOOL_CONFIG = {
    // Common tools (all chart types)
    common: {
        get_view_spec: { 
            description: '获取视图结构状态', 
            params: {}, 
            interaction: 'button' 
        },
        get_data: { 
            description: '获取原始数据', 
            params: { 
                scope: { type: 'select', options: ['all', 'filter', 'visible', 'selected'], default: 'all' } 
            }, 
            interaction: 'button' 
        },
        get_data_summary: { 
            description: '获取数据统计摘要', 
            params: { 
                scope: { type: 'select', options: ['all', 'filter', 'visible'], default: 'all' } 
            }, 
            interaction: 'button' 
        },
        reset_view: { 
            description: '重置视图到原始状态', 
            params: {}, 
            interaction: 'button' 
        },
        undo_view: { 
            description: '撤销上一步视图', 
            params: {}, 
            interaction: 'button' 
        }
    },

    // Scatter plot tools
    scatter: {
        zoom_dense_area: {
            description: '放大密集区域 (在图表上刷选)',
            params: { 
                x_range: { type: 'brush', required: true }, 
                y_range: { type: 'brush', required: true } 
            },
            interaction: 'brush'
        },
        select_region: {
            description: '选择区域',
            params: { 
                x_range: { type: 'brush', required: true }, 
                y_range: { type: 'brush', required: true } 
            },
            interaction: 'brush'
        },
        brush_region: {
            description: '刷选区域，区域外变淡',
            params: { 
                x_range: { type: 'brush', required: true }, 
                y_range: { type: 'brush', required: true } 
            },
            interaction: 'brush'
        },
        filter_categorical: {
            description: '过滤掉指定类别的数据点',
            params: { 
                categories_to_remove: { type: 'multi-select', source: 'all_categories', required: true },
                field: { type: 'select', source: 'categorical_fields', required: false }
            },
            interaction: 'select'
        },
        identify_clusters: {
            description: '识别聚类',
            params: { 
                n_clusters: { type: 'number', default: 3, min: 2 },
                method: { type: 'select', options: ['kmeans', 'dbscan'], default: 'kmeans' }
            },
            interaction: 'button'
        },
        calculate_correlation: {
            description: '计算相关性',
            params: { 
                method: { type: 'select', options: ['pearson', 'spearman'], default: 'pearson' } 
            },
            interaction: 'button'
        },
        show_regression: {
            description: '叠加回归线',
            params: { 
                method: { type: 'select', options: ['linear', 'log', 'exp', 'poly', 'quad'], default: 'linear' } 
            },
            interaction: 'button'
        },
        change_encoding: {
            description: '修改编码通道的字段映射',
            params: {
                channel: { type: 'select', options: ['color', 'size', 'shape', 'opacity', 'x', 'y'], required: true },
                field: { type: 'select', source: 'all_fields', required: true }
            },
            interaction: 'button'
        }
    },

    // Bar chart tools
    bar: {
        sort_bars: {
            description: '按值排序条形。堆叠图：传 by_subcategory 按该子类值排 x 轴；不传则按子类总值排堆叠层。',
            params: { 
                order: { type: 'select', options: ['ascending', 'descending'], default: 'descending' },
                by_subcategory: { type: 'select', source: 'color_categories', required: false, hint: '堆叠图：指定子类名则按该子类排序 x 轴；留空则排序堆叠层' }
            },
            interaction: 'button'
        },
        filter_categories: {
            description: '过滤类别（所选类别将被保留）',
            params: { 
                categories: { 
                    type: 'multi-select', 
                    source: 'x_categories', 
                    required: true,
                    hint: '选中的类别将被过滤，未选中的将被保留'
                } 
            },
            interaction: 'select'
        },
        filter_subcategories: {
            description: '过滤子类别（color/xOffset 编码的分组）',
            params: { 
                subcategories_to_remove: { 
                    type: 'multi-select', 
                    source: 'color_categories', 
                    required: true,
                    hint: '选择要移除的子类别（如 Chest pain type 1, 2, 3）'
                },
                sub_field: { 
                    type: 'select', 
                    source: 'categorical_fields', 
                    required: false 
                }
            },
            interaction: 'select'
        },
        highlight_top_n: {
            description: '高亮前N个条形',
            params: { 
                n: { type: 'number', default: 2, min: 1 },
                order: { type: 'select', options: ['ascending', 'descending'], default: 'descending' }
            },
            interaction: 'button'
        },
        toggle_stack_mode: {
            description: '切换堆叠/分组模式',
            params: { 
                mode: { type: 'select', options: ['grouped', 'stacked'], required: true } 
            },
            interaction: 'button'
        },
        expand_stack: {
            description: '展开某类别的堆叠为平排',
            params: { 
                category: { type: 'select', source: 'x_categories', required: true } 
            },
            interaction: 'select'
        },
        add_bars: {
            description: '增加整根条形',
            params: { 
                values: { type: 'multi-select', source: 'x_categories', required: true },
                x_field: { type: 'select', source: 'all_fields', required: false }
            },
            interaction: 'select'
        },
        remove_bars: {
            description: '减少整根条形',
            params: { 
                values: { type: 'multi-select', source: 'x_categories', required: true },
                x_field: { type: 'select', source: 'all_fields', required: false }
            },
            interaction: 'select'
        },
        change_encoding: {
            description: '修改编码通道的字段映射',
            params: {
                channel: { type: 'select', options: ['color', 'size', 'opacity', 'x', 'y'], required: true },
                field: { type: 'select', source: 'all_fields', required: true }
            },
            interaction: 'button'
        }
    },

    // Line chart tools
    line: {
        zoom_time_range: {
            description: '缩放时间范围',
            params: { 
                start: { type: 'select', source: 'time_values', required: true, placeholder: '开始时间' },
                end: { type: 'select', source: 'time_values', required: true, placeholder: '结束时间' }
            },
            interaction: 'select'
        },
        highlight_trend: {
            description: '高亮趋势',
            params: { 
                trend_type: { type: 'select', options: ['increasing', 'decreasing', 'stable'], default: 'increasing' } 
            },
            interaction: 'button'
        },
        detect_anomalies: {
            description: '检测异常点',
            params: { 
                threshold: { type: 'number', default: 2.0, step: 0.1 } 
            },
            interaction: 'button'
        },
        bold_lines: {
            description: '加粗指定的折线',
            params: { 
                line_names: { type: 'multi-select', source: 'series_or_categorical', required: true }
            },
            interaction: 'select'
        },
        filter_lines: {
            description: '过滤折线（所选系列将被移除）',
            params: { 
                lines_to_remove: { type: 'multi-select', source: 'series_or_categorical', required: true }
            },
            interaction: 'select'
        },
        focus_lines: {
            description: '聚焦指定系列，其余变暗',
            params: { 
                lines: { type: 'multi-select', source: 'series_or_categorical', required: true },
                mode: { type: 'select', options: ['dim', 'hide'], default: 'dim' }
            },
            interaction: 'select'
        },
        show_moving_average: {
            description: '叠加移动平均线',
            params: { 
                window_size: { type: 'number', default: 3, min: 2 } 
            },
            interaction: 'button'
        },
        resample_time: {
            description: '时间粒度切换',
            params: {
                granularity: { type: 'select', options: ['day', 'week', 'month', 'quarter', 'year'], required: true },
                agg: { type: 'select', options: ['mean', 'sum', 'max', 'min', 'median'], default: 'mean' }
            },
            interaction: 'button'
        },
        reset_resample: {
            description: '重置时间重采样',
            params: {},
            interaction: 'button'
        },
        drilldown_line_time: {
            description: '时间下钻（年→月→日）',
            params: {
                level: { type: 'select', options: ['year', 'month', 'date'], default: 'year', required: true },
                value: { type: 'select', source: 'drilldown_years', required: true },
                parent: { type: 'text', required: false, placeholder: '{"year": 2023} 等' }
            },
            interaction: 'select'
        },
        reset_line_drilldown: {
            description: '重置时间下钻',
            params: {},
            interaction: 'button'
        },
        change_encoding: {
            description: '修改编码通道的字段映射',
            params: {
                channel: { type: 'select', options: ['color', 'size', 'opacity', 'x', 'y'], required: true },
                field: { type: 'select', source: 'all_fields', required: true }
            },
            interaction: 'button'
        }
    },

    // Heatmap tools
    heatmap: {
        adjust_color_scale: {
            description: '调整颜色比例和范围',
            params: { 
                scheme: { type: 'select', options: ['viridis', 'blues', 'reds', 'greens', 'oranges', 'purples', 'inferno', 'magma'], default: 'viridis' }
            },
            interaction: 'button'
        },
        filter_cells: {
            description: '筛选单元格（按值范围，⚠️ 会过滤底层数据）。min 或 max 至少填一个即可',
            params: { 
                min_value: { type: 'number', required: false, hint: '下界（含）；只填此项保留 >= min' },
                max_value: { type: 'number', required: false, hint: '上界（含）；只填此项保留 <= max' }
            },
            interaction: 'button'
        },
        filter_cells_by_region: {
            description: '按格子坐标过滤聚合后的格子（⚠️ 会过滤底层数据）。x 或 y 至少选一个',
            params: {
                x_value: { type: 'select', source: 'x_categories', required: false, hint: '选择要过滤的列（x）；x 或 y 至少选一个' },
                y_value: { type: 'select', source: 'y_categories', required: false, hint: '选择要过滤的行（y）；x 或 y 至少选一个' }
            },
            interaction: 'select'
        },
        highlight_region: {
            description: '高亮特定区域。x 或 y 至少选一个（仅 x 高亮整列，仅 y 高亮整行）',
            params: {
                x_values: { type: 'multi-select', source: 'x_categories', required: false, hint: 'X 轴要高亮的值；x 或 y 至少选一个' },
                y_values: { type: 'multi-select', source: 'y_categories', required: false, hint: 'Y 轴要高亮的值；x 或 y 至少选一个' }
            },
            interaction: 'select'
        },
        highlight_region_by_value: {
            description: '按格子显示值高亮（范围外变淡，不删除数据；支持单侧阈值）',
            params: {
                min_value: { type: 'number', required: false, hint: '下阈值（可选）；只填一个也可以' },
                max_value: { type: 'number', required: false, hint: '上阈值（可选）' },
                outside_opacity: { type: 'number', default: 0.12, step: 0.01, min: 0, max: 1 }
            },
            interaction: 'button'
        },
        cluster_rows_cols: {
            description: '按行/列聚合重排（行=Y 轴，列=X 轴；按 color 的 sum/mean/max 降序）',
            params: { 
                cluster_rows: { type: 'checkbox', default: true, hint: '对行（Y）按聚合排序' },
                cluster_cols: { type: 'checkbox', default: true, hint: '对列（X）按聚合排序' },
                method: { type: 'select', options: ['sum', 'mean', 'max'], default: 'sum' }
            },
            interaction: 'button'
        },
        find_extremes: {
            description: '寻找极值单元格',
            params: { 
                top_n: { type: 'number', default: 5, min: 1 },
                mode: { type: 'select', options: ['max', 'min', 'both'], default: 'both' }
            },
            interaction: 'button'
        },
        transpose: { 
            description: '转置热力图', 
            params: {}, 
            interaction: 'button' 
        },
        add_marginal_bars: { 
            description: '添加边际条形图（顶部按列聚合、右侧按行聚合，默认 mean；与主图共享轴）', 
            params: { 
                op: { type: 'select', options: ['mean', 'sum', 'median', 'max', 'min'], default: 'mean' },
                show_top: { type: 'checkbox', default: true },
                show_right: { type: 'checkbox', default: true }
            }, 
            interaction: 'button' 
        },
        drilldown_time: {
            description: '按时间字段下钻（年→月→日）',
            params: {
                level: { type: 'select', options: ['year', 'month', 'date'], default: 'year', required: true },
                value: { type: 'select', source: 'drilldown_years', required: true },
                parent: { type: 'text', required: false, placeholder: '{"year": 2023} 或 {"year":2023,"month":5}' }
            },
            interaction: 'select'
        },
        reset_drilldown: {
            description: '重置时间下钻',
            params: {},
            interaction: 'button'
        },
        change_encoding: {
            description: '修改编码通道的字段映射',
            params: {
                channel: { type: 'select', options: ['color', 'size', 'opacity', 'x', 'y'], required: true },
                field: { type: 'select', source: 'all_fields', required: true }
            },
            interaction: 'button'
        }
    },

    // Parallel coordinates tools
    parallel: {
        reorder_dimensions: {
            description: '重排维度顺序',
            params: { 
                dimension_order: { type: 'sortable-list', source: 'dimensions', required: true } 
            },
            interaction: 'drag'
        },
        filter_by_category: {
            description: '按分类字段筛选',
            params: {
                field: { type: 'select', source: 'categorical_fields', required: true },
                values: { type: 'multi-select', source: 'category_values', required: true }
            },
            interaction: 'select'
        },
        highlight_category: {
            description: '高亮特定类别',
            params: {
                field: { type: 'select', source: 'categorical_fields', required: true },
                values: { type: 'multi-select', source: 'category_values', required: true }
            },
            interaction: 'select'
        },
        hide_dimensions: {
            description: '隐藏/显示维度',
            params: { 
                dimensions: { type: 'multi-select', source: 'dimensions', required: true },
                mode: { type: 'select', options: ['hide', 'show'], default: 'hide' }
            },
            interaction: 'select'
        },
        reset_hidden_dimensions: {
            description: '重置隐藏的维度',
            params: {},
            interaction: 'button'
        },
        change_encoding: {
            description: '修改编码通道的字段映射',
            params: {
                channel: { type: 'select', options: ['color', 'size', 'opacity', 'x', 'y'], required: true },
                field: { type: 'select', source: 'all_fields', required: true }
            },
            interaction: 'button'
        }
    },

    // Sankey diagram tools
    sankey: {
        filter_flow: {
            description: '筛选流量（大于阈值）',
            params: { 
                min_value: { type: 'number', required: true } 
            },
            interaction: 'button'
        },
        highlight_path: {
            description: '高亮多步路径',
            params: { 
                path: { type: 'path-builder', source: 'nodes', required: true } 
            },
            interaction: 'text'
        },
        trace_node: {
            description: '追踪节点（高亮相连的流）',
            params: { 
                node_name: { type: 'select', source: 'nodes', required: true } 
            },
            interaction: 'select'
        },
        collapse_nodes: {
            description: '折叠节点为聚合节点',
            params: { 
                nodes_to_collapse: { type: 'multi-select', source: 'nodes', required: true },
                aggregate_name: { type: 'text', default: 'Other', placeholder: '聚合节点名称' }
            },
            interaction: 'select'
        },
        expand_node: {
            description: '展开聚合节点',
            params: { 
                aggregate_name: { type: 'select', source: 'aggregate_nodes', required: true } 
            },
            interaction: 'select'
        },
        auto_collapse_by_rank: {
            description: '按排名自动折叠（每层保留TopN）',
            params: { 
                top_n: { type: 'number', default: 5, min: 1 } 
            },
            interaction: 'button'
        },
        color_flows: {
            description: '给与指定节点相连的流着色',
            params: {
                nodes: { type: 'multi-select', source: 'nodes', required: true },
                color: { type: 'text', default: '#e74c3c', placeholder: '颜色值' }
            },
            interaction: 'select'
        },
        find_bottleneck: {
            description: '识别瓶颈节点',
            params: { 
                top_n: { type: 'number', default: 3, min: 1 } 
            },
            interaction: 'button'
        },
        calculate_conversion_rate: {
            description: '计算转化率',
            params: {
                node_name: { type: 'select', source: 'nodes', required: false }
            },
            interaction: 'button'
        },
        reorder_nodes_in_layer: {
            description: '重排某层节点顺序',
            params: {
                depth: { type: 'number', required: true, min: 0, placeholder: '层号(0=第一层)' },
                sort_by: { type: 'select', options: ['value_desc', 'value_asc', 'name'], required: false }
            },
            interaction: 'button'
        },
        change_encoding: {
            description: '修改编码通道的字段映射（Sankey 为 Vega 图，仅支持有限字段）',
            params: {
                channel: { type: 'select', options: ['color', 'size', 'opacity', 'x', 'y'], required: true },
                field: { type: 'select', source: 'sankey_fields', required: true }
            },
            interaction: 'button'
        }
    }
};


/**
 * Get tools for a specific chart type
 */
function getToolsForChartType(chartType) {
    const t = (chartType || '').toLowerCase().replace(/[_\s-]/g, '');
    const map = {
        'bar': 'bar', 'barchart': 'bar',
        'line': 'line', 'linechart': 'line',
        'scatter': 'scatter', 'scatterplot': 'scatter',
        'heatmap': 'heatmap',
        'parallel': 'parallel', 'parallelcoordinates': 'parallel',
        'sankey': 'sankey', 'sankeydiagram': 'sankey'
    };
    return { 
        common: TOOL_CONFIG.common, 
        specific: TOOL_CONFIG[map[t]] || TOOL_CONFIG.scatter 
    };
}