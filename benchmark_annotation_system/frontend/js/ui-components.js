/**
 * UI Components - 标准化的表单组件
 * 参考 Ant Design / Element UI 风格
 */

const UIComponents = {
    /**
     * 创建单选下拉框
     */
    createSelect(name, options, config = {}) {
        const wrapper = document.createElement('div');
        wrapper.className = 'ui-select';
        wrapper.dataset.name = name;
        
        // 显示区域
        const display = document.createElement('div');
        display.className = 'ui-select-display';
        
        const displayText = document.createElement('span');
        displayText.className = 'ui-select-text';
        displayText.textContent = config.placeholder || '请选择';
        
        const arrow = document.createElement('span');
        arrow.className = 'ui-select-arrow';
        arrow.innerHTML = '▼';
        
        display.appendChild(displayText);
        display.appendChild(arrow);
        
        // 下拉面板
        const dropdown = document.createElement('div');
        dropdown.className = 'ui-select-dropdown';
        
        // 选项列表
        options.forEach(opt => {
            const item = document.createElement('div');
            item.className = 'ui-select-option';
            item.dataset.value = opt;
            item.textContent = opt;
            
            item.addEventListener('click', (e) => {
                e.stopPropagation();
                // 更新选中状态
                dropdown.querySelectorAll('.ui-select-option').forEach(o => o.classList.remove('selected'));
                item.classList.add('selected');
                displayText.textContent = opt;
                wrapper.dataset.value = opt;
                wrapper.classList.remove('open');
                wrapper.classList.add('has-value');
            });
            
            if (config.default === opt) {
                item.classList.add('selected');
                displayText.textContent = opt;
                wrapper.dataset.value = opt;
                wrapper.classList.add('has-value');
            }
            
            dropdown.appendChild(item);
        });
        
        wrapper.appendChild(display);
        wrapper.appendChild(dropdown);
        
        // 点击展开/收起
        display.addEventListener('click', (e) => {
            e.stopPropagation();
            // 关闭其他下拉
            document.querySelectorAll('.ui-select.open, .ui-multi-select.open').forEach(s => {
                if (s !== wrapper) s.classList.remove('open');
            });
            wrapper.classList.toggle('open');
        });
        
        // 隐藏的input用于表单收集
        const hiddenInput = document.createElement('input');
        hiddenInput.type = 'hidden';
        hiddenInput.name = name;
        wrapper.appendChild(hiddenInput);
        
        // 监听dataset变化更新hidden input
        const observer = new MutationObserver(() => {
            hiddenInput.value = wrapper.dataset.value || '';
        });
        observer.observe(wrapper, { attributes: true, attributeFilter: ['data-value'] });
        
        return wrapper;
    },
    
    /**
     * 创建多选下拉框
     */
    createMultiSelect(name, options, config = {}) {
        const wrapper = document.createElement('div');
        wrapper.className = 'ui-multi-select';
        wrapper.dataset.name = name;
        
        // 显示区域
        const display = document.createElement('div');
        display.className = 'ui-select-display';
        
        const tagsContainer = document.createElement('div');
        tagsContainer.className = 'ui-select-tags';
        
        const placeholder = document.createElement('span');
        placeholder.className = 'ui-select-placeholder';
        placeholder.textContent = config.placeholder || '请选择';
        tagsContainer.appendChild(placeholder);
        
        const arrow = document.createElement('span');
        arrow.className = 'ui-select-arrow';
        arrow.innerHTML = '▼';
        
        display.appendChild(tagsContainer);
        display.appendChild(arrow);
        
        // 下拉面板
        const dropdown = document.createElement('div');
        dropdown.className = 'ui-select-dropdown';
        
        // 已选值集合
        const selectedValues = new Set();
        
        // 更新显示
        const updateDisplay = () => {
            // 清除旧tags
            tagsContainer.querySelectorAll('.ui-tag').forEach(t => t.remove());
            
            if (selectedValues.size === 0) {
                placeholder.style.display = '';
            } else {
                placeholder.style.display = 'none';
                selectedValues.forEach(val => {
                    const tag = document.createElement('span');
                    tag.className = 'ui-tag';
                    tag.innerHTML = `${val}<span class="ui-tag-close">×</span>`;
                    tag.querySelector('.ui-tag-close').addEventListener('click', (e) => {
                        e.stopPropagation();
                        selectedValues.delete(val);
                        dropdown.querySelector(`[data-value="${val}"]`)?.classList.remove('selected');
                        updateDisplay();
                    });
                    tagsContainer.insertBefore(tag, placeholder);
                });
            }
            wrapper.dataset.value = JSON.stringify([...selectedValues]);
        };
        
        // 选项列表
        options.forEach(opt => {
            const item = document.createElement('div');
            item.className = 'ui-select-option';
            item.dataset.value = opt;
            
            const checkbox = document.createElement('span');
            checkbox.className = 'ui-checkbox';
            
            const text = document.createElement('span');
            text.textContent = opt;
            
            item.appendChild(checkbox);
            item.appendChild(text);
            
            item.addEventListener('click', (e) => {
                e.stopPropagation();
                if (selectedValues.has(opt)) {
                    selectedValues.delete(opt);
                    item.classList.remove('selected');
                } else {
                    selectedValues.add(opt);
                    item.classList.add('selected');
                }
                updateDisplay();
            });
            
            dropdown.appendChild(item);
        });
        
        wrapper.appendChild(display);
        wrapper.appendChild(dropdown);
        
        // 点击展开/收起
        display.addEventListener('click', (e) => {
            if (e.target.classList.contains('ui-tag-close')) return;
            e.stopPropagation();
            document.querySelectorAll('.ui-select.open, .ui-multi-select.open').forEach(s => {
                if (s !== wrapper) s.classList.remove('open');
            });
            wrapper.classList.toggle('open');
        });
        
        return wrapper;
    },
    
    /**
     * 创建数字输入框
     */
    createNumberInput(name, config = {}) {
        const wrapper = document.createElement('div');
        wrapper.className = 'ui-input-wrapper';
        
        const input = document.createElement('input');
        input.type = 'number';
        input.name = name;
        input.className = 'ui-input';
        if (config.default !== undefined) input.value = config.default;
        if (config.min !== undefined) input.min = config.min;
        if (config.max !== undefined) input.max = config.max;
        if (config.step) input.step = config.step;
        if (config.placeholder) input.placeholder = config.placeholder;
        
        wrapper.appendChild(input);
        return wrapper;
    },
    
    /**
     * 创建文本输入框
     */
    createTextInput(name, config = {}) {
        const wrapper = document.createElement('div');
        wrapper.className = 'ui-input-wrapper';
        
        const input = document.createElement('input');
        input.type = 'text';
        input.name = name;
        input.className = 'ui-input';
        if (config.placeholder) input.placeholder = config.placeholder;
        if (config.default) input.value = config.default;
        
        wrapper.appendChild(input);
        return wrapper;
    },
    
    /**
     * 创建开关/复选框
     */
    createCheckbox(name, config = {}) {
        const wrapper = document.createElement('label');
        wrapper.className = 'ui-checkbox-wrapper';
        
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.name = name;
        input.checked = config.default !== false;
        
        const indicator = document.createElement('span');
        indicator.className = 'ui-checkbox-indicator';
        
        wrapper.appendChild(input);
        wrapper.appendChild(indicator);
        
        return wrapper;
    },
    
    /**
     * 创建范围输入
     */
    createRangeInput(name, config = {}) {
        const wrapper = document.createElement('div');
        wrapper.className = 'ui-range-input';
        
        const minInput = document.createElement('input');
        minInput.type = 'number';
        minInput.name = name + '_min';
        minInput.className = 'ui-input';
        minInput.placeholder = 'Min';
        minInput.step = 'any';
        
        const separator = document.createElement('span');
        separator.className = 'ui-range-separator';
        separator.textContent = '~';
        
        const maxInput = document.createElement('input');
        maxInput.type = 'number';
        maxInput.name = name + '_max';
        maxInput.className = 'ui-input';
        maxInput.placeholder = 'Max';
        maxInput.step = 'any';
        
        wrapper.appendChild(minInput);
        wrapper.appendChild(separator);
        wrapper.appendChild(maxInput);
        
        return wrapper;
    },
    
    /**
     * 创建可排序列表
     */
    createSortableList(name, options, config = {}) {
        const wrapper = document.createElement('div');
        wrapper.className = 'ui-sortable-list';
        wrapper.dataset.name = name;
        
        const list = document.createElement('ul');
        list.id = `sortable-list-${name}`;
        
        options.forEach((item, idx) => {
            const li = document.createElement('li');
            li.className = 'ui-sortable-item';
            li.dataset.value = item;
            li.draggable = true;
            li.innerHTML = `
                <span class="ui-sortable-handle">⋮⋮</span>
                <span class="ui-sortable-index">${idx + 1}</span>
                <span class="ui-sortable-text">${item}</span>
            `;
            
            // 拖拽事件
            li.addEventListener('dragstart', (e) => {
                li.classList.add('dragging');
                e.dataTransfer.effectAllowed = 'move';
            });
            
            li.addEventListener('dragend', () => {
                li.classList.remove('dragging');
                // 更新序号
                list.querySelectorAll('.ui-sortable-item').forEach((item, i) => {
                    item.querySelector('.ui-sortable-index').textContent = i + 1;
                });
            });
            
            li.addEventListener('dragover', (e) => {
                e.preventDefault();
                const dragging = list.querySelector('.dragging');
                if (dragging && dragging !== li) {
                    const rect = li.getBoundingClientRect();
                    const midY = rect.top + rect.height / 2;
                    if (e.clientY < midY) {
                        list.insertBefore(dragging, li);
                    } else {
                        list.insertBefore(dragging, li.nextSibling);
                    }
                }
            });
            
            list.appendChild(li);
        });
        
        wrapper.appendChild(list);
        
        const hint = document.createElement('div');
        hint.className = 'ui-hint';
        hint.textContent = '拖拽调整顺序';
        wrapper.appendChild(hint);
        
        return wrapper;
    },
    
    /**
     * 创建Brush显示区域
     */
    createBrushDisplay(name, config = {}) {
        const wrapper = document.createElement('div');
        wrapper.className = 'ui-brush-display';
        wrapper.id = `brush-${name}`;
        wrapper.innerHTML = `
            <span class="ui-brush-icon">⬚</span>
            <span class="ui-brush-text">在图表上拖拽选择区域</span>
        `;
        return wrapper;
    },
    
    /**
     * 创建路径构建器（有序多选，用于 highlight_path）
     */
    createPathBuilder(name, options, config = {}) {
        const wrapper = document.createElement('div');
        wrapper.className = 'ui-path-builder';
        wrapper.dataset.name = name;
        
        // 已选路径列表
        const list = document.createElement('ul');
        list.className = 'ui-path-list';
        list.id = `path-list-${name}`;
        
        // 添加节点按钮和选择器
        const addSection = document.createElement('div');
        addSection.className = 'ui-path-add-section';
        
        const addBtn = document.createElement('button');
        addBtn.type = 'button';
        addBtn.className = 'ui-path-add-btn';
        addBtn.textContent = '+ 添加节点到路径';
        
        // 节点选择下拉（初始隐藏）
        const selectWrapper = document.createElement('div');
        selectWrapper.className = 'ui-path-select-wrapper';
        selectWrapper.style.display = 'none';
        
        const selectDropdown = document.createElement('div');
        selectDropdown.className = 'ui-path-select-dropdown';
        
        // 过滤已选节点
        const getAvailableNodes = () => {
            const selected = Array.from(list.querySelectorAll('.ui-path-item')).map(li => li.dataset.value);
            return options.filter(opt => !selected.includes(opt));
        };
        
        const updateDropdown = () => {
            selectDropdown.innerHTML = '';
            const available = getAvailableNodes();
            if (available.length === 0) {
                const empty = document.createElement('div');
                empty.className = 'ui-path-select-empty';
                empty.textContent = '所有节点已添加';
                selectDropdown.appendChild(empty);
            } else {
                available.forEach(opt => {
                    const item = document.createElement('div');
                    item.className = 'ui-path-select-option';
                    item.textContent = opt;
                    item.addEventListener('click', () => {
                        addNodeToPath(opt);
                        selectWrapper.style.display = 'none';
                    });
                    selectDropdown.appendChild(item);
                });
            }
        };
        
        const addNodeToPath = (nodeValue) => {
            const li = document.createElement('li');
            li.className = 'ui-path-item';
            li.dataset.value = nodeValue;
            
            const index = list.children.length + 1;
            li.innerHTML = `
                <span class="ui-path-index">${index}</span>
                <span class="ui-path-text">${nodeValue}</span>
                <button type="button" class="ui-path-remove" title="删除">×</button>
            `;
            
            // 删除按钮
            const removeBtn = li.querySelector('.ui-path-remove');
            removeBtn.addEventListener('click', () => {
                li.remove();
                updateIndices();
                updateDropdown();
            });
            
            list.appendChild(li);
            updateIndices();
            updateDropdown();
        };
        
        const updateIndices = () => {
            list.querySelectorAll('.ui-path-item').forEach((item, idx) => {
                item.querySelector('.ui-path-index').textContent = idx + 1;
            });
        };
        
        addBtn.addEventListener('click', () => {
            updateDropdown();
            selectWrapper.style.display = selectWrapper.style.display === 'none' ? 'block' : 'none';
        });
        
        selectWrapper.appendChild(selectDropdown);
        addSection.appendChild(addBtn);
        addSection.appendChild(selectWrapper);
        
        wrapper.appendChild(list);
        wrapper.appendChild(addSection);
        
        // 点击外部关闭下拉
        document.addEventListener('click', (e) => {
            if (!selectWrapper.contains(e.target) && !addBtn.contains(e.target)) {
                selectWrapper.style.display = 'none';
            }
        });
        
        return wrapper;
    },
    
    /**
     * 初始化：点击外部关闭下拉
     */
    init() {
        document.addEventListener('click', () => {
            document.querySelectorAll('.ui-select.open, .ui-multi-select.open').forEach(s => {
                s.classList.remove('open');
            });
        });
    }
};

// 导出
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UIComponents;
}