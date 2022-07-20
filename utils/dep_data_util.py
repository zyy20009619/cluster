import json


def load_json_file(filename, granularity, option):
    data = json.load(open(filename))
    variables = data['variables']
    cells = data['cells']

    # 实体信息
    variable_name = list()
    variable_map = dict()
    for item in variables:
        variable_name.append(item['qualifiedName'])
        variable_map[item['id']] = item

    # package contain module/class/method
    package_contain_module = dict()
    # 每层实体之间的映射（底层如果存在依赖，顶层必有依赖）
    class_or_method_defined_by_module = dict()  # module->class/method
    class_or_method_defined_by_class = dict()  # class->class/method
    # 单层实体之间的依赖
    alias = dict()  # module->module
    # class->method(如果method不在这个类中，那么产生两个类之间的依赖)
    class_call_method = dict()
    class_use_method = dict()
    # method1->method2(如果method1和method2不在同一类，那么即产生两个类之间的依赖)
    method_call_method = dict()
    method_use_method = dict()

    module_define = dict()
    class_define = dict()
    for c in cells:
        # contain
        if c['values']['kind'] == 'Contain':
            if c['src'] not in package_contain_module:
                package_contain_module[c['src']] = list()
            package_contain_module[c['src']].append(c['dest'])

        # define
        if c['values']['kind'] == 'Define':
            if variable_map[c['src']]['category'] == 'Module':
                class_or_method_defined_by_module[c['dest']] = c['src']
                if c['src'] not in module_define:
                    module_define[c['src']] = list()
                module_define[c['src']].append(c['dest'])
            elif variable_map[c['src']]['category'] == 'Class':
                class_or_method_defined_by_class[c['dest']] = c['src']
                if c['src'] not in class_define:
                    class_define[c['src']] = list()
                class_define[c['src']].append(c['dest'])

        # alias
        if c['values']['kind'] == 'Alias':
            if variable_map[c['src']]['category'] == 'Module':
                alias[c['src']] = c['dest']

        # call
        if c['values']['kind'] == 'Call':
            if variable_map[c['src']]['category'] == 'Class':
                class_call_method[c['src']] = c['dest']
            elif variable_map[c['src']]['category'] == 'Function':
                method_call_method[c['src']] = c['dest']

        # use
        if c['values']['kind'] == 'Use':
            if variable_map[c['src']]['category'] == 'Class' and variable_map[c['dest']]['category'] == 'Function':
                class_use_method[c['src']] = c['dest']
            elif variable_map[c['src']]['category'] == 'Function' and variable_map[c['dest']]['category'] == 'Function':
                method_use_method[c['src']] = c['dest']

    if option == 'gt':
        package_contain_class, package_contain_method = _del_package_contain_or_class_method(package_contain_module, module_define, class_define, variable_map)
        if granularity == 'method':
            return package_contain_method, variable_map
        elif granularity == 'class':
            return package_contain_class, variable_map
        return package_contain_module, variable_map

    result_cells = dict()
    if granularity == 'method':
        result_cells = dict(method_call_method)
        result_cells.update(method_use_method)
    elif granularity == 'class':
        # 如果因为method之间存在类外依赖，那么需要映射到class
        result_cells = dict(class_call_method)
        result_cells.update(class_use_method)
        result_cells.update(_del_low_to_high(method_call_method, class_or_method_defined_by_class))
        result_cells.update(_del_low_to_high(method_use_method, class_or_method_defined_by_class))
    elif granularity == 'module':
        # 如果因为method之间存在类外依赖，那么需要映射到class
        class_dic = dict(class_call_method)
        class_dic.update(class_use_method)
        class_dic.update(_del_low_to_high(method_call_method, class_or_method_defined_by_class))
        class_dic.update(_del_low_to_high(method_use_method, class_or_method_defined_by_class))
        # 如果类之间存在module外依赖，那么需要映射到module
        result_cells.update(alias)
        result_cells.update(_del_low_to_high(class_dic, class_or_method_defined_by_module))

    # TODO：暂时使用cells返回用于main里面的算法，后面再适配
    return data, cells, variable_name, variable_map, result_cells


def _del_low_to_high(rel, defined_rel):
    res = dict()

    for r_src in rel:
        if r_src in defined_rel and rel[r_src] in defined_rel and defined_rel[r_src] != defined_rel[rel[r_src]]:
            res[defined_rel[r_src]] = defined_rel[rel[r_src]]

    return res


def _del_package_contain_or_class_method(package_contain_module, module_define, class_define, variable_map):
    package_contain_class = dict()
    package_contain_method = dict()
    for pack in package_contain_module:
        class_list = list()
        method_list = list()
        for module in package_contain_module[pack]:
            if module in module_define:
                _del_module_contain(module_define[module], class_define, class_list, method_list, variable_map)
        package_contain_class[pack] = class_list
        package_contain_method[pack] = method_list

    return package_contain_class, package_contain_method


def _del_module_contain(module_define_obj, class_define, class_list, method_list, variable_map):
    for obj in module_define_obj:
        if variable_map[obj]['category'] == 'Class':
            class_list.append(obj)
            if obj in class_define:
                for c_obj in class_define[obj]:
                    if variable_map[c_obj]['category'] == 'Function':
                        method_list.append(c_obj)
                    elif variable_map[c_obj]['category'] == 'Class':
                        class_list.append(c_obj)
        elif variable_map[obj]['category'] == 'Function':
            method_list.append(obj)

