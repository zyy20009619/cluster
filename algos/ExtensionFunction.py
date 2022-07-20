import os
import json
from utils.dep_data_util import load_json_file


def extensionMain(project_name, project_path, file_path, lang, algo, granularity, base_out_path, option):
    data, cells, variable_name, variable_map, result_cells = load_json_file(file_path, granularity, option)
    depends_path = cell_to_depends(project_name, base_out_path, variable_map, result_cells)
    if algo == 'acdc':
        os.system('java -jar ./algos/tools/acdc.jar ' + depends_path + ' ' + base_out_path + '/' + project_name + '-acdc-out.rsf')
        _gen_json(base_out_path + '/' + project_name + '-acdc-out.rsf', base_out_path + '/' + project_name + '-acdc-out.json')
    elif algo == 'bunch':
        # TODO:没有任何显示(没有使用说明)
        os.system('java -jar bunch.jar ' + depends_path + ' ' + base_out_path + '/' + project_name + '-bunch-out.rsf')
    else:
        # TODO:报越界错误(推测原因：仅支持c/c++/java)
        cfg_path = gen_cfg(base_out_path, project_name, project_path, lang, depends_path, algo, granularity)
        os.system('java -Xmx4096m -Xss90m -jar ./algos/tools/arcade.jar -projfile ' + cfg_path)


def cell_to_depends(project_name, base_out_path, variable_map, cells):
    rsf_path = base_out_path + '/' + project_name + '-depends-out.rsf'
    with open(rsf_path, 'w', encoding='utf-8') as f:
        for c_src in cells:
            f.write('depends  ' + variable_map[c_src]['qualifiedName'] + ' ' + variable_map[cells[c_src]][
                'qualifiedName'] + '\n')
    return rsf_path


def gen_cfg(base_out_path, project_name, project_path, lang, rsf_path, algo, granularity):
    with open('./algos/cfg_template/arc_template.cfg', encoding='utf-8') as f:
        lines = f.readlines()

    cfg_path = base_out_path + '/' + project_name + '-out.cfg'
    with open(cfg_path, 'w', encoding='utf-8') as f:
        for line in lines:
            line = line.replace('\n', '')
            if line.startswith('project_name'):
                line += project_name
            elif line.startswith('lang'):
                line += lang
            elif line.startswith('granule'):
                line += granularity
            elif line.startswith('deps_rsf_file'):
                line += rsf_path
            elif line.startswith('src_dir'):
                line += project_path
            elif line.startswith('clustering_algorithm'):
                line += algo
            f.write(line + '\n')
    return cfg_path


def _gen_json(cluster_rsf_out, cluster_json_out):
    with open(cluster_rsf_out, 'r') as f:
        lines = f.readlines()
        jsonCluster = {}
        data = []
        cluster_dic = dict()
        for line in lines:
            line = line.replace('\n', '').split(' ')

            if line[1] not in cluster_dic:
                cluster_dic[line[1]] = list()
            cluster_dic[line[1]].append(line[2])

        for cluster in cluster_dic:
            res_data = dict()
            res_data["name"] = cluster
            children = list()
            for child in cluster_dic[cluster]:
                child_dic = dict()
                child_dic["name"] = child
                child_dic["value"] = 0.025
                child_dic["color"] = 1
                children.append(child_dic)
            res_data["children"] = children
            data.append(res_data)
        jsonCluster["data"] = data

    with open(cluster_json_out, 'w', encoding='utf-8') as f:
        json.dump(jsonCluster, f)


