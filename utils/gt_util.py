from utils.dep_data_util import load_json_file


def gen_ground_truth(base_out_path, dep_path, project_name, granularity, option):
    package_contain, variable_map = load_json_file(dep_path, granularity, option)

    rsf_path = base_out_path + '/' + project_name + '-gt-out.rsf'
    with open(rsf_path, 'w', encoding='utf-8') as f:
        for pack in package_contain:
            for obj in package_contain[pack]:
                f.writelines('Contain ' + variable_map[pack]['qualifiedName'] + ' ' + variable_map[obj]['qualifiedName'] + '\n')
