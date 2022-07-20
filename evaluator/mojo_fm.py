import os
from subprocess import Popen, PIPE


def get_mojo(file_path1, file_path2):
	cmd = "java -jar ./evaluator/mojo.jar  " + file_path1 + "  " + file_path2+" -fm"
	p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
	out, err = p.communicate()
	results = out.strip()
	return results


def get_mojo_fm(gt_path, cluster_path, base_out_path, project_name):
	with open(base_out_path + '/' + project_name + '-mojo-out.txt', 'w') as f:
		f.write(str(max(get_mojo(gt_path, cluster_path), get_mojo(cluster_path, gt_path))))
