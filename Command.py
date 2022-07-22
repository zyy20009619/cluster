import os
import argparse
from algos.CommonFunction import mainMethod
from algos.ExtensionFunction import extensionMain
from utils.gt_util import gen_ground_truth
from evaluator.mojo_fm import get_mojo_fm


def main():
    parser = argparse.ArgumentParser(description="Reverse Engineering")
    parser.add_argument('--opt', help='function options(gt/cluster/evaluator)', default='')  # 选择当前使用命令行功能：gt/cluster/evaluator
    parser.add_argument('--pro', help='project name', default='')  #与deppath保持一致
    parser.add_argument('--dep', help='project dep file path', default='')
    parser.add_argument('--res', help='result path', default='')
    # gt/cluster需要对分粒度进行依赖关系提取:module/class/method
    parser.add_argument('--g', help='granularity for gt or cluster', default='')
    # cluster：ACDC/Bunch/WCA/LIMBO/ARC/AGK/SKM/DBSCAN/SpectralClustering等
    parser.add_argument('--algo', help='clustering algorithm(ACDC/Bunch/WCA/LIMBO/ARC/AGK/SKM/DBSCAN/SpectralClustering)', default='ACDC')
    # cluster：AGK/SKM/DBSCAN/SpectralClustering等聚类算法输入参数
    # parser.add_argument('--an', help='association network name()', default='Construct')
    parser.add_argument('--ge', help='graph embedding algorithm(Struc2Vec/DeepWalk/Node2Vec)', default='Struc2Vec')
    parser.add_argument('--num', help='cluster number', default=6)
    # cluster：WCA/LIMBO/ARC聚类算法需输入项目路径和开发语言
    parser.add_argument('--propath', help='project path', default='')
    parser.add_argument('--lang', help='language', default='')
    # evaluator：使用mojoFM指标评估聚类结果和真实架构的相似度，需要输入gt和聚类结果
    parser.add_argument('--gt', help='ground truth file path', default='')
    parser.add_argument('--cluster', help='result of cluster file path', default='')

    args = parser.parse_args()
    project_name = args.pro
    option = args.opt
    dep_path = args.dep
    result_path = args.res
    AssociationNetworkName = 'Construct'
    GraphEmbeddingAlgorithm = args.ge
    algo = args.algo
    ClusterNum = args.num
    project_path = args.propath
    lang = args.lang
    gt = args.gt
    cluster = args.cluster
    granularity = args.g

    # 参数校验
    if option == '':
        print('请选择功能！')
        return
    elif project_name == '':
        print('请输入项目名！')
        return

    if option != 'gt' and dep_path == '':
        print('请输入依赖数据！')
        return

    if granularity == '':
        print('请输入粒度信息(module/class/method)！')
        return

    if result_path == '':
        result_path = '..//' + project_name + '-out'
    os.makedirs(result_path, exist_ok=True)

    # 根据选择功能进行gt收集/聚类/评估
    if option == 'gt':
        gen_ground_truth(result_path, dep_path, project_name, granularity, option)
    elif option == 'cluster':
        if algo in ['ACDC', 'Bunch', 'WCA', 'LIMBO', 'ARC']:
            extensionMain(project_name, project_path, dep_path, lang, algo.lower(), granularity, result_path, option)
        else:
            mainMethod(SoftwareName=project_name,
                       jsonFileName=dep_path,
                       AssociationNetworkName=AssociationNetworkName,
                       GraphEmbeddingAlgorithm=GraphEmbeddingAlgorithm,
                       ClusterAlgorithm=algo,
                       ClusterNum=ClusterNum,
                       base_out_path=result_path,
                       granularity=granularity,
                       option = option)
    else:
        get_mojo_fm(gt, cluster, result_path, project_name)


if __name__ == '__main__':
    main()
