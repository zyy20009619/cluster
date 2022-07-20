import os
import argparse
from algos.CommonFunction import mainMethod
from algos.ExtensionFunction import extensionMain
from utils.gt_util import gen_ground_truth
from evaluator.mojo_fm import get_mojo_fm


def main():
    parser = argparse.ArgumentParser(description="Reverse Engineering")
    parser.add_argument('--option', default='cluster')  # 选择当前使用命令行功能：gt/cluster/evaluator
    parser.add_argument('--projectname', default='numpy')  #与deppath保持一致
    parser.add_argument('--deppath', default=r'D:\research\paperdocs\zyy\架构逆向\cluster(1)\cluster\test_data\input\type\numpy\numpy-report-enre.json')
    parser.add_argument('--resultpath', default='')
    # gt/cluster需要对分粒度进行依赖关系提取:module/class/method
    parser.add_argument('--granularity', default='method')
    # cluster：ACDC/Bunch/WCA/LIMBO/ARC/AGK/SKM/DBSCAN/SpectralClustering等
    parser.add_argument('--algo', default='bunch')
    # cluster：AGK/SKM/DBSCAN/SpectralClustering等聚类算法输入参数
    parser.add_argument('--AssociationNetworkName', default='Construct')
    parser.add_argument('--GraphEmbeddingAlgorithm', default='Struc2Vec')
    parser.add_argument('--num', default=6)
    # cluster：WCA/LIMBO/ARC聚类算法需输入项目路径和开发语言
    parser.add_argument('--projectpath', default=r'D:\research\testdata\testproject\numpy')
    parser.add_argument('--lang', default='python')
    # evaluator：使用mojoFM指标评估聚类结果和真实架构的相似度，需要输入gt和聚类结果
    parser.add_argument('--gt', default='../backend-out/backend-gt-out.rsf')
    parser.add_argument('--cluster', default='../backend-out/backend-acdc-out.rsf')

    args = parser.parse_args()
    project_name = args.projectname
    option = args.option
    dep_path = args.deppath
    result_path = args.resultpath
    AssociationNetworkName = 'Construct'
    GraphEmbeddingAlgorithm = 'Node2Vec'
    algo = args.algo
    ClusterNum = args.num
    project_path = args.projectpath
    lang = args.lang
    gt = args.gt
    cluster = args.cluster
    granularity = args.granularity

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
