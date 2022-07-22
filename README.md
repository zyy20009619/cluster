# cluster

### 功能

- **采集项目ground truth数据**

  ```python
  python Command.py --opt gt --pro [project name] --dep [dep path] --g [granularity] <--res [result path]>
  ```

  用户可通过传入--g参数设置要收集的ground truth粒度

- **架构聚类**

  ```python
  python Command.py --opt cluster --pro [project name] --dep [dep path] --g [granularity] --algo [clustering algorithm] <--ge [graph embedding algorithm] > <--num [cluster number]> <--propath [project path]> <--lang [language]> <--res [result path]>
  ```

  `1.如果用户选择使用AGK/SKM/DBSCAN/SpectralClustering几种聚类算法，那么在使用工具时需要传入--num/--ge参数`

  `2.如果用户选择使用WCA/LIMBO/ARC等聚类算法，那么在使用工具时需要传入--propath/--lang参数`

- **架构评估**

  ```python
  python Command.py --opt cluster --pro [project name] --gt [result of ground truth path] --cluster [result of clustering path] 
  ```

  

### Usage

```python
Command.py [-h] [--opt OPT] [--pro PRO] [--dep DEP] [--res RES] [--g G] [--algo ALGO] [--ge GE] [--num NUM] [--propath PROPATH] [--lang LANG] [--gt GT] [--cluster CLUSTER]

optional arguments:
  -h, --help         show this help message and exit
  --opt OPT          function options(gt/cluster/evaluator)
  --pro PRO          project name
  --dep DEP          project dep file path
  --res RES          result path
  --g G              granularity for gt or cluster
  --algo ALGO        clustering algorithm                       (ACDC/Bunch/WCA/LIMBO/ARC/AGK/SKM/DBSCAN/SpectralClustering)
  --ge GE            graph embedding algorithm(Struc2Vec/DeepWalk/Node2Vec)
  --num NUM          cluster number
  --propath PROPATH  project path
  --lang LANG        language
  --gt GT            ground truth file path
  --cluster CLUSTER  result of cluster file path
```

### 使用流程

```python
1. 将工具克隆到本地
2. 根据命令行输入正确命令进行架构聚类/真实架构提取/架构评估
```

