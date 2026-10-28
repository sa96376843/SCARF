# 数据来源与校验

- BODMAS 官方说明：https://whyisyoung.github.io/BODMAS/
- 本实验使用的时间戳 Parquet 镜像：https://www.kaggle.com/datasets/dhoogla/bodmas
- 数据集论文：L. Yang, A. Ciptadi, I. Laziuk, A. Ahmadzadeh, and G. Wang, “BODMAS: An Open Dataset for Learning Based Temporal Analysis of PE Malware,” 2021 IEEE Security and Privacy Workshops, pp. 78–84.

取得镜像后，将 `bodmas-timestamps.parquet` 放入仓库根目录的 `data/`，或者通过 `--data` 指定其他路径。原始特征文件未重复上传 GitHub，使用与分发遵循原数据来源条款。

预期 Parquet 大小为 420,552,071 字节，SHA-256 为：

```
5e5123aba9a9c813fc3df49aa18c74c40c9e234eda62bbb6fc347632aee462f2
```

原下载 ZIP 的 SHA-256：`d6423e538fb0b84458bf584e9d25d75d5e77b973aeae302e85edca961c038148`。

镜像包含 134,428 条记录、2,381 个特征。原始发布数为 134,435，两者不是同一个逐行文件。当前协议使用 4,051 条初始训练记录、7,524 条校准记录与 112,122 条前向测试记录；其余 10,731 条不纳入该时间协议。预测文件中的 `index` 对应该校验版本的原始行号，不能映射到另一个排序或清洗版本。

```bash
python -m pip install requests
python SCARF_TDSC/code/download_datasets.py bodmas data/bodmas.zip
```

下载后按数据来源要求解压并核对上述哈希。动态执行器在读取数据时也会核验 SHA-256。本研究只处理预提取特征和标签，不需要执行恶意软件。
