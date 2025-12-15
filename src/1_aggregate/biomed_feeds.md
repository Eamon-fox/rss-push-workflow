# 生物/医学科研领域 RSS/Atom 测试用例

测试时间：2025-12-15（本机用 `curl -L` 做连通性检查）

## 已测通（HTTP 200）

### 预印本 / 论文聚合
- arXiv q-bio（RSS）：`https://export.arxiv.org/rss/q-bio`
- arXiv q-bio.BM（RSS）：`https://export.arxiv.org/rss/q-bio.BM`
- arXiv q-bio.GN（RSS）：`https://export.arxiv.org/rss/q-bio.GN`

### 期刊 / 出版社
- Nature（RSS）：`https://www.nature.com/nature.rss`
- Nature Genetics（RSS）：`https://www.nature.com/subjects/genetics.rss`
- Nature Cancer（RSS）：`https://www.nature.com/subjects/cancer.rss`
- Nature Microbiology（RSS）：`https://www.nature.com/subjects/microbiology.rss`
- PLOS ONE（Atom）：`https://journals.plos.org/plosone/feed/atom`
- PLOS Biology（Atom）：`https://journals.plos.org/plosbiology/feed/atom`
- PLOS Genetics（Atom）：`https://journals.plos.org/plosgenetics/feed/atom`
- Science（RSS）：`https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=science`
- NEJM（RSS）：`https://www.nejm.org/action/showFeed?type=etoc&feed=rss&jc=nejm`
- BMJ（RSS）：`https://www.bmj.com/rss.xml`

### medRxiv / bioRxiv（经 connect 子域提供 Atom 1.0）
- medRxiv 全部（Atom）：`http://connect.medrxiv.org/medrxiv_xml.php?subject=all`
- medRxiv 示例：`http://connect.medrxiv.org/medrxiv_xml.php?subject=infectious_diseases`
- medRxiv 示例：`http://connect.medrxiv.org/medrxiv_xml.php?subject=Genetic_and_Genomic_Medicine`
- medRxiv 示例（组合分类）：`http://connect.medrxiv.org/medrxiv_xml.php?subject=nursing+nutrition`
- bioRxiv 全部（Atom）：`http://connect.biorxiv.org/biorxiv_xml.php?subject=all`
- bioRxiv 示例：`http://connect.biorxiv.org/biorxiv_xml.php?subject=neuroscience`

## 未测通（用于容错/异常场景）

> 以下是“当时返回码”，后续可能会变化（常见原因：反爬/地区/需要特定 Header/限流）。

- bioRxiv 主站 RSS：`https://www.biorxiv.org/rss/latest.xml`（403）
- bioRxiv 主站 RSS：`https://www.biorxiv.org/rss/biorxiv.xml`（403）
- medRxiv 主站 RSS：`https://www.medrxiv.org/rss/latest.xml`（404）
- medRxiv 主站 RSS：`https://www.medrxiv.org/rss/medrxiv.xml`（404）
- eLife RSS：`https://elifesciences.org/rss.xml`（406）
- PubMed RSS 搜索：`https://pubmed.ncbi.nlm.nih.gov/rss/search/1/?term=CRISPR&sort=date`（500）
- PubMed RSS 搜索：`https://pubmed.ncbi.nlm.nih.gov/rss/search/1/?term=immunotherapy&sort=date`（500）

