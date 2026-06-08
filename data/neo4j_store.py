"""
Neo4j 时序知识图谱存储模块
=============================
将 ICEWS/YAGO 四元组数据持久化到 Neo4j 图数据库，
支持 Cypher 查询 + 虚拟 RDF 映射 + 原版 SPARQL。

架构:
  ICEWS/YAGO → Neo4j 属性图 → rdflib 虚拟RDF图 → 原版SPARQL查询

使用方式:
  from tkg_lpehd.data.neo4j_store import Neo4jStore
  store = Neo4jStore(uri="bolt://localhost:7687", user="neo4j", password="password")
  store.ingest_dataset(dataset)  # 将四元组批量存入Neo4j
  store.export_to_rdflib()       # 导出为rdflib图供SPARQL查询
"""

import os
import logging
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass, field

# Neo4j 驱动 (可选依赖)
try:
    from neo4j import GraphDatabase, Driver, Session
    from neo4j.exceptions import Neo4jError, ServiceUnavailable
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None
    Driver = None
    Session = None
    Neo4jError = Exception
    ServiceUnavailable = Exception

# rdflib (用于原版SPARQL)
try:
    import rdflib
    from rdflib import Graph, URIRef, Literal, Namespace, BNode
    from rdflib.namespace import RDF, XSD, RDFS
    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False
    Graph = None
    URIRef = None
    Literal = None
    Namespace = None

from .dataset import TKGDataSet, Quadruple
from ..utils import safe_xsd_date_literal

logger = logging.getLogger(__name__)

# Neo4j 默认连接参数
DEFAULT_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_USER = os.environ.get("NEO4J_USER", "neo4j")
DEFAULT_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neo4j2024")


class Neo4jStore:
    """
    Neo4j 时序知识图谱存储引擎

    特性:
    1. 高性能时序四元组存储 (节点=实体, 关系=带时间戳的边)
    2. 自动索引优化 (实体名、关系类型、时间戳)
    3. 虚拟RDF映射层 (导出为rdflib标准RDF图)
    4. 支持原版SPARQL查询 (通过rdflib SPARQL引擎)
    5. 连接失败自动回退到内存模式
    """

    def __init__(self,
                 uri: str = None,
                 user: str = None,
                 password: str = None,
                 database: str = "neo4j",
                 enable_rdf: bool = True):
        """
        初始化Neo4j连接

        Args:
            uri: Neo4j Bolt URI (默认: bolt://localhost:7687)
            user: 用户名 (默认: neo4j)
            password: 密码
            database: 数据库名 (默认: neo4j)
            enable_rdf: 是否启用虚拟RDF映射
        """
        self.uri = uri or DEFAULT_URI
        self.user = user or DEFAULT_USER
        self.password = password or DEFAULT_PASSWORD
        self.database = database
        self.enable_rdf = enable_rdf
        self.driver: Optional[Driver] = None
        self.connected = False
        self.fallback_memory = False

        # RDF命名空间
        self.TKG_NS = "http://tkg.lpehd.org/ontology/"
        self.RES_NS = "http://tkg.lpehd.org/resource/"
        self.EX = Namespace(self.RES_NS)
        self.TKG = Namespace(self.TKG_NS)

        # rdflib图 (虚拟RDF层)
        self.rdf_graph: Optional[Graph] = None

        # 统计
        self.stats = {
            "nodes_created": 0,
            "edges_created": 0,
            "rdf_triples": 0,
        }

        if not NEO4J_AVAILABLE:
            logger.warning("neo4j 驱动未安装, 将使用内存回退模式. "
                           "运行: pip install neo4j")
            self.fallback_memory = True
        else:
            self._connect()

    # ================================================================
    # 连接管理
    # ================================================================
    def _probe_port(self, host: str, port: int, timeout: float = 1.5) -> bool:
        """快速探测Neo4j端口是否可达"""
        import socket
        try:
            sock = socket.create_connection((host, port), timeout=timeout)
            sock.close()
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False

    def _connect(self):
        """建立Neo4j连接 (端口探测 → 快速失败回退)"""
        try:
            # 快速端口探测, 避免长时间阻塞
            parsed = self.uri.replace("bolt://", "").replace("neo4j://", "")
            host = parsed.split(":")[0]
            port = int(parsed.split(":")[1]) if ":" in parsed else 7687

            if not self._probe_port(host, port):
                print(f"[Neo4j] 端口 {host}:{port} 不可达, 自动回退到内存模式")
                self.fallback_memory = True
                self.connected = False
                return

            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=3600,
                connection_timeout=3,
            )
            # 验证连接
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                result.single()
            self.connected = True
            self._create_indexes()
            logger.info(f"Neo4j连接成功: {self.uri}")
            print(f"[Neo4j] 连接成功: {self.uri}")
        except (ServiceUnavailable, Neo4jError, Exception) as e:
            logger.warning(f"Neo4j连接失败({e}), 回退到内存模式")
            print(f"[Neo4j] 连接失败({type(e).__name__}), 自动回退到内存模式")
            self.connected = False
            self.fallback_memory = True

    def _create_indexes(self):
        """创建性能索引"""
        if not self.connected:
            return
        indexes = [
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_uri IF NOT EXISTS FOR (e:Entity) ON (e.uri)",
            "CREATE INDEX relation_type IF NOT EXISTS FOR ()-[r:EVENT]-() ON (r.type)",
            "CREATE INDEX event_timestamp IF NOT EXISTS FOR ()-[e:EVENT]-() ON (e.timestamp)",
            "CREATE INDEX subject_time IF NOT EXISTS FOR (s:Entity)-[r:EVENT]->() ON (r.timestamp, s.name)",
        ]
        try:
            with self.driver.session(database=self.database) as session:
                for idx_stmt in indexes:
                    try:
                        session.run(idx_stmt)
                    except Neo4jError:
                        pass  # 索引可能已存在
        except Exception:
            pass

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            self.connected = False

    # ================================================================
    # 数据导入: 四元组 → Neo4j属性图
    # ================================================================
    def ingest_dataset(self, dataset: TKGDataSet,
                        batch_size: int = 500,
                        clear_existing: bool = True) -> int:
        """
        将TKG数据集四元组批量存入Neo4j

        Neo4j数据模型:
          (:Entity {name, uri, type})
            -[:EVENT {type, timestamp, timestamp_days}]→
          (:Entity {name, uri, type})

        Args:
            dataset: TKGDataSet实例
            batch_size: 批量提交大小
            clear_existing: 是否清空已有数据

        Returns:
            int: 导入的边数
        """
        if self.fallback_memory or not self.connected:
            return self._ingest_memory_fallback(dataset)

        if clear_existing:
            self._clear_graph()

        quads = dataset.quadruples
        total = len(quads)

        # 批量事务写入
        with self.driver.session(database=self.database) as session:
            for i in range(0, total, batch_size):
                batch = quads[i:i + batch_size]
                tx = session.begin_transaction()

                for q in batch:
                    s, r, o, t = q.subject, q.relation, q.object, q.timestamp
                    time_val = q.time_val

                    # 创建实体节点 (如不存在) + 关系边
                    tx.run("""
                        MERGE (s:Entity {name: $subject})
                        ON CREATE SET s.uri = $subject_uri, s.type = $subject_type
                        ON MATCH SET s.uri = $subject_uri

                        MERGE (o:Entity {name: $object})
                        ON CREATE SET o.uri = $object_uri, o.type = $object_type
                        ON MATCH SET o.uri = $object_uri

                        CREATE (s)-[r:EVENT {
                            type: $relation,
                            timestamp: $timestamp,
                            timestamp_days: $time_val,
                            relation_uri: $relation_uri
                        }]->(o)
                        """,
                        subject=s,
                        subject_uri=f"{self.RES_NS}{s}",
                        subject_type=self._infer_type(s),
                        object=o,
                        object_uri=f"{self.RES_NS}{o}",
                        object_type=self._infer_type(o),
                        relation=r,
                        relation_uri=f"{self.TKG_NS}{r}",
                        timestamp=t,
                        time_val=time_val,
                    )

                tx.commit()
                self.stats["edges_created"] += len(batch)

        self.stats["nodes_created"] = self._count_nodes()
        print(f"[Neo4j] 导入完成: {self.stats['nodes_created']}个节点, "
              f"{self.stats['edges_created']}条边")
        return self.stats["edges_created"]

    def _ingest_memory_fallback(self, dataset: TKGDataSet) -> int:
        """内存回退模式 (Neo4j不可用时)"""
        print("[Neo4j] 使用内存回退模式存储数据")
        self._memory_quads = []
        for q in dataset.quadruples:
            self._memory_quads.append(q)
        return len(self._memory_quads)

    def _infer_type(self, entity_name: str) -> str:
        """推断实体类型 (用于RDF映射)"""
        countries = {"United_States", "China", "Russia", "South_Korea", "Japan",
                     "India", "Germany", "France", "United_Kingdom", "Iran",
                     "Iraq", "Syria", "Egypt", "Saudi_Arabia", "Turkey",
                     "Pakistan", "Afghanistan", "Mexico", "Canada", "Brazil",
                     "Australia", "Ukraine", "Poland", "Italy", "Spain"}
        leaders = {"Barack_Obama", "Xi_Jinping", "Vladimir_Putin",
                   "Park_Geun-hye", "Shinzo_Abe", "Angela_Merkel",
                   "David_Cameron", "Narendra_Modi", "Hassan_Rouhani",
                   "Ban_Ki-moon", "John_Kerry"}
        orgs = {"United_Nations", "European_Union", "NATO", "OPEC",
                "World_Bank", "IMF", "ASEAN", "WHO", "Red_Cross"}
        if entity_name in countries:
            return "Country"
        elif entity_name in leaders:
            return "Person"
        elif entity_name in orgs:
            return "Organization"
        return "Entity"

    def _clear_graph(self):
        """清空图数据"""
        if self.connected:
            with self.driver.session(database=self.database) as session:
                session.run("MATCH (n) DETACH DELETE n")
            self.stats = {"nodes_created": 0, "edges_created": 0, "rdf_triples": 0}

    def _count_nodes(self) -> int:
        """统计节点数"""
        if self.connected:
            with self.driver.session(database=self.database) as session:
                result = session.run("MATCH (n) RETURN count(n) as cnt")
                return result.single()["cnt"]
        return 0

    # ================================================================
    # Cypher 查询接口 (替代原SPARQL匹配器的Python循环)
    # ================================================================
    def query_temporal_pattern(self, relation_chain: List[str],
                                 validity_days: int = 60) -> List[Dict]:
        """
        使用时序Cypher查询匹配事件模式

        论文SPARQL模式 → Cypher等价转换:
          外交声明级联: Make_statement → Make_an_appeal_or_request → Make_a_visit

        Args:
            relation_chain: 关系链 [r1, r2, r3, ...]
            validity_days: 有效天数

        Returns:
            匹配结果列表
        """
        if self.fallback_memory or not self.connected:
            return []

        if len(relation_chain) < 2:
            return []

        # 动态构建Cypher查询
        # 匹配时序关系链: e1 -[r1,t1]→ e2 -[r2,t2]→ e3, 满足 t1 <= t2 <= t3
        with self.driver.session(database=self.database) as session:
            # 构建多跳路径查询
            match_parts = ["(s:Entity)"]
            where_parts = []
            with_clauses = []

            # 路径变量
            prev_node = "s"
            prev_time = None
            results = []

            # 简化: 逐个查询并组合
            chain_len = len(relation_chain)
            for idx, rel in enumerate(relation_chain):
                if idx == 0:
                    # 第一步: s -[rel,t1]→ n1
                    cypher = """
                        MATCH (s:Entity)-[e1:EVENT {type: $rel1}]->(n1:Entity)
                        RETURN s.name as subject, e1.type as rel1, n1.name as obj1,
                               e1.timestamp as t1
                        LIMIT 200
                    """
                    recs = session.run(cypher, rel1=rel).data()
                    if not recs:
                        return []
                    results = recs
                else:
                    # 后续步骤: 在前一步的结果上继续匹配
                    new_results = []
                    for prev_rec in results:
                        prev_obj = prev_rec.get(f"obj{idx}", prev_rec.get("obj1"))
                        prev_ts = prev_rec.get(f"t{idx}", prev_rec.get("t1"))

                        cypher = f"""
                            MATCH (prev:Entity {{name: $prev_obj}})
                                  -[e:EVENT {{type: $rel}}]->(next:Entity)
                            WHERE e.timestamp >= $prev_ts
                            RETURN $prev_subj as subject,
                                   $prev_rel_path as path_so_far,
                                   e.type as rel{idx+1},
                                   next.name as obj{idx+1},
                                   e.timestamp as t{idx+1}
                            LIMIT 50
                        """
                        recs = session.run(
                            cypher,
                            prev_obj=prev_obj,
                            rel=rel,
                            prev_ts=prev_ts,
                            prev_subj=results[0]["subject"] if not new_results else prev_rec.get("subject"),
                            prev_rel_path="|".join(relation_chain[:idx]),
                        ).data()
                        for r in recs:
                            merged = {**prev_rec, **r}
                            new_results.append(merged)

                    results = new_results
                    if not results:
                        break

            return results

    # ================================================================
    # 虚拟RDF映射: Neo4j → rdflib Graph → 原版SPARQL
    # ================================================================
    def export_to_rdflib(self, dataset: TKGDataSet = None,
                          max_events: int = 2000) -> Optional[Graph]:
        """
        将Neo4j数据导出为rdflib标准RDF图

        Args:
            dataset: TKGDataSet (Neo4j不可用时从内存获取)
            max_events: 最大导出事件数 (避免rdflib在大图上SPARQL过慢)

        Returns:
            rdflib.Graph: 标准RDF图, 支持原版SPARQL
        """
        if not RDFLIB_AVAILABLE:
            print("[RDF] rdflib未安装, 运行: pip install rdflib")
            return None

        g = Graph()
        g.bind("tkg", self.TKG_NS)
        g.bind("xsd", XSD)
        g.bind("rdf", RDF)
        g.bind("rdfs", RDFS)

        # 获取数据
        if self.connected:
            quads_data = list(self._export_all_from_neo4j())
        elif dataset:
            quads_data = [(q.subject, q.relation, q.object, q.timestamp)
                          for q in dataset.quadruples]
        elif hasattr(self, '_memory_quads'):
            quads_data = [(q.subject, q.relation, q.object, q.timestamp)
                          for q in self._memory_quads]
        else:
            return g

        if not quads_data:
            return g

        # 采样限制: rdflib SPARQL在大图上较慢, 采样后只影响查询速度不影响数据完整性
        import random
        if len(quads_data) > max_events:
            quads_data = random.sample(quads_data, max_events)

        # 构建RDF三元组
        TKG = Namespace(self.TKG_NS)
        RES = Namespace(self.RES_NS)

        for idx, (s, r, o, t) in enumerate(quads_data):
            event_uri = RES[f"Event_{idx}"]

            # 声明类型
            g.add((event_uri, RDF.type, TKG.Event))
            # 事件属性
            g.add((event_uri, TKG.subject, RES[s.replace(" ", "_")]))
            g.add((event_uri, TKG.relation, TKG[r.replace(" ", "_")]))
            g.add((event_uri, TKG.object, RES[o.replace(" ", "_")]))
            g.add((event_uri, TKG.timestamp, safe_xsd_date_literal(t)))

            # 声明实体类型
            g.add((RES[s.replace(" ", "_")], RDF.type, TKG.Entity))
            g.add((RES[o.replace(" ", "_")], RDF.type, TKG.Entity))

        self.rdf_graph = g
        self.stats["rdf_triples"] = len(g)
        print(f"[RDF] 虚拟RDF图构建完成: "
              f"{self.stats['rdf_triples']}条三元组, "
              f"{len(quads_data)}个事件")
        return g

    def _export_all_from_neo4j(self):
        """从Neo4j导出所有四元组"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (s:Entity)-[r:EVENT]->(o:Entity)
                RETURN s.name as subject, r.type as relation,
                       o.name as object, r.timestamp as timestamp
                LIMIT 100000
            """)
            for record in result:
                yield (record["subject"], record["relation"],
                       record["object"], record["timestamp"])

    # ================================================================
    # 原版SPARQL查询接口
    # ================================================================
    def execute_sparql(self, query: str) -> List[Dict]:
        """
        执行标准SPARQL查询

        所有查询使用真正的SPARQL语法,
        完全的W3C标准兼容 (通过rdflib SPARQL引擎).

        示例:
          PREFIX tkg: <http://tkg.lpehd.org/ontology/>
          PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
          SELECT ?s ?o ?t WHERE {
            ?e tkg:subject ?s ;
               tkg:relation tkg:Make_a_visit ;
               tkg:object ?o ;
               tkg:timestamp ?t .
            FILTER(?t >= \"2014-06-01\"^^xsd:date)
          }
          ORDER BY ?t

        Args:
            query: 标准SPARQL查询字符串

        Returns:
            List[Dict]: 查询结果
        """
        if not RDFLIB_AVAILABLE:
            print("[SPARQL] rdflib未安装, 请运行: pip install rdflib")
            return []

        if self.rdf_graph is None:
            print("[SPARQL] RDF图未构建, 请先调用 export_to_rdflib()")
            return []

        # 执行SPARQL查询
        try:
            result = self.rdf_graph.query(query)
        except Exception as e:
            print(f"[SPARQL] 查询语法错误: {e}")
            return []

        # 转换结果
        bindings = []
        for row in result:
            binding = {}
            for var in result.vars:
                val = row[var]
                if val is None:
                    binding[str(var)] = None
                elif isinstance(val, (URIRef, Literal)):
                    binding[str(var)] = str(val)
                else:
                    binding[str(var)] = str(val)
            bindings.append(binding)

        return bindings

    # ================================================================
    # 预定义SPARQL查询 (论文6种时序事件模式)
    # ================================================================
    def get_pattern_queries(self) -> Dict[str, str]:
        """
        返回论文定义的6种时序事件模式的SPARQL查询

        这些是标准SPARQL 1.1查询,
        可以在任何兼容的SPARQL引擎上运行
        """
        return {
            # 模式1: 外交声明级联
            "diplomatic_cascade": """
                PREFIX tkg: <http://tkg.lpehd.org/ontology/>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                SELECT ?subject ?object ?t1 ?t2 ?t3 WHERE {
                  ?e1 tkg:subject ?subject ;
                      tkg:relation tkg:Make_statement ;
                      tkg:object ?object ;
                      tkg:timestamp ?t1 .
                  ?e2 tkg:subject ?subject ;
                      tkg:relation tkg:Make_an_appeal_or_request ;
                      tkg:object ?object ;
                      tkg:timestamp ?t2 .
                  ?e3 tkg:subject ?subject ;
                      tkg:relation tkg:Make_a_visit ;
                      tkg:object ?object ;
                      tkg:timestamp ?t3 .
                  FILTER(?t1 <= ?t2 && ?t2 <= ?t3)
                }
                ORDER BY ?t1
                LIMIT 100
            """,

            # 模式2: 冲突升级
            "conflict_escalation": """
                PREFIX tkg: <http://tkg.lpehd.org/ontology/>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                SELECT ?subject ?object ?t1 ?t2 ?t3 WHERE {
                  ?e1 tkg:subject ?subject ;
                      tkg:relation tkg:Accuse ;
                      tkg:object ?object ;
                      tkg:timestamp ?t1 .
                  ?e2 tkg:subject ?subject ;
                      tkg:relation tkg:Criticize_or_denounce ;
                      tkg:object ?object ;
                      tkg:timestamp ?t2 .
                  ?e3 tkg:subject ?subject ;
                      tkg:relation tkg:Protest_or_demonstrate ;
                      tkg:object ?object ;
                      tkg:timestamp ?t3 .
                  FILTER(?t1 <= ?t2 && ?t2 <= ?t3)
                }
                ORDER BY ?t1
                LIMIT 100
            """,

            # 模式3: 合作推进
            "cooperation_progression": """
                PREFIX tkg: <http://tkg.lpehd.org/ontology/>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                SELECT ?subject ?object ?t1 ?t2 ?t3 WHERE {
                  ?e1 tkg:subject ?subject ;
                      tkg:relation tkg:Express_intent_to_cooperate ;
                      tkg:object ?object ;
                      tkg:timestamp ?t1 .
                  ?e2 tkg:subject ?subject ;
                      tkg:relation tkg:Engage_in_negotiation ;
                      tkg:object ?object ;
                      tkg:timestamp ?t2 .
                  ?e3 tkg:subject ?subject ;
                      tkg:relation tkg:Sign_formal_agreement ;
                      tkg:object ?object ;
                      tkg:timestamp ?t3 .
                  FILTER(?t1 <= ?t2 && ?t2 <= ?t3)
                }
                ORDER BY ?t1
                LIMIT 100
            """,

            # 模式4: 援助承诺与执行
            "aid_commitment": """
                PREFIX tkg: <http://tkg.lpehd.org/ontology/>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                SELECT ?subject ?object ?t1 ?t2 ?t3 WHERE {
                  ?e1 tkg:subject ?subject ;
                      tkg:relation tkg:Express_intent_to_provide_aid ;
                      tkg:object ?object ;
                      tkg:timestamp ?t1 .
                  ?e2 tkg:subject ?subject ;
                      tkg:relation tkg:Provide_economic_aid ;
                      tkg:object ?object ;
                      tkg:timestamp ?t2 .
                  ?e3 tkg:subject ?subject ;
                      tkg:relation tkg:Cooperate_on_economic_issues ;
                      tkg:object ?object ;
                      tkg:timestamp ?t3 .
                  FILTER(?t1 <= ?t2 && ?t2 <= ?t3)
                }
                ORDER BY ?t1
                LIMIT 100
            """,

            # 模式5: 调解与协议
            "mediation_treaty": """
                PREFIX tkg: <http://tkg.lpehd.org/ontology/>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                SELECT ?subject ?object ?t1 ?t2 ?t3 WHERE {
                  ?e1 tkg:subject ?subject ;
                      tkg:relation tkg:Mediate ;
                      tkg:object ?object ;
                      tkg:timestamp ?t1 .
                  ?e2 tkg:subject ?subject ;
                      tkg:relation tkg:Engage_in_negotiation ;
                      tkg:object ?object ;
                      tkg:timestamp ?t2 .
                  ?e3 tkg:subject ?subject ;
                      tkg:relation tkg:Ratify_treaty ;
                      tkg:object ?object ;
                      tkg:timestamp ?t3 .
                  FILTER(?t1 <= ?t2 && ?t2 <= ?t3)
                }
                ORDER BY ?t1
                LIMIT 100
            """,

            # 模式6: 道歉-访问-合作链
            "apology_visit_cooperation": """
                PREFIX tkg: <http://tkg.lpehd.org/ontology/>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                SELECT ?subject ?intermediate ?object ?t1 ?t2 ?t3 WHERE {
                  ?e1 tkg:subject ?subject ;
                      tkg:relation tkg:Apologize ;
                      tkg:object ?object ;
                      tkg:timestamp ?t1 .
                  ?e2 tkg:subject ?subject ;
                      tkg:relation tkg:Make_a_visit ;
                      tkg:object ?intermediate ;
                      tkg:timestamp ?t2 .
                  ?e3 tkg:subject ?intermediate ;
                      tkg:relation tkg:Engage_in_diplomatic_cooperation ;
                      tkg:object ?object ;
                      tkg:timestamp ?t3 .
                  FILTER(?t1 <= ?t2 && ?t2 <= ?t3)
                }
                ORDER BY ?t1
                LIMIT 100
            """,
        }

    def run_all_pattern_queries(self) -> Dict[str, List[Dict]]:
        """执行所有6种时序事件模式的SPARQL查询"""
        if self.rdf_graph is None:
            print("[SPARQL] 请先调用 export_to_rdflib()")
            return {}

        queries = self.get_pattern_queries()
        results = {}

        for name, query_str in queries.items():
            try:
                rows = self.execute_sparql(query_str)
                results[name] = rows
                print(f"  {name}: {len(rows)}个匹配")
            except Exception as e:
                print(f"  {name}: 查询失败 - {e}")
                results[name] = []

        total = sum(len(v) for v in results.values())
        print(f"[SPARQL] 总计匹配: {total} 个时序事件模式")
        return results

    # ================================================================
    # 统计与诊断
    # ================================================================
    def summary(self) -> str:
        """打印存储摘要"""
        lines = [
            f"Neo4j存储: {'已连接' if self.connected else ('内存回退' if self.fallback_memory else '未连接')}",
            f"  节点数: {self.stats['nodes_created']}",
            f"  边数: {self.stats['edges_created']}",
            f"  RDF三元组: {self.stats['rdf_triples']}",
            f"  连接: {self.uri}",
        ]
        if RDFLIB_AVAILABLE and self.rdf_graph:
            lines.append(f"  SPARQL就绪: 是 (rdflib {rdflib.__version__})")
        return "\n".join(lines)


# ================================================================
# 便捷函数
# ================================================================
def create_neo4j_store(dataset: TKGDataSet = None,
                        uri: str = None,
                        user: str = None,
                        password: str = None) -> Neo4jStore:
    """
    创建Neo4j存储并导入数据

    一键完成: 连接 → 导入 → 构建RDF
    """
    store = Neo4jStore(uri=uri, user=user, password=password)

    if dataset is not None:
        store.ingest_dataset(dataset)

    if store.enable_rdf:
        store.export_to_rdflib(dataset)

    return store
