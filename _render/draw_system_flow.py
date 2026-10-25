# -*- coding: utf-8 -*-
"""用 Pillow 手绘系统流程图 -> PNG（base 环境 Pillow 11.1.0）"""
import os
from PIL import Image, ImageDraw, ImageFont

S = 2  # 超采样倍数，输出 2 倍分辨率
HERE = os.path.dirname(os.path.abspath(__file__))
REG = r"C:\Windows\Fonts\msyh.ttc"
BOLD = r"C:\Windows\Fonts\msyhbd.ttc"

INK = (31, 41, 55)
GREY = (107, 114, 128)
LINE = (148, 163, 184)


def F(sz, bold=False):
    return ImageFont.truetype(BOLD if bold else REG, int(round(sz * S)))


# kind -> (fill, outline, text)
KIND = {
    "start":  ((31, 41, 55),    (31, 41, 55),    (255, 255, 255)),
    "proc":   ((255, 255, 255), (148, 163, 184), INK),
    "faq":    ((220, 252, 231), (22, 163, 74),   INK),
    "rag":    ((219, 234, 254), (37, 99, 235),   INK),
    "direct": ((254, 243, 199), (217, 119, 6),   INK),
    "fb":     ((254, 226, 226), (220, 38, 38),   INK),
    "db":     ((237, 233, 254), (124, 58, 237),  INK),
    "dec":    ((254, 249, 195), (245, 158, 11),  INK),
}


class Cv:
    def __init__(self, w, h):
        self.W, self.H = w, h
        self.im = Image.new("RGB", (int(w * S), int(h * S)), (255, 255, 255))
        self.d = ImageDraw.Draw(self.im)

    def box(self, x0, y0, x1, y1, fill, line, r=12, width=2):
        self.d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], radius=r * S,
                                 fill=fill, outline=line, width=width * S)

    def txt(self, x, y, s, sz=13, bold=False, color=INK, anchor="mm"):
        self.d.text((x * S, y * S), s, font=F(sz, bold), fill=color, anchor=anchor)

    def node(self, cx, cy, w, h, main, sub=None, kind="proc", fs=13, subfs=10.5):
        fill, line, fg = KIND[kind]
        self.box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, fill, line)
        if sub:
            self.txt(cx, cy - h * 0.16, main, fs, False, fg)
            self.txt(cx, cy + h * 0.25, sub, subfs, False, GREY)
        else:
            self.txt(cx, cy, main, fs, False, fg)

    def ar(self, x0, y, x1, color=LINE):
        self.d.line([x0 * S, y * S, (x1 - 10) * S, y * S], fill=color, width=2 * S)
        self.d.polygon([(x1 * S, y * S), ((x1 - 12) * S, (y - 6) * S),
                        ((x1 - 12) * S, (y + 6) * S)], fill=color)

    def ad(self, x, y0, y1, color=LINE):
        self.d.line([x * S, y0 * S, x * S, (y1 - 10) * S], fill=color, width=2 * S)
        self.d.polygon([(x * S, y1 * S), ((x - 6) * S, (y1 - 12) * S),
                        ((x + 6) * S, (y1 - 12) * S)], fill=color)

    def save(self, name):
        p = os.path.join(HERE, name)
        self.im.save(p)
        return p


W = 1520
H = 1232
c = Cv(W, H)

# ============ 标题 ============
c.txt(W / 2, 32, "考研就业智能问答系统 · 整体流程", 25, True)
c.txt(W / 2, 66, "FAQ 高速匹配 + RAG 检索增强 双引擎    |    "
                 "FastAPI + WebSocket · LangChain · Milvus · BGE-M3 · bge-reranker-large · BERT · BM25 · MySQL + Redis · DeepSeek",
      10.5, False, GREY)

# ============ 块 1：离线知识库构建 ============
Y1, H1 = 88, 124
c.box(30, Y1, W - 30, Y1 + H1, (255, 251, 235), (251, 191, 36), r=14)
c.txt(52, Y1 + 20, "① 离线知识库构建（跑一次，可重复）", 14.5, True, (180, 83, 9), anchor="lm")

row1 = [
    ("文档 8 类格式", ".txt .pdf .docx .ppt .pptx .jpg .png .md", "proc"),
    ("RapidOCR", "PDF 内嵌图 / Word / PPT 图片", "proc"),
    ("父子分层分块", "父 1200 / 子 300 / 重叠 50", "proc"),
    ("BGE-M3 双向量", "dense 语义 + sparse 关键词", "proc"),
    ("Milvus itcast.edurag", "IVF_FLAT + SPARSE_INVERTED · 41 块", "db"),
]
NW, GAP = 232, 30
x = (W - (len(row1) * NW + (len(row1) - 1) * GAP)) / 2
cy = Y1 + 62
for i, (m, s, k) in enumerate(row1):
    c.node(x + NW / 2, cy, NW, 54, m, s, k, fs=12.5, subfs=9.5)
    x += NW
    if i < len(row1) - 1:
        c.ar(x, cy, x + GAP)
        x += GAP
c.txt(W / 2, Y1 + H1 - 16, "元数据挂载：source 学科 / file_path / timestamp    ·    "
                           "文本 MD5 作主键 upsert，重复入库幂等", 10.5, False, GREY)

# ============ 块 2：在线问答链路 ============
Y2, H2 = 224, 352
c.box(30, Y2, W - 30, Y2 + H2, (239, 246, 255), (96, 165, 250), r=14)
c.txt(52, Y2 + 20, "② 在线问答链路", 14.5, True, (29, 78, 216), anchor="lm")

# 行 1
cy1 = Y2 + 62
r1 = [("用户提问 + 学科筛选", None, "start", 190),
      ("问候语正则（4 条）", "你好 / 你是谁 / 在吗 / 干嘛呢", "dec", 230),
      ("直接返回问候语", None, "proc", 175)]
tot = sum(w for _, _, _, w in r1) + 2 * 34
x = (W - tot) / 2
greet_cx = 0
for i, (m, s, k, w) in enumerate(r1):
    c.node(x + w / 2, cy1, w, 52, m, s, k, fs=12.5, subfs=9.5)
    if i == 1:
        greet_cx = x + w / 2
    x += w
    if i < len(r1) - 1:
        if i == 1:
            c.txt(x + 17, cy1 - 16, "命中", 9, False, (21, 128, 61))
        c.ar(x, cy1, x + 34)
        x += 34

# 行 2：FAQ 层
cy2 = cy1 + 90
# 未命中：从问候语节点底部正中垂直向下到 FAQ 层
c.ad(greet_cx, cy1 + 26, cy2 - 23)
c.txt(greet_cx + 10, (cy1 + 26 + cy2 - 23) / 2, "未命中", 9.5, False, GREY, anchor="lm")
c.node(W / 2, cy2, 470, 46, "FAQ 层：BM25 检索 + Softmax 置信度", None, "faq", fs=14)
c.ad(W / 2, cy2 + 23, cy2 + 55)

# 行 3：四道闸
cy3 = cy2 + 84
gates = [
    ("第 0 道", "零词项重叠短路"),
    ("第 1 道", "置信度 ≥ 0.68"),
    ("第 2 道", "学科掩码 common + 选中"),
    ("第 3 道", "实体校验 校名/年份"),
]
GW, GG = 200, 16
tot = 4 * GW + 3 * GG
x = (W - tot) / 2
for title, desc in gates:
    c.box(x, cy3 - 26, x + GW, cy3 + 26, (240, 253, 244), (22, 163, 74), r=10)
    c.txt(x + GW / 2, cy3 - 12, title, 9.5, True, (21, 128, 61))
    c.txt(x + GW / 2, cy3 + 10, desc, 11, False, INK)
    x += GW + GG

# 行 4：两条分支
cy4 = cy3 + 80
c.ad(470, cy3 + 26, cy4 - 26)
c.txt(480, cy3 + 44, "全部通过", 9.5, False, (21, 128, 61), anchor="lm")
c.node(470, cy4, 300, 50, "返回题库标准答案", "约 400 ms · 写入 Redis 缓存", "faq", fs=13, subfs=9.5)

c.ad(1020, cy3 + 26, cy4 - 26)
c.txt(1030, cy3 + 44, "任一不通过 → 降级", 9.5, False, (220, 38, 38), anchor="lm")
c.node(1020, cy4, 300, 50, "BERT 分类器：通用知识 or 专业咨询？", None, "dec", fs=12.5)

# ============ 块 3：RAG 分支 ============
Y3, H3 = 596, 204
c.box(30, Y3, W - 30, Y3 + H3, (239, 246, 255), (37, 99, 235), r=14)
c.txt(52, Y3 + 20, "③ RAG 检索增强分支（FAQ 降级后进入）", 14.5, True, (29, 78, 216), anchor="lm")

# 通用知识行
cya = Y3 + 60
c.node(150, cya, 170, 46, "通用知识", None, "direct", fs=12.5)
c.ar(240, cya, 278)
c.node(430, cya, 300, 46, "LLM 直接生成", "不检索，省一次完整 RAG 开销", "direct", fs=12.5, subfs=9.5)

# 专业咨询行
cyb = Y3 + 128
c.node(150, cyb, 170, 46, "专业咨询", None, "rag", fs=12.5)
c.ar(240, cyb, 278)
seq = [
    ("策略选择", "直接检索 / HyDE / 子查询"),
    ("双路 ANN 召回", "dense nprobe=10 + sparse"),
    ("WeightedRanker 融合", "稠密 0.7 / 稀疏 1.0"),
    ("bge-reranker 精排", "≥2 条触发，去重取 Top-2"),
    ("Prompt 拼装", "父块全文 + 最近 5 轮历史"),
    ("DeepSeek 流式生成", "约 5000 ms"),
]
NW2, GAP2 = 176, 20
x = 278
for i, (m, s) in enumerate(seq):
    c.node(x + NW2 / 2, cyb, NW2, 50, m, s, "rag", fs=11.5, subfs=9)
    x += NW2
    if i < len(seq) - 1:
        c.ar(x, cyb, x + GAP2)
        x += GAP2

c.txt(W / 2, Y3 + H3 - 18, "父子分层：检索单元是子块（精确），送进 LLM 的是父块全文（完整上下文）",
      10.5, False, GREY)

# ============ 块 4：统一出口 ============
Y4, H4 = 814, 112
c.box(30, Y4, W - 30, Y4 + H4, (248, 250, 252), (31, 41, 55), r=14)
c.txt(52, Y4 + 20, "④ 统一出口", 14.5, True, INK, anchor="lm")

cy5 = Y4 + 70
c.node(150, cy5 - 26, 200, 40, "FAQ 标准答案", None, "faq", fs=11.5)
c.node(150, cy5 + 26, 200, 40, "通用直答", None, "direct", fs=11.5)
c.node(W / 2, cy5, 430, 56, "WebSocket 三级协议", "start → token → end（回传 processing_time）", "rag", fs=13.5, subfs=10)
c.node(W - 150, cy5 - 26, 200, 40, "RAG 生成答案", None, "rag", fs=11.5)
c.node(W - 150, cy5 + 26, 200, 40, "兜底文案", None, "fb", fs=11.5)
c.ar(252, cy5 - 26, W / 2 - 217)
c.ar(252, cy5 + 26, W / 2 - 217)
c.ar(W / 2 + 217, cy5 - 26, W - 252)
c.ar(W / 2 + 217, cy5 + 26, W - 252)

# ============ 块 5：数据层 ============
Y5, H5 = 938, 104
c.box(30, Y5, W - 30, Y5 + H5, (245, 243, 255), (167, 139, 250), r=14)
c.txt(52, Y5 + 20, "⑤ 数据层与会话记忆", 14.5, True, (109, 40, 217), anchor="lm")

cells = [
    ("MySQL subjects_kg", "jpkb 题库 100 条 · conversations"),
    ("Redis db0", "原题/分词快照 + 答案缓存 · TTL 3600s"),
    ("MySQL 会话记忆", "每 session 保留最近 5 轮"),
    ("REST 接口 8 个", "会话/历史/列表/清空/学科/健康"),
]
CW, CG = 336, 22
x = (W - (4 * CW + 3 * CG)) / 2
for m, s in cells:
    c.node(x + CW / 2, Y5 + 62, CW, 46, m, s, "db", fs=12, subfs=9.5)
    x += CW + CG

# ============ 块 6：评估闭环 ============
Y6, H6 = 1054, 140
c.box(30, Y6, W - 30, Y6 + H6, (236, 253, 245), (52, 211, 153), r=14)
c.txt(52, Y6 + 20, "⑥ 效果评估闭环", 14.5, True, (4, 120, 87), anchor="lm")

e1 = [
    ("评测集 120 条", "question + ground_truth", "start"),
    ("eval_pipeline.py", "跑完整链路，采集 answer/contexts/延迟", "rag"),
    ("eval_ragas.py", "RAGAS 四指标打分", "faq"),
    ("summary.csv", "多组横向对比（Δ 相对基准）", "db"),
]
EW, EG = 300, 26
x = (W - (4 * EW + 3 * EG)) / 2
for i, (m, s, k) in enumerate(e1):
    c.node(x + EW / 2, Y6 + 58, EW, 44, m, s, k, fs=12, subfs=9.5)
    x += EW
    if i < 3:
        c.ar(x, Y6 + 58, x + EG)
        x += EG

m1 = ["faithfulness · 是否有据", "answer_relevancy · 是否切题",
      "context_precision · 检索是否干净", "context_recall · 证据是否召全",
      "Bad Case 定位 · 最低分样本"]
MW, MG = 224, 18
x = (W - (5 * MW + 4 * MG)) / 2
for s in m1:
    k = "fb" if s.startswith("Bad") else "proc"
    c.node(x + MW / 2, Y6 + 114, MW, 34, s, None, k, fs=10.5)
    x += MW + MG

# ============ 图例 ============
lg = [("faq", "FAQ 秒回路径"), ("rag", "RAG 检索生成路径"),
      ("direct", "通用知识直答"), ("fb", "兜底")]
x = W / 2 - 300
for k, label in lg:
    fill, line, _ = KIND[k]
    c.box(x, 1204, x + 15, 1219, fill, line, r=4, width=1)
    c.txt(x + 22, 1211, label, 11, False, (75, 85, 99), anchor="lm")
    x += 152

p = c.save("system_flow.png")
print("已生成:", p, flush=True)
print("尺寸:", Image.open(p).size, flush=True)
