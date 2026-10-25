# -*- coding: utf-8 -*-
"""C 组脚本的集成契约自检（不是实验，不调用 LLM，不跑问答）。

只回答一个问题：`_exp_c1.py` / `_exp_c2.py` 依赖的每一个外部符号，
在当前代码基线上是否真实存在、名字是否对。
--fix 模式：把发现的问题落成补丁建议，不自动改文件。
"""
import inspect
import json
import os
import sys
import tempfile

HERE = r"D:\Study\code6\intergrated_qa_system"
sys.path.insert(0, HERE)
os.chdir(HERE)

fails = []
oks = []


def check(label, cond, detail=""):
    (oks if cond else fails).append(f"{label}{(' — ' + detail) if detail else ''}")


# ---- 1. 公共模块能否导入（连带触发 config.LOG_FILE 机制） ----
try:
    import _c_common as C
    check("import _c_common", True)
except Exception as e:
    check("import _c_common", False, repr(e))
    print("FATAL: _c_common 无法导入，后续检查无意义")
    print(json.dumps({"ok": oks, "fail": fails}, ensure_ascii=False, indent=2))
    raise SystemExit(1)

for name in ("build_samples", "force_faq_sink", "restore_faq_threshold",
             "clear_answer_cache", "context_metrics", "summarize_retrieval",
             "mean_of", "fmt", "numbers", "core_terms", "samples_as_questions_json"):
    check(f"_c_common.{name}", hasattr(C, name))
for name in ("OUT_DIR", "QA_PATH", "SAMPLES_PER_SUBJECT", "EXPECTED_SAMPLES",
             "FAQ_SINK_THRESHOLD"):
    check(f"_c_common.{name}", hasattr(C, name))

D = getattr(C, "OUT_DIR", None)
check("_c_common.OUT_DIR 存在", bool(D) and os.path.isdir(D), f"dir={D}")
QT = getattr(C, "QA_PATH", None)
check("_c_common.QA_PATH 存在", bool(QT) and os.path.isfile(QT), f"path={QT}")
check("EXPECTED_SAMPLES == 20", getattr(C, "EXPECTED_SAMPLES", None) == 20,
      f"={getattr(C, 'EXPECTED_SAMPLES', None)}")

# ---- 2. eval_pipeline 的符号（_exp_c2 直接 import 私有名） ----
try:
    from rag_qa.rag_assesment import eval_pipeline as EP
    check("import eval_pipeline", True)
except Exception as e:
    EP = None
    check("import eval_pipeline", False, repr(e))

if EP is not None:
    for name in ("run_one", "_norm", "_number_coverage"):
        check(f"eval_pipeline.{name}", hasattr(EP, name))
    if hasattr(EP, "_norm"):
        try:
            check("_norm 可调用", EP._norm("测试 文本") == EP._norm("测试 文本"),
                  EP._norm("a b"))
        except Exception as e:
            check("_norm 可调用", False, repr(e))
    if hasattr(EP, "_number_coverage"):
        try:
            v = EP._number_coverage("满分 500 分", "满分 500 分")
            check("_number_coverage 可调用", v is not None, f"→{v}")
        except Exception as e:
            check("_number_coverage 可调用", False, repr(e))

# ---- 3. _exp_c2 依赖的 config 字段 ----
from base.config import config
for name in ("LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL", "FAQ_THRESHOLD",
             "PARENT_CHUNK_SIZE", "CHILD_CHUNK_SIZE", "CHUNK_OVERLAP",
             "RETRIEVAL_K", "CANDIDATE_M"):
    v = getattr(config, name, "<MISSING>")
    # API key 只报"是否已配置"，绝不打印明文：本脚本可能被分享/截图。
    shown = ("已配置(%d 字符)" % len(str(v))) if name == "LLM_API_KEY" else repr(v)
    check(f"config.{name}", hasattr(config, name), f"={shown}"[:70])

# ---- 4. prompt 开关真实存在且三版本互不相同 ----
try:
    from rag_qa.core.prompts import RAGPrompts
    tmpl = {}
    for v in ("v1", "v2", "v3"):
        os.environ["EXPERIMENT_PROMPT"] = v
        tmpl[v] = RAGPrompts.rag_prompt().template
    os.environ.pop("EXPERIMENT_PROMPT", None)
    check("prompt v1/v2/v3 三者互不相同",
          len({tmpl["v1"], tmpl["v2"], tmpl["v3"]}) == 3)
    for v in ("v1", "v2", "v3"):
        check(f"prompt {v} 非空", len(tmpl[v]) > 50, f"{len(tmpl[v])} 字符")
except Exception as e:
    check("RAGPrompts 三版本可切换", False, repr(e))

# ---- 5. 策略开关真实存在（_exp_c1 依赖） ----
try:
    from rag_qa.core.strategy_selector import StrategySelector
    import rag_qa.core.strategy_selector as SS
    src = inspect.getsource(SS.StrategySelector._call_llm)
    check("StrategySelector._call_llm 存在", True)
    check("_call_llm 是绑定方法（含 self）", "self" in src.split("\n")[0],
          src.strip().split("\n")[0][:60])
    tmpl = inspect.getsource(SS.StrategySelector.strategy_select)
    check("strategy_select 读取 EXPERIMENT_STRATEGY", "EXPERIMENT_STRATEGY" in tmpl)
    check("strategy_select 设置 self.last_strategy", "last_strategy" in tmpl)
except Exception as e:
    check("StrategySelector 契约", False, repr(e))

# ---- 6. IntegratedQASystem 契约（两脚本共用） ----
try:
    from faq_milvus import IntegratedQASystem
    check("IntegratedQASystem.query 是异步生成器函数",
          inspect.isasyncgenfunction(IntegratedQASystem.query))
    sig = inspect.signature(IntegratedQASystem.query)
    params = list(sig.parameters)
    check("query 形参含 trace",
          "trace" in params, f"params={params}")
except Exception as e:
    check("IntegratedQASystem 契约", False, repr(e))

# ---- 7. _exp_c1 里会替换的属性真实存在 ----
try:
    from rag_qa.core.rag_system import RAGSystem
    from rag_qa.core.strategy_selector import StrategySelector
    check("RAGSystem.__init__ 有 llm 属性",
          "llm" in inspect.getsource(RAGSystem.__init__))
    check("StrategySelector 有 _call_llm 方法",
          hasattr(StrategySelector, "_call_llm"))
    check("qa.rag 会在 __init__ 里创建",
          "self.rag" in inspect.getsource(IntegratedQASystem.__init__))
except Exception as e:
    check("替换点契约", False, repr(e))

print("=" * 78)
print(f"通过 {len(oks)} 项，失败 {len(fails)} 项")
print("=" * 78)
for o in oks:
    print(f"  OK   {o}")
if fails:
    print("-" * 78)
    for f in fails:
        print(f"  FAIL {f}")
print("\nRESULT_JSON " + json.dumps(
    {"pass": len(oks), "fail": len(fails), "failures": fails}, ensure_ascii=False))
raise SystemExit(1 if fails else 0)
