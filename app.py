# Web框架核心
from fastapi import FastAPI, WebSocket, HTTPException, Query, Depends, Request, Response
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
# 系统与工具
import os
import asyncio
import json
import uuid
import time
import re
from typing import Optional, List, Dict, Any
# 业务层依赖
from pydantic import BaseModel
from faq_milvus import IntegratedQASystem
from base.config import config
from mysql_qa.db.mysql_client import MySQLClient


# 应用实例化
app = FastAPI(title="问答系统API", description="集成FAQ和RAG的智能问答系统")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 实例初始化
qa_system = IntegratedQASystem()
mysql_client = MySQLClient()

GREETING_PATTERNS = [
    {
        "pattern": r"^(你好|您好|hi|hello)",
        "response": "你好！我是考研就业智能助手，专注考研择校、专业方向与就业答疑，很高兴为你服务！"
    },
    {
        "pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",
        "response": "我是考研就业智能助手，你的考研与就业咨询助手，致力于提供考研政策和就业方向相关的解答！"
    },
    {
        "pattern": r"^(在吗|在不在|有人吗)",
        "response": "我在！我是考研就业智能助手，随时为你解答考研与就业问题！"
    },
    {
        "pattern": r"^(干嘛呢|你在干嘛|做什么)",
        "response": "我正在待命，随时为你解答考研与就业相关的问题！有什么我可以帮你的？"
    }
]

# 检查日常问候
def check_greeting(query: str) -> Optional[str]:
    query_text = query.strip()
    for pattern_info in GREETING_PATTERNS:
        if re.match(pattern_info["pattern"], query_text, re.IGNORECASE):
            return pattern_info["response"]
    return None
# 创建新会话
@app.post("/api/create_session")
async def create_session():
    session_id = str(uuid.uuid4())  # 生成唯一会话 ID
    return {"session_id": session_id}  # 返回会话 ID

# 查询历史消息
@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    try:
        # 获取指定会话的历史记录
        history = mysql_client.fetch_recent_history(session_id)
        # 获取该会话的总对话轮数
        total_count = mysql_client.get_session_count(session_id)
        # 返回会话 ID、历史记录和总数
        return {
            "session_id": session_id,
            "history": history,
            "total_count": total_count
        }
    except Exception as e:
        # 抛出 HTTP 异常，包含错误信息
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")

# 查询会话总数
@app.get("/api/session_count/{session_id}")
async def get_session_count(session_id: str):
    try:
        total_count = mysql_client.get_session_count(session_id)
        return {"session_id": session_id, "total_count": total_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话数失败: {str(e)}")


# 查询所有会话列表
@app.get("/api/sessions")
async def get_all_sessions(response: Response):
    try:
        sessions = mysql_client.fetch_all_sessions()
        return {"sessions": sessions, "total": len(sessions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")

# 清除所有会话历史
@app.delete("/api/history")
async def clear_all_history():
    try:
        success = mysql_client.clear_all_sessions_history()
        if success:
            return {"status": "success", "message": "所有历史记录已清除"}
        else:
            raise HTTPException(status_code=500, detail="清除历史记录失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清除历史记录失败: {str(e)}")

@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    # 接受客户端建立WebSocket长连接
    await websocket.accept()
    try:
        # 循环监听客户端消息，理论支持同一连接多次发送提问
        while True:
            # 接收前端发来的JSON字符串消息
            data = await websocket.receive_text()
            # 字符串转为字典，提取请求参数
            request_data = json.loads(data)
            # 用户提问内容
            query = request_data.get("query")
            # 知识库数据源过滤条件，用于限定检索范围
            source_filter = request_data.get("source_filter")
            # 会话ID：前端传则复用，不传自动生成UUID作为会话唯一标识
            session_id = request_data.get("session_id", str(uuid.uuid4()))
            # 记录本轮问答起始时间，用于计算整体耗时
            start_time = time.time()

            # 参数校验：提问为空/全空格，返回错误并继续等待下一条消息
            if not query or not query.strip():
                await websocket.send_json({"type": "error", "error": "查询内容不能为空"})
                continue

            # 向前端发送开始标记，告知前端准备接收流式回答
            await websocket.send_json({"type": "start", "session_id": session_id})

            # ========== 分支1：问候语快捷匹配逻辑 ==========
            # 校验用户输入是否为你好、在吗、你是谁等固定问候语
            greeting_response = check_greeting(query)
            if greeting_response:
                # 将用户问题+AI固定问候回答存入MySQL会话历史表
                mysql_client.update_session_history(session_id, query, greeting_response)
                # 逐字符推送问候文本，模拟打字机输出效果
                for char in greeting_response:
                    await websocket.send_json({
                        "type": "token",
                        "token": char,
                        "session_id": session_id
                    })
                # 全部字符推送完成，发送结束标识，附带总耗时
                await websocket.send_json({
                    "type": "end",
                    "session_id": session_id,
                    "is_complete": True,
                    "processing_time": time.time() - start_time
                })
                # 本轮问候语已处理完毕，继续等待下一条消息
                # 注意用 continue：原实现是 break，会跳出 while 并进入 finally 关闭整个 WebSocket 连接
                continue

            # ========== 分支2：正式RAG知识库问答业务逻辑 ==========
            try:
                # qa_system.query 是异步生成器，持续返回(文本分片, 是否输出完成)
                # 必须用 async for：query 内部用 await asyncio.sleep 做流式节流，同步 for 无法消费
                async for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                    # 推送单段回答分片给前端
                    await websocket.send_json({
                        "type": "token",
                        "token": token,
                        "session_id": session_id
                    })
                    # 生成器标记回答全部输出完毕，发送结束包并终止循环
                    if is_complete:
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                        break
            # RAG检索/大模型生成过程中出现异常，向前端返回错误信息
            except Exception as query_error:
                await websocket.send_json({"type": "error", "error": f"查询处理失败: {str(query_error)}"})
                # 异常后终止本轮问答，跳出循环关闭连接
                break
    # 捕获客户端主动断开连接事件（关闭页面、刷新、手动关闭ws）
    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected: code={e.code}, reason={e.reason}")
    # 捕获其余所有未知异常
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        # 尝试向前端推送错误提示；连接已断开时发送会报错，用try忽略
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except:
            pass
    # 无论正常结束、异常、客户端断开，最终都会执行，关闭ws连接释放资源
    finally:
        try:
            await websocket.close()
        except Exception as e:
            print(f"Error closing WebSocket: {str(e)}")

# 健康检查接口
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# 获取学科类别
@app.get("/api/sources")
async def get_sources():
    return {"sources": config.VALID_SOURCES}


# 挂载静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse(
        "static/index.html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)