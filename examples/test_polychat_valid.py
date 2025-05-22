import asyncio
import random
import json
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://polychat.echooo.link/api/v1",
    api_key="123"
)

assistant_id = "6819d2df205fdfc739a36cfc"

test_questions = [
    "搜索女士羽绒服", "搜索补水精华", "搜索游戏鼠标", "搜索智能手环",
    "搜索降噪耳机", "搜索微单相机", "搜索婴儿纸尿裤", "搜索扫地机器人",
    "搜索男士运动鞋", "搜索美白面膜", "搜索机械键盘", "搜索蓝牙音箱",
    "搜索运动背包", "搜索眼部精华", "搜索儿童玩具",
    "推荐跑步运动鞋", "推荐车载空气净化器",
    "退货政策", "支付方式", "发票问题", "售后服务流程",
    "17472213709029467", "我的订单17472213709029467"
]

def is_valid_json_object(text: str) -> bool:
    try:
        obj = json.loads(text)
        return isinstance(obj, dict)
    except Exception:
        return False

async def send_user_message(thread_id: str, content: str):
    await client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content
    )

async def get_assistant_response(thread_id: str) -> str:
    run = await client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        stream=True,
    )
    output_text = ""
    async for chunk in run:
        if hasattr(chunk, "data") and hasattr(chunk.data, "delta"):
            content_blocks = chunk.data.delta.content or []
            for block in content_blocks:
                if block.type == "text" and block.text and block.text.value:
                    output_text += block.text.value
    return output_text

async def run_thread_session(thread_id: str, thread_index: int, turns: int) -> tuple[str, int]:
    log = f"\n====== 测试线程 {thread_index} ======\n"
    success_count = 0

    for turn in range(1, turns + 1):
        await asyncio.sleep(3)
        question = random.choice(test_questions)

        if turn == 1:
            log += f"\n[初始化对话]\n"
            await send_user_message(thread_id, "SYSTEM:用户基础信息如下:\n用户ID:2505145019563\n用户语言:zh_CN\n用户货币:USD\n输出语言请以用户语言为准，商品价格请以用户货币为准")

        log += f"\n--- 第 {turn} 轮问题 ---\n用户提问：{question}\n"

        try:
            await send_user_message(thread_id, question)
            output = await get_assistant_response(thread_id)
            log += f"返回内容：{output}\n"

            if is_valid_json_object(output):
                log += "合法 JSON 响应\n"
                success_count += 1
            else:
                log += "非法 JSON 响应\n"

        except Exception as e:
            log += f"异常发生：{str(e)}\n"

    return log, success_count

async def run_test_round(thread_count: int, turns_per_thread: int) -> tuple[int, int]:
    threads = [await client.beta.threads.create() for _ in range(thread_count)]
    print(f"创建 {thread_count} 个线程：{[t.id for t in threads]}")
    tasks = [run_thread_session(t.id, i + 1, turns_per_thread) for i, t in enumerate(threads)]
    results = await asyncio.gather(*tasks)
    
    total_success = sum(success for _, success in results)
    for log, _ in results:
        print(log)

    return total_success, thread_count * turns_per_thread

async def main():
    thread_count = 2
    turns_per_thread = 20
    repeat_rounds = 10

    total_runs = 0
    total_success = 0

    for current_round in range(1, repeat_rounds + 1):
        print(f"\n==== 第 {current_round} 轮测试 ====")
        try:
            success, runs = await run_test_round(thread_count, turns_per_thread)
            total_success += success
            total_runs += runs

            print(f"本轮合法 JSON：{success}/{runs}")
            print(f"累计合法 JSON：{total_success}/{total_runs}，合法率：{total_success / total_runs:.2%}")

        except KeyboardInterrupt:
            print("\n>>> 手动终止 <<<")
            break
        except Exception as e:
            print(f"本轮出现异常：{str(e)}")

if __name__ == "__main__":
    asyncio.run(main())