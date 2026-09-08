import unittest

from services.agent import ChatMessage, DigitalXinyuWorkflow, InMemoryMemoryStore, RiskLevel


class FailingProvider:
    async def generate(self, messages):
        raise AssertionError("高风险路径不应调用普通对话模型")


class RecordingProvider:
    def __init__(self):
        self.messages = []

    async def generate(self, messages):
        self.messages = list(messages)
        return "我记得，我们继续聊。"


class DigitalXinyuWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_chat_runs_through_companion_and_avatar_nodes(self):
        workflow = DigitalXinyuWorkflow()

        result = await workflow.run(
            user_text="今天工作特别累",
            trace_id="trace-normal",
            session_id="session-normal",
        )

        self.assertEqual(result["safety"].risk_level, RiskLevel.LOW)
        self.assertEqual(
            result["execution_path"],
            [
                "safety_triage",
                "intent_router",
                "emotion_analyzer",
                "knowledge_retriever",
                "companion",
                "avatar_director",
            ],
        )
        self.assertEqual(set(result["node_timings_ms"]), set(result["execution_path"]))
        self.assertTrue(all(value >= 0 for value in result["node_timings_ms"].values()))
        self.assertEqual(result["intent"], "emotional_support")
        self.assertEqual(result["avatar_command"].motion, "Respond")
        self.assertTrue(result["final_response"])

    async def test_high_risk_chat_bypasses_companion_provider(self):
        workflow = DigitalXinyuWorkflow(provider=FailingProvider())

        result = await workflow.run(
            user_text="我不想活了",
            trace_id="trace-risk",
            session_id="session-risk",
        )

        self.assertEqual(result["safety"].risk_level, RiskLevel.HIGH)
        self.assertEqual(
            result["execution_path"],
            ["safety_triage", "safe_response", "emotion_analyzer", "avatar_director"],
        )
        self.assertEqual(set(result["node_timings_ms"]), set(result["execution_path"]))
        self.assertEqual(result["avatar_command"].motion, "Comfort")
        self.assertEqual(result["emotion_context"].emotion, "Concerned")
        self.assertNotIn("draft_response", result)
        self.assertEqual(result["messages"][-1].role, "assistant")

    async def test_workflow_passes_previous_turns_to_provider(self):
        provider = RecordingProvider()
        workflow = DigitalXinyuWorkflow(provider=provider)

        result = await workflow.run(
            user_text="那我应该怎么办？",
            trace_id="trace-context",
            session_id="session-context",
            messages=[
                ChatMessage(role="user", content="我最近工作压力很大"),
                ChatMessage(role="assistant", content="愿意说说压力来自哪里吗？"),
            ],
        )

        self.assertEqual([message.content for message in provider.messages], [
            "我最近工作压力很大",
            "愿意说说压力来自哪里吗？",
            "那我应该怎么办？",
        ])
        self.assertEqual(result["messages"][-1].content, "我记得，我们继续聊。")

    async def test_workflow_bounds_long_conversation_context(self):
        provider = RecordingProvider()
        workflow = DigitalXinyuWorkflow(provider=provider)
        history = [ChatMessage(role="user", content=f"消息{i}") for i in range(60)]

        result = await workflow.run(
            user_text="最新消息",
            trace_id="trace-bounded",
            session_id="session-bounded",
            messages=history,
        )

        self.assertLessEqual(len(provider.messages), 39)
        self.assertLessEqual(len(result["messages"]), 40)
        self.assertEqual(provider.messages[-1].content, "最新消息")

    async def test_emotion_node_directs_anxious_avatar_to_listen(self):
        provider = RecordingProvider()
        workflow = DigitalXinyuWorkflow(provider=provider)

        result = await workflow.run(
            user_text="我最近总是焦虑和睡不着",
            trace_id="trace-anxiety",
            session_id="session-anxiety",
        )

        self.assertEqual(result["emotion_context"].emotion, "Anxiety")
        self.assertEqual(result["avatar_command"].motion, "Listen")
        self.assertEqual(result["execution_path"][3], "knowledge_retriever")
        self.assertGreaterEqual(len(result["retrieved_knowledge"]), 1)
        self.assertEqual(result["tool_calls"][0].name, "psychology_knowledge")
        self.assertEqual(provider.messages[0].role, "system")
        self.assertIn("心理教育知识", provider.messages[0].content)

    async def test_memory_nodes_require_authenticated_consent(self):
        store = InMemoryMemoryStore()
        await store.remember(8, "用户喜欢睡前听轻音乐", "preference")
        provider = RecordingProvider()
        workflow = DigitalXinyuWorkflow(provider=provider, memory_store=store)

        result = await workflow.run(
            user_text="我喜欢睡前听轻音乐，最近有点焦虑",
            trace_id="trace-memory",
            session_id="session-memory",
            user_id=8,
            memory_consent=True,
        )

        self.assertIn("memory_retriever", result["execution_path"])
        self.assertIn("memory_writer", result["execution_path"])
        self.assertEqual(len(result["retrieved_memories"]), 1)
        self.assertTrue(any("长期记忆" in message.content for message in provider.messages))

    async def test_high_risk_chat_never_reads_or_writes_memory(self):
        store = InMemoryMemoryStore()
        workflow = DigitalXinyuWorkflow(provider=FailingProvider(), memory_store=store)

        result = await workflow.run(
            user_text="我不想活了",
            trace_id="trace-risk-memory",
            session_id="session-risk-memory",
            user_id=9,
            memory_consent=True,
        )

        self.assertNotIn("memory_retriever", result["execution_path"])
        self.assertNotIn("memory_writer", result["execution_path"])
        self.assertEqual(await store.search(9, "不想活"), [])


if __name__ == "__main__":
    unittest.main()
