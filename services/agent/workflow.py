from __future__ import annotations

from time import perf_counter
from typing import Any, Literal, Sequence

from langgraph.graph import END, START, StateGraph

from .emotion import EmotionAnalyzer
from .knowledge import BuiltInKnowledgeRetriever, KnowledgeRetriever
from .providers import CompanionProvider, FakeCompanionProvider
from .safety import SafetyTriage
from .state import AgentState, AvatarCommand, ChatMessage, ToolCallRecord


SAFE_RESPONSE = (
    "听起来你正在经历非常艰难的时刻。请先不要独自承受，"
    "尽快联系一位你信任的人陪在身边。如果你正面临立即危险，"
    "请马上联系当地紧急服务或前往最近的急诊机构。"
)
MAX_CONTEXT_MESSAGES = 40


class DigitalXinyuWorkflow:
    def __init__(
        self,
        provider: CompanionProvider | None = None,
        safety_triage: SafetyTriage | None = None,
        emotion_analyzer: EmotionAnalyzer | None = None,
        knowledge_retriever: KnowledgeRetriever | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self._provider = provider or FakeCompanionProvider()
        self._safety_triage = safety_triage or SafetyTriage()
        self._emotion_analyzer = emotion_analyzer or EmotionAnalyzer()
        self._knowledge_retriever = knowledge_retriever or BuiltInKnowledgeRetriever()
        self.graph = self._build_graph().compile(checkpointer=checkpointer)

    def _build_graph(self) -> StateGraph[AgentState]:
        graph = StateGraph(AgentState)
        graph.add_node("safety_triage", self._run_safety_triage)
        graph.add_node("safe_response", self._create_safe_response)
        graph.add_node("intent_router", self._route_intent)
        graph.add_node("emotion_analyzer", self._analyze_emotion)
        graph.add_node("knowledge_retriever", self._retrieve_knowledge)
        graph.add_node("companion", self._generate_companion_response)
        graph.add_node("avatar_director", self._direct_avatar)

        graph.add_edge(START, "safety_triage")
        graph.add_conditional_edges(
            "safety_triage",
            self._select_safety_route,
            {"safe_response": "safe_response", "intent_router": "intent_router"},
        )
        graph.add_edge("safe_response", "emotion_analyzer")
        graph.add_edge("intent_router", "emotion_analyzer")
        graph.add_conditional_edges(
            "emotion_analyzer",
            self._select_response_route,
            {
                "knowledge_retriever": "knowledge_retriever",
                "companion": "companion",
                "avatar_director": "avatar_director",
            },
        )
        graph.add_edge("knowledge_retriever", "companion")
        graph.add_edge("companion", "avatar_director")
        graph.add_edge("avatar_director", END)
        return graph

    @staticmethod
    def _complete_node(
        state: AgentState,
        node: str,
        started_at: float,
        **updates: Any,
    ) -> dict[str, Any]:
        return {
            **updates,
            "execution_path": [*state.get("execution_path", []), node],
            "node_timings_ms": {
                **state.get("node_timings_ms", {}),
                node: round((perf_counter() - started_at) * 1000, 3),
            },
        }

    async def _run_safety_triage(self, state: AgentState) -> dict[str, Any]:
        started_at = perf_counter()
        return self._complete_node(
            state,
            "safety_triage",
            started_at,
            safety=self._safety_triage.evaluate(state.get("user_text", "")),
        )

    @staticmethod
    def _select_safety_route(state: AgentState) -> Literal["safe_response", "intent_router"]:
        if state["safety"].requires_safe_response:
            return "safe_response"
        return "intent_router"

    async def _create_safe_response(self, state: AgentState) -> dict[str, Any]:
        started_at = perf_counter()
        messages = [
            *state.get("messages", []),
            ChatMessage(role="user", content=state["user_text"]),
            ChatMessage(role="assistant", content=SAFE_RESPONSE),
        ][-MAX_CONTEXT_MESSAGES:]
        return self._complete_node(
            state,
            "safe_response",
            started_at,
            final_response=SAFE_RESPONSE,
            messages=messages,
        )

    async def _route_intent(self, state: AgentState) -> dict[str, Any]:
        started_at = perf_counter()
        text = state.get("user_text", "")
        emotional_keywords = ("难过", "累", "焦虑", "孤独", "压力", "害怕")
        intent = "emotional_support" if any(word in text for word in emotional_keywords) else "chat"
        return self._complete_node(
            state,
            "intent_router",
            started_at,
            intent=intent,
        )

    async def _analyze_emotion(self, state: AgentState) -> dict[str, Any]:
        started_at = perf_counter()
        emotion = self._emotion_analyzer.analyze(state["user_text"], state["safety"])
        return self._complete_node(
            state,
            "emotion_analyzer",
            started_at,
            emotion_context=emotion,
        )

    @staticmethod
    def _select_response_route(
        state: AgentState,
    ) -> Literal["knowledge_retriever", "companion", "avatar_director"]:
        if state["safety"].requires_safe_response:
            return "avatar_director"
        if state.get("intent") == "emotional_support":
            return "knowledge_retriever"
        return "companion"

    async def _retrieve_knowledge(self, state: AgentState) -> dict[str, Any]:
        started_at = perf_counter()
        snippets = list(await self._knowledge_retriever.retrieve(state["user_text"], top_k=3))
        elapsed_ms = round((perf_counter() - started_at) * 1000, 3)
        return self._complete_node(
            state,
            "knowledge_retriever",
            started_at,
            retrieved_knowledge=snippets,
            tool_calls=[
                *state.get("tool_calls", []),
                ToolCallRecord(
                    name="psychology_knowledge",
                    reason=state.get("intent", ""),
                    status="completed",
                    elapsed_ms=elapsed_ms,
                ),
            ],
        )

    async def _generate_companion_response(self, state: AgentState) -> dict[str, Any]:
        started_at = perf_counter()
        conversation_messages: Sequence[ChatMessage] = [
            *state.get("messages", []),
            ChatMessage(role="user", content=state["user_text"]),
        ][-(MAX_CONTEXT_MESSAGES - 1):]
        knowledge = state.get("retrieved_knowledge", [])
        provider_messages = list(conversation_messages)
        if knowledge:
            context = "\n".join(
                f"[{index}] {item.content}（来源：{item.source}）"
                for index, item in enumerate(knowledge, 1)
            )
            provider_messages.insert(0, ChatMessage(
                role="system",
                content=f"以下是可参考的心理教育知识，不要将其当作医疗诊断：\n{context}",
            ))
        response = await self._provider.generate(provider_messages)
        return self._complete_node(
            state,
            "companion",
            started_at,
            draft_response=response,
            final_response=response,
            messages=[*conversation_messages, ChatMessage(role="assistant", content=response)][
                -MAX_CONTEXT_MESSAGES:
            ],
        )

    async def _direct_avatar(self, state: AgentState) -> dict[str, Any]:
        started_at = perf_counter()
        emotion = state["emotion_context"].emotion
        presentations = {
            "Concerned": (0.7, "Comfort"),
            "Sad": (0.55, "Listen"),
            "Anxiety": (0.55, "Listen"),
            "Happy": (0.6, "Respond"),
            "Neutral": (0.4, "Respond"),
        }
        intensity, motion = presentations[emotion]
        command = AvatarCommand(
            emotion=emotion,
            intensity=intensity,
            motion=motion,
            speaking_state="speaking",
        )
        return self._complete_node(
            state,
            "avatar_director",
            started_at,
            avatar_command=command,
        )

    async def run(
        self,
        *,
        user_text: str,
        trace_id: str,
        session_id: str,
        messages: Sequence[ChatMessage] = (),
        config: dict[str, Any] | None = None,
    ) -> AgentState:
        initial_state: AgentState = {
            "trace_id": trace_id,
            "thread_id": session_id,
            "session_id": session_id,
            "turn_id": 1,
            "user_text": user_text,
            "messages": list(messages)[-(MAX_CONTEXT_MESSAGES - 2):],
            "execution_path": [],
            "node_timings_ms": {},
            "retrieved_knowledge": [],
            "tool_calls": [],
            "cancelled": False,
            "memory_consent": False,
            "errors": [],
        }
        return await self.graph.ainvoke(initial_state, config=config)
