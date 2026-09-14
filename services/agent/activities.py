"""可选的日常陪伴活动，不承诺情绪治疗效果。"""
from .state import ActivityCard


def requests_activities(text: str) -> bool:
    return ("无聊" in text and any(word in text for word in ("怎么办", "做什么", "干什么"))) or (
        any(word in text for word in ("推荐", "安排", "给我", "帮我选"))
        and any(word in text for word in ("小活动", "陪伴活动", "小任务", "活动"))
    )


def activity_cards() -> list[ActivityCard]:
    return [
        ActivityCard(id="tidy", title="整理一个小角落", minutes=3,
                     description="选桌面的一小块，把三件物品放回原位。做到多少都可以。"),
        ActivityCard(id="journal", title="写下此刻的心情", minutes=2,
                     description="在自己的纸上写一句：我现在感觉……。不需要分享，也不用评判。"),
        ActivityCard(id="music", title="听一首喜欢的歌", minutes=5,
                     description="用你常用的音乐软件选一首喜欢的歌。不想继续时可以随时停下。"),
    ]
