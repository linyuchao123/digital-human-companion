"""Conservative local signals for psychoeducation, without a classification API call."""
import re

def declines_knowledge(text):
    return bool(re.search(r'不要科普|不用科普|别讲道理|只(?:想|要)?(?:让你)?听我说|不需要建议|不要建议|只想倾诉',text))

def requests_knowledge(text):
    if declines_knowledge(text):
        return False
    if re.search(r'(?:推荐|找|听|播放|写).*(?:歌曲|歌|小说|故事)',text):
        return False
    return bool(re.search(
        r'心理学|心理健康|心理韧性|认知行为|(?<![a-z])CBT(?![a-z])|灾难化|非黑即白|思维记录|担忧时间|正念|'
        r'失眠|睡不着|睡不好|睡前.*(?:想|担心)|'
        r'不会(?:说不|拒绝)|不敢拒绝|讨好别人|人际边界|关系边界|'
        r'亲人去世|失去亲人|丧亲|哀伤|照护压力|照顾(?:老人|家人).*累|'
        r'任务太多|不堪重负|反复担忧|一直担心|不完美就是失败|'
        r'(?:怎么|如何).*(?:安慰|调节情绪|缓解压力|处理冲突)|'
        r'(?:什么是|了解|介绍).*(?:自我关怀|认知重评|问题解决)',text,re.I))
