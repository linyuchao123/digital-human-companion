"""Conservative local signals for psychoeducation, without a classification API call."""
import re

def declines_knowledge(text):
    return bool(re.search(r'不要科普|不用科普|别讲道理|只(?:想|要)?(?:让你)?听我说|不需要建议|不要建议|只想倾诉',text))

def implicit_psychological_context(text):
    """Local personal-context signals, not a diagnosis or a complete classifier."""
    return bool(re.search(
        r'(?:躺(?:在)?床上|一到晚上|睡前).{0,80}(?:脑子.{0,20}(?:开会|停不下来)|还在想|一直想)|'
        r'(?:觉得|感到|我|最近).{0,40}(?:孤单|格格不入|局外人|融不进去|没人陪)|'
        r'(?:妈妈|爸爸|母亲|父亲|爷爷|奶奶|外公|外婆).{0,15}去世|'
        r'(?:我|自己).{0,40}(?:没用|太苛刻|自责|不好意思说不|不敢说不)|'
        r'(?:工作|任务).{0,30}(?:一大堆|堆在一起|优先级)|'
        r'(?:我|最近|还是|一直).{0,15}(?:睡不好|睡不着|失眠)',text))

def requests_knowledge(text):
    if declines_knowledge(text):
        return False
    if implicit_psychological_context(text):
        return True
    if re.search(r'(?:推荐|找|听|播放|写).*(?:歌曲|歌|小说|故事)',text):
        return False
    return bool(re.search(
        r'心理学|心理健康|心理咨询|心理韧性|认知行为|(?<![a-z])CBT(?![a-z])|灾难化|非黑即白|思维记录|担忧时间|正念|'
        r'失眠|睡不着|睡不好|睡前.*(?:想|担心)|'
        r'不会(?:说不|拒绝)|不敢拒绝|讨好别人|人际边界|关系边界|'
        r'亲人去世|失去亲人|丧亲|哀伤|照护压力|照顾(?:老人|家人).*累|'
        r'任务太多|不堪重负|反复担忧|一直担心|不完美就是失败|'
        r'(?:怎么|如何).*(?:安慰|调节情绪|缓解压力|处理冲突)|'
        r'(?:什么是|了解|介绍).*(?:自我关怀|认知重评|问题解决)',text,re.I))
