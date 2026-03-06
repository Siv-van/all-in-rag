"""
生成集成模块
"""

import os
import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""
    
    def __init__(self, model_name: str = "kimi-k2-0711-preview", temperature: float = 0.1, max_tokens: int = 2048):
        """
        初始化生成集成模块
        
        Args:
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()
    
    def setup_llm(self):
        """初始化大语言模型"""
        logger.info(f"正在初始化LLM: {self.model_name}")

        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

        self.llm = MoonshotChat(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            moonshot_api_key=api_key
        )
        
        logger.info("LLM初始化完成")
    
    def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成基础回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            生成的回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的宠物猫养护专家。请根据以下猫咪养护知识，回答用户关于猫行为、健康、营养、护理或急救的问题。
用户问题: {question}
相关知识文档:
{context}
请遵循以下原则回答：
- 以文档内容为主，不要凭空猜测
- 给出清晰、可操作的建议（例如可以做/不应该做什么）
- 一旦涉及可能危及生命或需要专业处置的情况，要明确提醒用户尽快就医或联系兽医
- 如果信息不足，请坦诚说明，而不是编造答案
请在回答结尾提醒用户：本回答仅作为参考建议，不能替代专业兽医的诊断和治疗。
回答:""")

        # 使用LCEL构建链
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response
    
    def generate_step_by_step_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成分步骤回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            分步骤的详细回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的宠物猫养护专家。请根据以下猫咪养护知识，为用户提供分步骤的指导。
用户问题: {question}
相关知识文档:
{context}
请尽量按照下列结构组织答案（若不适用可适当调整或省略）：
## 问题概览
- 简要说明这是哪一类问题（如行为问题、健康风险、营养、日常护理、紧急情况）
- 给出风险级别的直观判断（例如：轻微/需要尽快关注/应立即就医）
## 可能的原因
- 列出文档中提到的常见原因
- 不要过度推测未在文档中出现的病因
## 建议的处理步骤
- 分步骤说明主人在家可以做的处理（例如环境调整、行为引导、简单急救等）
- 对每一步明确说明注意事项和不要做的事
## 何时必须就医
- 明确列出哪些症状或持续时间意味着“需要尽快带猫去医院或联系兽医”
## 预防与长期建议
- 如果文档中有预防或长期管理建议，可以总结在这里
注意：
- 所有建议应以文档内容为主，不要进行医疗诊断
- 避免给出具体药物剂量或替代兽医的治疗方案
- 在涉及中毒、休克、严重外伤等情况时，优先强调“立即就医”的重要性

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response
    
    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写 - 让大模型判断是否需要重写查询

        Args:
            query: 原始查询

        Returns:
            重写后的查询或原查询
        """
        prompt = PromptTemplate(
            template="""
你是一个智能查询分析助手。请分析用户关于“宠物猫养护”的问题，判断是否需要重写以提高文档搜索效果。
原始查询: {query}
分析规则：
1. **具体明确的查询**（直接返回原查询）：
   - 已经包含明确主题或问题：如“猫中毒怎么办”“两只猫总打架怎么处理”
   - 针对特定场景的问法：如“老年猫虚脱前有哪些表现”“猫梳毛时总咬人怎么办”
   - 问的是具体措施：如“猫误食巧克力需要做什么”“猫肥胖应该怎么调整饮食”
2. **模糊不清的查询**（需要重写）：
   - 过于宽泛：如“养猫”“健康问题”“猫最近不太对劲”
   - 缺乏关键信息：如“行为问题”“饮食”“紧急情况”
   - 口语化表达：如“猫怎么搞的”“感觉猫有点奇怪”
重写原则：
- 保持原意不变
- 增加与猫养护相关的关键词（如“行为问题”“日常护理”“紧急处理”“中毒急救”等）
- 使查询更接近文档标题或段落主题
- 保持简洁
示例：
- “养猫” → “宠物猫日常养护指南”
- “健康问题” → “宠物猫常见健康问题及处理建议”
- “猫最近不太对劲” → “猫行为异常时的常见原因及观察要点”
- “有没有紧急情况要注意” → “宠物猫常见紧急情况及处理原则”
- “猫中毒怎么办” → “猫中毒时的急救措施和就医建议”（也可以保持原查询）
请输出最终查询（如果不需要重写就返回原查询）:""",
            input_variables=["query"]
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query).strip()

        # 记录重写结果
        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response



    def query_router(self, query: str) -> str:
        """
        查询路由 - 根据查询类型选择不同的处理方式

        Args:
            query: 用户查询

        Returns:
            路由类型 ('list', 'detail', 'general')
        """
        prompt = ChatPromptTemplate.from_template("""
根据用户关于“宠物猫”的问题，将其分类为以下三种类型之一：
1. 'list' - 用户想要获取相关主题或问题的“列表/推荐”，只需要列出条目或方向
   例如：推荐几个需要重点注意的猫咪紧急情况、列出常见的猫行为问题、有哪些老年猫常见疾病

2. 'detail' - 用户想要某个具体问题的“详细处理方法或分步骤指导”
   例如：猫中毒怎么办、猫虚脱时具体应该怎么处理、两只猫总是打架如何一步步干预

3. 'general' - 其他一般性或科普类问题
   例如：老年猫需要注意什么、如何判断猫是否肥胖、为什么要给猫定期梳毛

请只返回分类结果：list、detail 或 general

用户问题: {query}

分类结果:""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke(query).strip().lower()

        # 确保返回有效的路由类型
        if result in ['list', 'detail', 'general']:
            return result
        else:
            return 'general'  # 默认类型

    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成列表式回答 - 适用于推荐类查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            列表式回答
        """
        if not context_docs:
            return "抱歉，没有找到与该问题相关的猫咪养护主题。"

        # 提取主题名称（来自文档元数据）
        topic_names = []
        for doc in context_docs:
            topic_name = doc.metadata.get('topic_name', '未命名主题')
            if topic_name not in topic_names:
                topic_names.append(topic_name)

        if len(topic_names) == 1:
            return f"为你找到一个最相关的猫咪养护主题：{topic_names[0]}"
        elif len(topic_names) <= 3:
            return "为你列出几个相关的猫咪养护主题：\n" + "\n".join(
                f"{i+1}. {name}" for i, name in enumerate(topic_names)
            )
        else:
            return (
                "为你列出几个最相关的猫咪养护主题：\n"
                + "\n".join(f"{i+1}. {name}" for i, name in enumerate(topic_names[:3]))
                + f"\n\n另外还有 {len(topic_names)-3} 个相关主题，你也可以尝试更具体地提问来缩小范围。"
            )

    def generate_basic_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成基础回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            生成的回答片段
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的宠物猫养护专家。请根据以下猫咪养护知识，回答用户关于猫行为、健康、营养、护理或急救的问题。
用户问题: {question}
相关知识文档:
{context}
请遵循以下原则回答：
- 以文档内容为主，不要凭空猜测
- 给出清晰、可操作的建议（例如可以做/不应该做什么）
- 一旦涉及可能危及生命或需要专业处置的情况，要明确提醒用户尽快就医或联系兽医
- 如果信息不足，请坦诚说明，而不是编造答案
请在回答结尾提醒用户：本回答仅作为参考建议，不能替代专业兽医的诊断和治疗。
回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def generate_step_by_step_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成详细步骤回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            详细步骤回答片段
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的宠物猫养护专家。请根据以下猫咪养护知识，为用户提供分步骤的指导。
用户问题: {question}
相关知识文档:
{context}
请尽量按照下列结构组织答案（若不适用可适当调整或省略）：
## 问题概览
- 简要说明这是哪一类问题（如行为问题、健康风险、营养、日常护理、紧急情况）
- 给出风险级别的直观判断（例如：轻微/需要尽快关注/应立即就医）
## 可能的原因
- 列出文档中提到的常见原因
- 不要过度推测未在文档中出现的病因
## 建议的处理步骤
- 分步骤说明主人在家可以做的处理（例如环境调整、行为引导、简单急救等）
- 对每一步明确说明注意事项和不要做的事
## 何时必须就医
- 明确列出哪些症状或持续时间意味着“需要尽快带猫去医院或联系兽医”
## 预防与长期建议
- 如果文档中有预防或长期管理建议，可以总结在这里
注意：
- 所有建议应以文档内容为主，不要进行医疗诊断
- 避免给出具体药物剂量或替代兽医的治疗方案
- 在涉及中毒、休克、严重外伤等情况时，优先强调“立即就医”的重要性

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def _build_context(self, docs: List[Document], max_length: int = 2000) -> str:
        """
        构建上下文字符串
        
        Args:
            docs: 文档列表
            max_length: 最大长度
            
        Returns:
            格式化的上下文字符串
        """
        if not docs:
            return "暂无相关猫咪养护信息。"
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(docs, 1):
            # 添加元数据信息
            metadata_info = f"【知识条目 {i}】"
            if 'topic_name' in doc.metadata:
                metadata_info += f" 标题: {doc.metadata['topic_name']}"
            if 'category' in doc.metadata:
                metadata_info += f" | 分类: {doc.metadata['category']}"
            if 'difficulty' in doc.metadata:
                metadata_info += f" | 难度: {doc.metadata['difficulty']}"
            
            # 构建文档文本
            doc_text = f"{metadata_info}\n{doc.page_content}\n"
            
            # 检查长度限制
            if current_length + len(doc_text) > max_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n" + "="*50 + "\n".join(context_parts)
