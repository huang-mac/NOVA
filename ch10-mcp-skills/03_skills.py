"""
===============================================
第十章 示例 3：Skills 技能包设计
===============================================
目标：设计 Skill 技能包，把相关的 Tools + Prompt + 知识打包在一起。

如果说 MCP 解决的是"怎么连"（通信协议），那 Skills 解决的就是"怎么管"（能力组织）。
"""

import json


# ============================================================
# Skill 技能包
# ============================================================
class Skill:
    """
    技能包 —— 把相关的 Tools + Prompt + 知识打包在一起。

    类比：
    - MCP Server 像是"工具箱"
    - Skill 像是"技能书"——告诉你面对什么问题时该用什么工具、怎么用

    Skill 和 Multi-Agent 的关系：
    - Multi-Agent 是"人"的分工（不同 Agent 扮演不同角色）
    - Skills 是"能力"的组织（同一个 Agent 可以加载不同技能包）
    - 两者可以结合：每个 Agent 默认加载自己的专属技能包
    """

    def __init__(self, name: str, description: str,
                 system_prompt: str = "",
                 required_tools: list = None,
                 required_servers: list = None,
                 knowledge: str = "",
                 examples: list = None):
        """
        Args:
            name: 技能包名称（唯一标识）
            description: 技能包功能描述（用于 LLM 匹配选择）
            system_prompt: 专属的系统提示词
            required_tools: 需要的工具名列表
            required_servers: 需要的 MCP Server 名称列表
            knowledge: 关联的知识文档内容
            examples: 示例对话列表
        """
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.required_tools = required_tools or []
        self.required_servers = required_servers or []
        self.knowledge = knowledge
        self.examples = examples or []

    def get_full_prompt(self, tool_descriptions: str = "") -> str:
        """获取完整的系统 Prompt（含工具描述和知识文档）。"""
        parts = [self.system_prompt]

        if tool_descriptions:
            parts.append(f"\n\n可用工具：\n{tool_descriptions}")

        if self.knowledge:
            parts.append(f"\n\n知识库：\n{self.knowledge}")

        if self.examples:
            parts.append("\n\n参考示例：")
            for ex in self.examples:
                parts.append(f"  用户：{ex['user']}\n  回复：{ex['assistant']}")

        return "".join(parts)

    def info(self) -> dict:
        """返回技能包信息。"""
        return {
            "name": self.name,
            "description": self.description,
            "required_tools": self.required_tools,
            "required_servers": self.required_servers,
            "has_knowledge": bool(self.knowledge),
            "example_count": len(self.examples),
        }


# ============================================================
# 创建客服场景的技能包
# ============================================================
def create_customer_service_skills():
    """创建客服场景的三个技能包。"""

    refund_skill = Skill(
        name="refund",
        description="处理退款、退货、换货请求",
        system_prompt="""你是星辰科技的退款专员。

处理流程：
1. 查询订单信息
2. 判断退货条件（7天无理由 / 质保期内）
3. 告知用户具体操作步骤

语气：亲切、专业、简洁。""",
        required_tools=["query_order", "check_refund_policy"],
        required_servers=["order_service", "product_service"],
        knowledge="""退款政策：
- 7天无理由退货：签收后 7 天内可无理由退换
- 15天质量问题：质量问题 15 天内免费换新
- 1年质保：非人为损坏 1 年内免费维修
- 退款方式：原路退回，3-5 个工作日到账""",
        examples=[
            {"user": "我买了 3 天能退吗", "assistant": "您购买 3 天，完全在 7 天无理由退货期内！请在 APP 中进入「我的订单」→ 选择对应订单 → 点击「申请退款」即可。退款会在 3-5 个工作日原路退回。"},
        ],
    )

    logistics_skill = Skill(
        name="logistics",
        description="查询物流和订单配送状态",
        system_prompt="""你是星辰科技的物流查询专员。

处理流程：
1. 查询订单信息确认商品
2. 查询物流信息获取当前位置
3. 告知用户物流详情

语气：简洁、高效。""",
        required_tools=["query_order", "query_logistics"],
        required_servers=["order_service"],
        examples=[
            {"user": "我的快递到哪了", "assistant": "您的 StarPods Pro 订单目前由顺丰速运配送，已到达北京分拨中心，预计明天送达。"},
        ],
    )

    tech_skill = Skill(
        name="tech_support",
        description="处理产品故障和使用咨询",
        system_prompt="""你是星辰科技的技术支持工程师。

处理流程：
1. 了解具体症状和产品型号
2. 给出逐步排查建议
3. 如果是硬件故障，建议用户申请售后维修

语气：耐心、专业，避免使用过于技术化的术语。""",
        required_tools=["query_order", "query_inventory"],
        required_servers=["order_service", "product_service"],
        knowledge="""常见问题排查：
1. 蓝牙连不上：确认蓝牙已开启 → 忘记已配对设备 → 重新配对
2. 有杂音：清洁充电触点 → 尝试另一台设备 → 重置耳机
3. 充不进电：更换充电线 → 清洁充电盒触点 → 检查是否进水
4. 单耳无声：检查是否耳机模式 → 尝试左右耳交换使用""",
        examples=[
            {"user": "耳机连不上手机", "assistant": "别着急，我来帮您排查。请问您使用的是哪个型号的耳机？请先确认手机蓝牙已开启，然后按以下步骤操作：1. 在手机蓝牙设置中「忘记」之前的配对记录 → 2. 将耳机放回充电盒，长按按钮 3 秒重置 → 3. 重新打开充电盒盖，在手机上搜索并连接。试试看能不能连上？"},
        ],
    )

    return [refund_skill, logistics_skill, tech_skill]


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第十章 示例 3：Skills 技能包设计")
    print("=" * 60)

    skills = create_customer_service_skills()

    print("\n--- 技能包列表 ---")
    for skill in skills:
        info = skill.info()
        print(f"\n  [{info['name']}] {info['description']}")
        print(f"    需要工具: {info['required_tools']}")
        print(f"    需要 Server: {info['required_servers']}")
        print(f"    有知识文档: {'是' if info['has_knowledge'] else '否'}")
        print(f"    示例数量: {info['example_count']}")

    # 演示完整 Prompt 生成
    print("\n--- 技能包 Prompt 生成示例 ---")
    refund_skill = skills[0]
    tool_descriptions = "- query_order: 查询订单详情\n- check_refund_policy: 查询退款政策"
    full_prompt = refund_skill.get_full_prompt(tool_descriptions)
    print(f"  退款技能包完整 Prompt（前 200 字）：\n{full_prompt[:200]}...")

    print("\n" + "=" * 60)
    print("关键要点：")
    print("  1. Skill 是能力模块，打包了 Prompt + 工具需求 + 知识 + 示例")
    print("  2. Skill 声明自己需要哪些工具（required_tools）")
    print("  3. Skill 声明自己需要哪些 MCP Server（required_servers）")
    print("  4. 知识文档和示例让 Agent 更专业、更一致")
    print("  5. Skill 和 Multi-Agent 不冲突，可以结合使用")
    print("=" * 60)
