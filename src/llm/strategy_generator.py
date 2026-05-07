"""Credit strategy generation with Deepseek and compliance re-checking."""

from __future__ import annotations

import json
from typing import Any, Dict

from .deepseek_client import DeepseekClient


class CreditStrategyGenerator:
    """Generate compliant credit strategies and customer communication copy."""

    COMPLIANCE_RED_LINES = (
        "合规红线必须严格遵守：禁止暴力催收、禁止威胁辱骂、禁止虚假承诺、"
        "禁止诱导客户提供虚假信息、禁止承诺未经审批的减免或展期。"
    )

    def __init__(self) -> None:
        """Initialize the generator and its Deepseek API client."""

        self.client = DeepseekClient()

    def _build_prompt(self, user_info: Dict[str, Any], scenario: str) -> str:
        """Build a scenario-specific prompt from user information."""

        user_profile = json.dumps(user_info, ensure_ascii=False, indent=2)
        scenario_templates = {
            "正常维护": (
                "你是一名资深客户运营专家，需要为优质或正常客户制定维护策略。\n"
                "请输出以下内容：\n"
                "1. 复贷建议\n"
                "2. 权益方案\n"
                "3. 维护话术\n"
                "4. 风险关注点\n"
                "要求：语气专业、友好、稳健，突出客户价值和长期关系维护。"
            ),
            "风险预警": (
                "你是一名风险预警专员，需要为存在潜在还款压力的客户制定提醒策略。\n"
                "请输出以下内容：\n"
                "1. 还款提醒\n"
                "2. 风险提示\n"
                "3. 还款规划建议\n"
                "4. 客户沟通话术\n"
                "要求：表达清晰、克制、合规，避免制造恐慌。"
            ),
            "逾期催收": (
                "你是一名合规催收专员，需要为逾期客户制定合规催收策略。\n"
                "请输出以下内容：\n"
                "1. 合规催收话术\n"
                "2. 可执行还款方案\n"
                "3. 逾期后果说明\n"
                "4. 升级处理建议\n"
                "要求：强调合规、事实清晰、尊重客户，不得出现过激表达。"
            ),
            "协商还款": (
                "你是一名坏账处置专员，需要为高风险客户制定协商还款方案。\n"
                "请输出以下内容：\n"
                "1. 协商方案\n"
                "2. 分期建议\n"
                "3. 沟通话术\n"
                "4. 风险处置建议\n"
                "要求：以解决问题为目标，兼顾风险控制和客户沟通体验。"
            ),
        }

        if scenario not in scenario_templates:
            raise ValueError(
                f"Unsupported scenario '{scenario}'. "
                f"Supported scenarios: {sorted(scenario_templates)}"
            )

        return (
            f"{scenario_templates[scenario]}\n\n"
            f"{self.COMPLIANCE_RED_LINES}\n\n"
            "请基于以下用户信息生成结构化输出。\n"
            "输出要求：\n"
            "- 使用简体中文\n"
            "- 分点输出\n"
            "- 结尾补充“合规提醒”小节\n"
            "- 所有建议必须可执行、可落地、可审计\n\n"
            f"用户信息：\n{user_profile}"
        )

    def generate_strategy(self, user_info: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically map the scenario and generate strategy content."""

        scenario = self._match_scenario(user_info)
        prompt = self._build_prompt(user_info, scenario)
        generated_content = self.client.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1200,
        )
        compliant_content = self.compliance_check(generated_content)

        return {
            "scenario": scenario,
            "prompt": prompt,
            "generated_content": generated_content,
            "final_content": compliant_content,
        }

    def compliance_check(self, generated_content: str) -> str:
        """Run a second-pass compliance check and auto-revise if needed."""

        if not generated_content.strip():
            raise ValueError("Generated content must be a non-empty string.")

        compliance_prompt = (
            "你是一名金融内容合规审核专家。请审核以下策略内容是否存在不合规表达，"
            "重点检查：暴力催收、威胁辱骂、虚假承诺、误导性表述、未经审批的减免承诺。\n\n"
            "如果内容完全合规，请返回如下 JSON：\n"
            '{"compliant": true, "issues": [], "revised_content": "<原文>"}\n\n'
            "如果内容不合规，请返回如下 JSON：\n"
            '{"compliant": false, "issues": ["问题1", "问题2"], "revised_content": "<修正后的合规内容>"}\n\n'
            "只返回 JSON，不要额外解释。\n\n"
            f"待审核内容：\n{generated_content}"
        )

        try:
            review_content = self.client.generate(
                prompt=compliance_prompt,
                temperature=0.1,
                max_tokens=1200,
            )
            review_payload = self._parse_json_response(review_content)
            revised_content = review_payload.get("revised_content", generated_content)
            if not revised_content or not str(revised_content).strip():
                return generated_content
            return str(revised_content).strip()
        except Exception:
            # Keep the original content if the compliance check request fails.
            return generated_content

    def _match_scenario(self, user_info: Dict[str, Any]) -> str:
        """Map user risk information to one of the four strategy scenarios."""

        if "postloan_scene_label_name" in user_info:
            scene_name = str(user_info["postloan_scene_label_name"]).strip()
            mapping = {
                "正常维护": "正常维护",
                "风险预警": "风险预警",
                "逾期催收": "逾期催收",
                "协商还款": "协商还款",
            }
            if scene_name in mapping:
                return mapping[scene_name]

        if "postloan_scene_label" in user_info:
            scene_label = int(user_info["postloan_scene_label"])
            mapping = {
                0: "正常维护",
                1: "风险预警",
                2: "逾期催收",
                3: "协商还款",
            }
            if scene_label in mapping:
                return mapping[scene_label]

        risk_label = int(user_info.get("preloan_risk_label", user_info.get("risk_label", 0)))
        risk_mapping = {
            0: "正常维护",
            1: "风险预警",
            2: "逾期催收",
            3: "协商还款",
            4: "协商还款",
        }
        return risk_mapping.get(risk_label, "风险预警")

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse a JSON payload from a model response."""

        cleaned_content = content.strip()
        if cleaned_content.startswith("```"):
            cleaned_content = cleaned_content.strip("`")
            if cleaned_content.startswith("json"):
                cleaned_content = cleaned_content[4:].strip()

        start_index = cleaned_content.find("{")
        end_index = cleaned_content.rfind("}")
        if start_index == -1 or end_index == -1 or end_index <= start_index:
            raise ValueError("No JSON object could be extracted from compliance output.")

        return json.loads(cleaned_content[start_index : end_index + 1])
