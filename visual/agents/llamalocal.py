"""LocalAgent — on-device VLM agent using llama.cpp (GGUF models, Windows/Linux/macOS)."""

import base64
import io
import logging
import os
import re
import sys
import time
import uuid
import ctypes
import json
from typing import Any, Dict, List, Optional, Tuple

from jinja2.sandbox import ImmutableSandboxedEnvironment
from llama_cpp import Llama, llama, llama_types, llama_grammar, llama_cpp
from llama_cpp.llama_chat_format import (
    Llava15ChatHandler,
    _get_system_message,
    _convert_completion_to_chat,
    _grammar_for_response_format,
)

from PIL import Image

from visual.agents.base import BaseAgent
from visual.config.visual_config import AUTOMATION_CONFIG


logger = logging.getLogger("mano.local")

LOCAL_AGENT_CONFIG = {
    "MAX_NEW_TOKENS": 2048,
    "TEMPERATURE": 0.0,
    "TOP_P": 1.0,
    "SCREENSHOT_WIDTH": 1280,
    "HISTORY_IMAGE_COUNT": 1,
    # llama.cpp specific
    "N_CTX": 8192,          # context size – multi-modal needs headroom for vision tokens
    "N_GPU_LAYERS": -1,      # 0 = CPU only; set to -1 for all layers on GPU (requires CUDA/Vulkan build)
    "N_BATCH": 512,
    "VERBOSE": False,
    # multimodal
    "MMPROJ_PATH": "",       # path to mmproj GGUF (vision projector), e.g. "mano-p-mmproj-f16.gguf"
    "CHAT_FORMAT": "",           # chat format (e.g. "qwen3-vl"); requires llama-cpp-python >= 0.3.8
    "clip_model_name": "Mano-CUA-2.0-4B-mmproj-F16.gguf",
    "model_name": "Mano-CUA-2.0-4B-F16.gguf",
}

template_str = """{%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, do_vision_count) %}
    {%- if content is string %}
        {{- content }}
    {%- else %}
        {%- for item in content %}
            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                {%- if do_vision_count %}
                    {%- set image_count.value = image_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}Picture {{ image_count.value }}: {% endif -%}
                <|vision_start|><|image_pad|><|vision_end|>
            {%- elif 'video' in item or item.type == 'video' %}
                {%- if do_vision_count %}
                    {%- set video_count.value = video_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}Video {{ video_count.value }}: {% endif -%}
                <|vision_start|><|video_pad|><|vision_end|>
            {%- elif 'text' in item %}
                {{- item.text }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
{%- endmacro %}
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- render_content(messages[0].content, false) + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + render_content(messages[0].content, false) + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" %}
        {%- set content = render_content(message.content, false) %}
        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- set content = render_content(message.content, True) %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n<think>\n' }}
{%- endif %}
"""

class LocalAgent(BaseAgent):
    """On-device VLM agent using llama.cpp (Qwen2-VL / LLaVA / any GGUF multimodal model)."""

    agent_type = "local"

    SYSTEM_PROMPT = "You are a helpful assistant."

    INSTRUCTION_TEMPLATE = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
<think>思考过程</think>
<action_desp>动作描述</action_desp>
<action>具体动作</action>

## Action Space

open_app(app_name='') # Open an application by name.
open_url(url='') # Open a URL in the browser.
hover(start_box='<|box_start|>(x1,y1)<|box_end|>')
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
triple_click(start_box='<|box_start|>(x1,y1)<|box_end|>') left click at the coordinate (x1,y1) three times
hotkey_click(start_box='<|box_start|>(x1,y1)<|box_end|>', key=''). press command key and click at the coordinate (x1,y1)
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>').  right click at the coordinate (x1,y1)
type(content='') type the content.
doubleclick(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>') # Drag an element from the start coordinate (x1,y1) to the end coordinate (x3,y3).
hotkey(key='') # Trigger a keyboard shortcut.
wait(duration='') # Sleep for specified duration (in seconds) and take a screenshot to check for any changes.
call_user() # Request human assistance
stop(reason='') # If the item can not found in the image, give the reason
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left', amount='scroll_amount') # Scroll on the specified direction at the coordinate (x1,y1) by the given amount
finish() # The task is completed.

## Note
- Use Chinese in `<think>` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `<action_desp>` part.

## User Instruction:
{instruction}

"""

    def __init__(self, model_path: str):
        self.model_load_path = model_path
        self._model_path = os.path.expanduser(os.path.join(model_path, LOCAL_AGENT_CONFIG.get("model_name")))
        self.model_name = os.path.basename(self._model_path)
        self._model_loaded = False
        self.prompt_history = []
        self.cfg = dict(LOCAL_AGENT_CONFIG)  # shallow copy so per-instance overrides don't leak

        self.model = None       # llama_cpp.Llama instance
        self._model_loaded = False

        self.prompt_history: list = []
        self.step_count = 0

    def _ensure_model_loaded(self):
        """Lazy-load GGUF model on first predict."""
        if self._model_loaded:
            return

        from llama_cpp import Llama

        logger.info(f"Loading GGUF model from {self._model_path} ...")

        llama_kwargs = {
            "model_path": self._model_path,
            "n_ctx": self.cfg["N_CTX"],
            "n_gpu_layers": self.cfg["N_GPU_LAYERS"],
            "n_batch": self.cfg["N_BATCH"],
            "verbose": self.cfg["VERBOSE"],
            "chat_handler" :CustomQwenVLHandler(clip_model_path=os.path.join(self.model_load_path, LOCAL_AGENT_CONFIG.get("clip_model_name")),verbose=False)
        }

        chat_format = self.cfg.get("CHAT_FORMAT", template_str)
        if chat_format:
            llama_kwargs["chat_format"] = chat_format
        self.model = Llama(**llama_kwargs)
        self._model_loaded = True
        logger.info("GGUF model loaded successfully.")

    # ─── BaseAgent interface ──────────────────────────────────

    def predict(
        self,
        task_instruction: str,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[Dict[str, Any]], str, str]:
        self._ensure_model_loaded()
        _t0 = time.time()

        # 1. Extract screenshot
        screenshot_b64 = self._extract_screenshot(tool_results)
        if screenshot_b64 is None:
            screenshot_b64 = self._take_screenshot_b64()

        # 2. Build prompt
        user_text, images = self._build_prompt(task_instruction, screenshot_b64)

        # 3. Run inference
        response_text = self._infer(user_text, images)
        print(f"  [model output] {response_text}")

        # Save raw response to file
        self._save_raw_response(response_text)
        self.last_raw_response = response_text

        # 4. Parse response
        parsed = self._parse_response(response_text)
        think = parsed["think"]
        action_desp = parsed["action_desp"]
        parsed_actions = parsed["actions"]

        # 5. Record prompt history
        if screenshot_b64:
            self.prompt_history.append({
                "desc": action_desp or str(parsed_actions),
                "screenshot_b64": screenshot_b64,
            })

        # 6. Convert to Claude-compatible actions and determine status
        if not parsed_actions:
            actions = [{"action_type": "FAIL"}]
            status = "FAIL"
            action_str = "FAIL"
        else:
            actions = []
            for a in parsed_actions:
                actions.extend(self._convert_action(a))
            status = self._determine_status(actions)
            action_str = " → ".join(self._format_action_desc([a]) for a in actions)

        self.step_count += 1
        elapsed = time.time() - _t0
        print(f"  [step {self.step_count}] {elapsed:.1f}s — {action_str}")

        return think, actions, status, action_str

    def close(self, skip_eval: bool = False, close_reason: Optional[str] = None) -> Optional[dict]:
        if self.model is not None:
            self.model.close()
            self.model = None
            self._model_loaded = False
        return None

    def _save_raw_response(self, text: str):
        import json
        log_path = os.path.expanduser("~/.mano/raw_responses.jsonl")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": self.step_count, "raw": text}, ensure_ascii=False) + "\n")

    def agree_to_continue(self) -> None:
        self.prompt_history.append({
            "desc": "用户已确认继续",
            "screenshot_b64": "",
        })

    # ─── Screenshot handling ──────────────────────────────────

    def _take_screenshot_b64(self) -> str:
        from visual.computer.computer_use_util import screenshot_to_bytes, b64_png
        raw_bytes = screenshot_to_bytes()
        raw_b64 = b64_png(raw_bytes)
        return self._resize_screenshot_b64(raw_b64)

    def _extract_screenshot(self, tool_results: Optional[List[Dict[str, Any]]]) -> Optional[str]:
        if not tool_results:
            return None
        for tr in reversed(tool_results):
            b64 = tr.get("screenshot_b64")
            if b64:
                return self._resize_screenshot_b64(b64)
        return None

    def _resize_screenshot_b64(self, b64: str) -> str:
        target_w = self.cfg["SCREENSHOT_WIDTH"]
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes))
        if img.width == target_w:
            return b64
        ratio = target_w / img.width
        new_h = int(img.height * ratio)
        img = img.resize((target_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ─── Prompt building ──────────────────────────────────────

    def _build_prompt(self, task: str, current_screenshot_b64: Optional[str]) -> Tuple[str, list]:
        import platform as _platform
        images: list = []
        history_count = self.cfg["HISTORY_IMAGE_COUNT"]
        recent = self.prompt_history[-history_count:] if history_count else []

        history_parts = []
        for i, h in enumerate(self.prompt_history):
            step_num = i + 1
            desc = h["desc"]
            if h in recent and h.get("screenshot_b64"):
                images.append(h["screenshot_b64"])
                history_parts.append(f"第{step_num}步：{desc}，对应截图为<image>")
            else:
                history_parts.append(f"第{step_num}步：{desc}")

        instruction_parts = [task]
        if history_parts:
            instruction_parts.append("")
            instruction_parts.extend(history_parts)
        if current_screenshot_b64:
            images.append(current_screenshot_b64)
            instruction_parts.append("当前步骤的截图为<image>")

        text = self.INSTRUCTION_TEMPLATE.format(
            platform=_platform.system(),
            instruction="\n".join(instruction_parts),
        )
        return text, images

    # ─── Inference (llama.cpp) ────────────────────────────────

    def _infer(self, user_text: str, images: list) -> str:
        """
        Run multimodal inference via llama.cpp chat-completion API.

        ``images`` is a list of base64-encoded PNG strings (already resized).
        ``user_text`` contains ``<image>`` placeholders; matching is done from
        back to front (mirroring local.py) to ensure correct image-to-slot
        mapping in edge cases where placeholder count differs from image count.
        """
        org_placeholder = "<image>"

        # ---- find <image> slots from back to front (consistent with local.py) ----
        pi = len(images)
        img_slots = []  # (position_in_text, image_index)
        while pi > 0:
            pi -= 1
            pos = user_text.rfind(org_placeholder)
            if pos >= 0:
                img_slots.append((pos, pi))
            else:
                break

        # Sort by position ascending to build content left-to-right
        img_slots.sort(key=lambda x: x[0])

        # ---- diagnostic: image-to-placeholder mapping ----
        placeholder_count = user_text.count(org_placeholder)
        print(f"  [diagnose] <image> placeholders in prompt: {placeholder_count}, "
              f"images provided: {len(images)}, "
              f"matched slots: {len(img_slots)}")
        if placeholder_count != len(images):
            print(f"  [diagnose] MISMATCH: placeholders({placeholder_count}) != images({len(images)})")
        if img_slots:
            slot_info = ", ".join(f"text_pos={p}→img[{i}]" for p, i in img_slots)
            print(f"  [diagnose] mapping: {slot_info}")

        # ---- build alternating text / image_url user_content ----
        user_content: list = []
        last = 0
        for pos, idx in img_slots:
            if pos > last:
                t = user_text[last:pos]
                if t:
                    user_content.append({"type": "text", "text": t})
            if idx < len(images):
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{images[idx]}"
                    },
                })
            last = pos + len(org_placeholder)

        # Remaining text after the last <image>
        if last < len(user_text):
            t = user_text[last:]
            if t:
                user_content.append({"type": "text", "text": t})

        # ---- build messages ----
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # ---- call llama.cpp ----
        _t0 = time.time()
        result = self.model.create_chat_completion(
            messages=messages,
            max_tokens=self.cfg["MAX_NEW_TOKENS"],
            temperature=self.cfg["TEMPERATURE"],
            top_p=self.cfg["TOP_P"],
        )

        response_text = result["choices"][0]["message"]["content"]

        # ---- logging ----
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        gen_tokens = usage.get("completion_tokens", 0)
        elapsed = time.time() - _t0
        tps = gen_tokens / elapsed if elapsed > 0 else 0
        print(f"  [decode] prompt={prompt_tokens}, gen={gen_tokens} tokens, {tps:.1f} tok/s")

        # ---- diagnostic: vision token estimation ----
        # Estimate text-only token count (rough: 1 token ~ 4 chars for Chinese+English)
        text_only = user_text.replace("<image>", "").strip()
        estimated_text_tokens = max(1, len(text_only) // 3)
        vision_tokens = prompt_tokens - estimated_text_tokens
        if len(images) > 0 and vision_tokens > 0:
            avg_vision_per_image = vision_tokens / len(images)
            print(f"  [diagnose] vision_tokens≈{vision_tokens}, images={len(images)}, "
                  f"avg_vision_per_image≈{avg_vision_per_image:.1f}, "
                  f"estimated_text_tokens≈{estimated_text_tokens}")
        elif len(images) > 0:
            print(f"  [diagnose] prompt_tokens({prompt_tokens}) <= estimated_text_tokens({estimated_text_tokens}), "
                  f"images might not be ingested correctly")

        return response_text

    # ─── Response parsing ─────────────────────────────────────

    def _parse_response(self, text: str) -> dict:
        think = self._extract_tag(text, "think") or ""
        action_desp = self._extract_tag(text, "action_desp") or ""
        action_raw = self._extract_tag(text, "action") or ""
        actions = []
        if action_raw:
            for m in re.finditer(r"(\w+\(.*?\))(?=\s*\n\s*\w+\(|\s*$)", action_raw.strip(), re.DOTALL):
                parsed = self._parse_action(m.group(1).strip())
                if parsed:
                    actions.append(parsed)
        return {"think": think.strip(), "action_desp": action_desp.strip(), "actions": actions}

    def _extract_tag(self, text: str, tag: str) -> Optional[str]:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        if m:
            return m.group(1)
        if tag == "think":
            m = re.search(r"^(.*?)</think>", text, re.DOTALL)
            return m.group(1) if m else None
        return None

    def _parse_box(self, box_str: str) -> list:
        m = re.search(r"\((\d+)\s*,\s*(\d+)\)", box_str)
        if not m:
            return [0, 0]
        return [int(m.group(1)), int(m.group(2))]

    def _parse_action(self, action_str: str) -> Optional[dict]:
        action_str = action_str.strip()
        m = re.match(r"(\w+)\((.*)\)$", action_str, re.DOTALL)
        if not m:
            return None

        func_name = m.group(1)
        args_str = m.group(2).strip()

        kwargs = {}
        for km in re.finditer(r"(\w+)\s*=\s*'(.*?)'", args_str, re.DOTALL):
            kwargs[km.group(1)] = km.group(2)

        if func_name in ("click", "doubleclick", "hover"):
            return {"action": func_name, "coords": self._parse_box(kwargs.get("start_box", ""))}
        if func_name == "triple_click":
            return {"action": "triple_click", "coords": self._parse_box(kwargs.get("start_box", ""))}
        if func_name == "right_single":
            return {"action": "right_click", "coords": self._parse_box(kwargs.get("start_box", ""))}
        if func_name == "hotkey_click":
            return {"action": "hotkey_click", "coords": self._parse_box(kwargs.get("start_box", "")), "key": kwargs.get("key", "")}
        if func_name == "type":
            return {"action": "type", "text": kwargs.get("content", "")}
        if func_name == "hotkey":
            return {"action": "hotkey", "key": kwargs.get("key", "")}
        if func_name == "scroll":
            amount = kwargs.get("amount", "5")
            try:
                amount = int(amount)
            except (ValueError, TypeError):
                amount = 5
            result = {"action": "scroll", "direction": kwargs.get("direction", "down"), "amount": amount}
            box = kwargs.get("start_box", "")
            if box:
                result["coords"] = self._parse_box(box)
            return result
        if func_name == "drag":
            return {
                "action": "drag",
                "start": self._parse_box(kwargs.get("start_box", "")),
                "end": self._parse_box(kwargs.get("end_box", "")),
            }
        if func_name == "wait":
            duration = kwargs.get("duration", "5")
            try:
                duration = float(duration)
            except (ValueError, TypeError):
                duration = 5.0
            return {"action": "wait", "duration": duration}
        if func_name == "finish":
            return {"action": "finish"}
        if func_name == "open_app":
            return {"action": "open_app", "app_name": kwargs.get("app_name", "")}
        if func_name == "open_url":
            return {"action": "open_url", "url": kwargs.get("url", "")}
        if func_name == "stop":
            return {"action": "stop", "reason": kwargs.get("reason", "")}
        if func_name == "call_user":
            return {"action": "call_user"}
        return None

    # ─── Action conversion: VLM → Claude format ───────────────

    def _norm_coord(self, x: int, y: int) -> list:
        return [int(x / 1000 * AUTOMATION_CONFIG["SCREEN_SCALE_WIDTH"]),
                int(y / 1000 * AUTOMATION_CONFIG["SCREEN_SCALE_HEIGHT"])]

    def _make_tool_action(self, input_dict: dict) -> dict:
        return {
            "name": "computer",
            "input": input_dict,
            "id": str(uuid.uuid4()),
            "action_type": "tool_use",
        }

    def _determine_status(self, actions: List[Dict[str, Any]]) -> str:
        for a in actions:
            at = (a.get("action_type") or "").upper()
            if at == "DONE":
                return "DONE"
            if at == "FAIL":
                return "FAIL"
            if at == "CALL_USER":
                return "CALL_USER"
        return "RUNNING"

    def _format_action_desc(self, actions: List[Dict[str, Any]]) -> str:
        if not actions:
            return ""
        a = actions[0]
        at = (a.get("action_type") or "").upper()
        if at in ("DONE", "FAIL", "CALL_USER"):
            return at
        inp = a.get("input", {})
        name = a.get("name", "")
        if name == "open_app":
            return f"open_app(\"{inp.get('app_name', '')}\")"
        if name == "open_url":
            return f"open_url(\"{inp.get('url', '')}\")"
        action = inp.get("action", "unknown")
        coord = inp.get("coordinate")
        if coord:
            return f"{action}({coord[0]}, {coord[1]})"
        text = inp.get("text")
        if text:
            return f"{action}(\"{text[:30]}\")"
        direction = inp.get("scroll_direction")
        if direction:
            return f"{action} {direction}"
        return action

    def _convert_action(self, action: dict) -> List[Dict[str, Any]]:
        act = action["action"]

        if act == "finish":
            return [{"action_type": "DONE"}]
        if act == "open_app":
            return [{
                "name": "open_app",
                "input": {"app_name": action.get("app_name", "")},
                "id": str(uuid.uuid4()),
                "action_type": "tool_use",
            }]
        if act == "open_url":
            return [{
                "name": "open_url",
                "input": {"url": action.get("url", "")},
                "id": str(uuid.uuid4()),
                "action_type": "tool_use",
            }]
        if act == "stop":
            return [{"action_type": "FAIL"}]
        if act == "call_user":
            return [{"action_type": "CALL_USER"}]

        if act == "click":
            coords = action.get("coords", [0, 0])
            return [self._make_tool_action({
                "action": "left_click",
                "coordinate": self._norm_coord(coords[0], coords[1]),
            })]

        if act == "doubleclick":
            coords = action.get("coords", [0, 0])
            return [self._make_tool_action({
                "action": "double_click",
                "coordinate": self._norm_coord(coords[0], coords[1]),
            })]

        if act == "triple_click":
            coords = action.get("coords", [0, 0])
            return [self._make_tool_action({
                "action": "triple_click",
                "coordinate": self._norm_coord(coords[0], coords[1]),
            })]

        if act == "right_click":
            coords = action.get("coords", [0, 0])
            return [self._make_tool_action({
                "action": "right_click",
                "coordinate": self._norm_coord(coords[0], coords[1]),
            })]

        if act == "hover":
            coords = action.get("coords", [0, 0])
            return [self._make_tool_action({
                "action": "mouse_move",
                "coordinate": self._norm_coord(coords[0], coords[1]),
            })]

        if act == "hotkey_click":
            coords = action.get("coords", [0, 0])
            return [self._make_tool_action({
                "action": "left_click",
                "coordinate": self._norm_coord(coords[0], coords[1]),
                "text": action.get("key", ""),
            })]

        if act == "type":
            return [self._make_tool_action({
                "action": "type",
                "text": action.get("text", ""),
            })]

        if act == "hotkey":
            return [self._make_tool_action({
                "action": "key",
                "text": action.get("key", ""),
            })]

        if act == "scroll":
            direction = action.get("direction", "down")
            amount = action.get("amount", 3)
            coords = action.get("coords")
            coordinate = self._norm_coord(coords[0], coords[1]) if coords else [640, 360]
            return [self._make_tool_action({
                "action": "scroll",
                "scroll_direction": direction,
                "coordinate": coordinate,
                "scroll_amount": amount,
            })]

        if act == "drag":
            start = action.get("start", [0, 0])
            end = action.get("end", [0, 0])
            return [self._make_tool_action({
                "action": "left_click_drag",
                "start_coordinate": self._norm_coord(start[0], start[1]),
                "coordinate": self._norm_coord(end[0], end[1]),
            })]

        if act == "wait":
            duration = action.get("duration", 5)
            return [self._make_tool_action({
                "action": "wait",
                "duration": duration,
            })]

        return [{"action_type": "FAIL"}]

class CustomQwenVLHandler(Llava15ChatHandler):
    """读取 GGUF 内嵌 chat_template，输出中把 <|image_pad|> 替换为 C 层 media_marker"""

    def __init__(self,  clip_model_path: str, verbose: bool = True):
        # 初始化父类（加载 mmproj / mtmd 上下文）
        # Step 3：用 GGUF 内嵌模板渲染 prompt
        template_env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True,
        )
        self.template = template_env.from_string(open(r"D:\workdir2\mano-skill\model\chat_template.jinja", "r").read())
        super().__init__(clip_model_path=clip_model_path, verbose=verbose)

    def __call__(self, **kwargs):
        llama_inst = kwargs["llama"]

        # Step 0：初始化 mtmd 上下文
        self._init_mtmd_context(llama_inst)
        assert self.mtmd_ctx is not None

        # Step 1：提取参数
        messages = kwargs.get("messages", [])
        functions = kwargs.get("functions")
        function_call = kwargs.get("function_call")
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        temperature = kwargs.get("temperature", 0.2)
        top_p = kwargs.get("top_p", 0.95)
        top_k = kwargs.get("top_k", 40)
        min_p = kwargs.get("min_p", 0.05)
        typical_p = kwargs.get("typical_p", 1.0)
        stream = kwargs.get("stream", False)
        stop = kwargs.get("stop", [])
        seed = kwargs.get("seed")
        response_format = kwargs.get("response_format")
        max_tokens = kwargs.get("max_tokens")
        presence_penalty = kwargs.get("presence_penalty", 0.0)
        frequency_penalty = kwargs.get("frequency_penalty", 0.0)
        repeat_penalty = kwargs.get("repeat_penalty", 1.1)
        tfs_z = kwargs.get("tfs_z", 1.0)
        mirostat_mode = kwargs.get("mirostat_mode", 0)
        mirostat_tau = kwargs.get("mirostat_tau", 5.0)
        mirostat_eta = kwargs.get("mirostat_eta", 0.1)
        model = kwargs.get("model")
        logits_processor = kwargs.get("logits_processor")
        grammar = kwargs.get("grammar")
        logit_bias = kwargs.get("logit_bias")
        logprobs = kwargs.get("logprobs")
        top_logprobs = kwargs.get("top_logprobs")

        # Step 2：提取图片 URL
        image_urls = self.get_image_urls(messages)


        text = self.template.render(
            messages=messages,
            add_generation_prompt=True,
            add_vision_id=True,
            tools=tools,
            eos_token=llama_inst.detokenize([llama_inst.token_eos()]),
            bos_token=llama_inst.detokenize([llama_inst.token_bos()]),
        )

        # Step 4：将 <|image_pad|> 替换为 C 层 media_marker
        media_marker = self._mtmd_cpp.mtmd_default_marker().decode("utf-8")
        text = text.replace("<|image_pad|>", media_marker)

        if self.verbose:
            print(text, file=sys.stderr)

        # Step 5：图片 → bitmap → tokenize → eval（以下均复制自父类 __call__）
        bitmaps = []
        bitmap_cleanup = []
        try:
            for image_url in image_urls:
                image_bytes = self.load_image(image_url)
                bitmap = self._create_bitmap_from_bytes(image_bytes)
                bitmaps.append(bitmap)
                bitmap_cleanup.append(bitmap)

            input_text = self._mtmd_cpp.mtmd_input_text()
            input_text.text = text.encode("utf-8")
            input_text.add_special = True
            input_text.parse_special = True

            chunks = self._mtmd_cpp.mtmd_input_chunks_init()
            if chunks is None:
                raise ValueError("Failed to create input chunks")

            try:
                bitmap_array = (self._mtmd_cpp.mtmd_bitmap_p_ctypes * len(bitmaps))(
                    *bitmaps
                )
                result = self._mtmd_cpp.mtmd_tokenize(
                    self.mtmd_ctx,
                    chunks,
                    ctypes.byref(input_text),
                    bitmap_array,
                    len(bitmaps),
                )
                if result != 0:
                    raise ValueError(f"Failed to tokenize input: error code {result}")

                llama_inst.reset()
                llama_inst._ctx.kv_cache_clear()

                n_chunks = self._mtmd_cpp.mtmd_input_chunks_size(chunks)
                for i in range(n_chunks):
                    chunk = self._mtmd_cpp.mtmd_input_chunks_get(chunks, i)
                    if chunk is None:
                        continue

                    chunk_type = self._mtmd_cpp.mtmd_input_chunk_get_type(chunk)

                    if chunk_type == self._mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_TEXT:
                        n_tokens_out = ctypes.c_size_t()
                        tokens_ptr = (
                            self._mtmd_cpp.mtmd_input_chunk_get_tokens_text(
                                chunk, ctypes.byref(n_tokens_out)
                            )
                        )
                        if tokens_ptr and n_tokens_out.value > 0:
                            tokens = [
                                tokens_ptr[j] for j in range(n_tokens_out.value)
                            ]
                            if (
                                llama_inst.n_tokens + len(tokens)
                                > llama_inst.n_ctx()
                            ):
                                raise ValueError(
                                    f"Prompt exceeds n_ctx: "
                                    f"{llama_inst.n_tokens + len(tokens)} > {llama_inst.n_ctx()}"
                                )
                            llama_inst.eval(tokens)

                    elif chunk_type in [
                        self._mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_IMAGE,
                        self._mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_AUDIO,
                    ]:
                        chunk_n_tokens = (
                            self._mtmd_cpp.mtmd_input_chunk_get_n_tokens(chunk)
                        )
                        if (
                            llama_inst.n_tokens + chunk_n_tokens
                            > llama_inst.n_ctx()
                        ):
                            raise ValueError(
                                f"Prompt exceeds n_ctx: "
                                f"{llama_inst.n_tokens + chunk_n_tokens} > {llama_inst.n_ctx()}"
                            )
                        new_n_past = llama_cpp.llama_pos(0)
                        result_eval = self._mtmd_cpp.mtmd_helper_eval_chunk_single(
                            self.mtmd_ctx,
                            llama_inst._ctx.ctx,
                            chunk,
                            llama_cpp.llama_pos(llama_inst.n_tokens),
                            llama_cpp.llama_seq_id(0),
                            llama_inst.n_batch,
                            False,
                            ctypes.byref(new_n_past),
                        )
                        if result_eval != 0:
                            raise ValueError(
                                f"Failed to evaluate chunk: error code {result_eval}"
                            )
                        llama_inst.n_tokens = new_n_past.value

                prompt = llama_inst.input_ids[: llama_inst.n_tokens].tolist()

            finally:
                self._mtmd_cpp.mtmd_input_chunks_free(chunks)

        finally:
            for bitmap in bitmap_cleanup:
                self._mtmd_cpp.mtmd_bitmap_free(bitmap)

        # Step 6：response_format / tool grammar 处理（复制自父类）
        if response_format is not None and response_format["type"] == "json_object":
            grammar = _grammar_for_response_format(response_format)

        if functions is not None:
            tools = [
                {"type": "function", "function": function}
                for function in functions
            ]

        if function_call is not None:
            if isinstance(function_call, str) and function_call in ("none", "auto"):
                tool_choice = function_call
            if isinstance(function_call, dict) and "name" in function_call:
                tool_choice = {
                    "type": "function",
                    "function": {"name": function_call["name"]},
                }

        tool = None
        if (
            tool_choice is not None
            and isinstance(tool_choice, dict)
            and tools is not None
        ):
            name = tool_choice["function"]["name"]
            tool = next((t for t in tools if t["function"]["name"] == name), None)
            if tool is None:
                raise ValueError(f"Tool choice '{name}' not found in tools.")
            schema = tool["function"]["parameters"]
            try:
                grammar = llama_grammar.LlamaGrammar.from_json_schema(
                    json.dumps(schema), verbose=llama_inst.verbose
                )
            except Exception as e:
                if llama_inst.verbose:
                    print(str(e), file=sys.stderr)
                grammar = llama_grammar.LlamaGrammar.from_string(
                    llama_grammar.JSON_GBNF, verbose=llama_inst.verbose
                )

        # Step 7：推理
        completion_or_chunks = llama_inst.create_completion(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            logprobs=top_logprobs if logprobs else None,
            stream=stream,
            stop=stop,
            seed=seed,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logits_processor=logits_processor,
            grammar=grammar,
            logit_bias=logit_bias,
        )

        return _convert_completion_to_chat(completion_or_chunks, stream=stream)

