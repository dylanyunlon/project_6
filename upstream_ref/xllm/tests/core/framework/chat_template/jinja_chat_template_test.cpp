/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "jinja_chat_template.h"

#include <gtest/gtest.h>

namespace xllm {

class TestableJinjaChatTemplate : public JinjaChatTemplate {
 public:
  TestableJinjaChatTemplate(const TokenizerArgs& args)
      : JinjaChatTemplate(args) {}

  using JinjaChatTemplate::apply;
};

TEST(JinjaChatTemplate, OpenChatModel) {
  // clang-format off
  const std::string template_str =
      "<s>"
      "{% for message in messages %}"
        "{{ 'GPT4 Correct ' + message['role'] + ': ' + message['content'] + '<|end_of_turn|>'}}"
      "{% endfor %}"
      "{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}";

  nlohmann::ordered_json messages = {
      {{"role", "system"}, {"content", "you are a helpful assistant."}},
      {{"role", "user"}, {"content", "hi"}},
      {{"role", "assistant"}, {"content", "what i can do for you?"}},
      {{"role", "user"}, {"content", "how are you?"}}};
  const std::string expected =
    "<s>"
    "GPT4 Correct system: you are a helpful assistant.<|end_of_turn|>"
    "GPT4 Correct user: hi<|end_of_turn|>"
    "GPT4 Correct assistant: what i can do for you?<|end_of_turn|>"
    "GPT4 Correct user: how are you?<|end_of_turn|>"
    "GPT4 Correct Assistant:";
  // clang-format on

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("<|end_of_turn|>");
  TestableJinjaChatTemplate template_(args);
  auto result = template_.apply(messages);
  ASSERT_TRUE(result.has_value());

  EXPECT_EQ(result.value(), expected);
}

TEST(JinjaChatTemplate, AppliesChatTemplateKwargs) {
  const std::string template_str =
      "{% if enable_thinking %}<think>{% endif %}"
      "{% for message in messages %}"
      "{{ message['role'] + ': ' + message['content'] }}"
      "{% endfor %}"
      "{% if not enable_thinking %}<no_think>{% endif %}";

  nlohmann::ordered_json messages = {
      {{"role", "user"}, {"content", "describe this image"}}};
  nlohmann::ordered_json chat_template_kwargs = {{"enable_thinking", false}};

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  TestableJinjaChatTemplate template_(args);
  const nlohmann::ordered_json tools = nlohmann::json::array();
  auto result = template_.apply(messages, tools, chat_template_kwargs);
  ASSERT_TRUE(result.has_value());

  EXPECT_EQ(result.value(), "user: describe this image<no_think>");
}

// TC-04 / TC-05: Qwen3.8-27B uses "is undefined" / "is not undefined" in its
// chat template.  Minja doesn't support that test, so
// normalize_minja_tests() rewrites it to "is none" / "is not none" before
// parsing.  Verify both the default path (kwarg absent → treated as
// none → default branch) and the configured path (kwarg present).
TEST(JinjaChatTemplate, SupportsUndefinedTests) {
  // Simulates Qwen3.8-27B's pattern:
  //   {% if enable_thinking is undefined %}default{% else %}configured{% endif %}
  // After normalization this becomes:
  //   {% if enable_thinking is none %}default{% else %}configured{% endif %}
  const std::string template_str =
      "{% if enable_thinking is undefined %}"
      "default_thinking"
      "{% else %}"
      "configured_thinking"
      "{% endif %}"
      "{% for message in messages %}"
      "{{ message['role'] + ': ' + message['content'] }}"
      "{% endfor %}";

  nlohmann::ordered_json messages = {
      {{"role", "user"}, {"content", "hello"}}};

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");

  // Construction should succeed (normalize rewrites "is undefined" → "is none")
  TestableJinjaChatTemplate template_(args);

  // Case 1: No kwargs → enable_thinking is absent → treated as none → default
  {
    const nlohmann::ordered_json tools = nlohmann::json::array();
    const nlohmann::ordered_json kwargs = nlohmann::json::object();
    auto result = template_.apply(messages, tools, kwargs);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), "default_thinkinguser: hello");
  }

  // Case 2: kwargs with enable_thinking=false → not none → configured
  {
    const nlohmann::ordered_json tools = nlohmann::json::array();
    const nlohmann::ordered_json kwargs = {{"enable_thinking", false}};
    auto result = template_.apply(messages, tools, kwargs);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), "configured_thinkinguser: hello");
  }
}

// TC-01 / TC-03: Verify "is not undefined" is also rewritten, and that
// plain text outside Jinja blocks is preserved.
TEST(JinjaChatTemplate, IsNotUndefinedRewritten) {
  // "is not undefined" should become "is not none"
  const std::string template_str =
      "plain text: value is undefined "
      "{% if enable_thinking is not undefined %}"
      "has_value"
      "{% else %}"
      "missing"
      "{% endif %}"
      "{% for message in messages %}"
      "{{ message['role'] + ': ' + message['content'] }}"
      "{% endfor %}";

  nlohmann::ordered_json messages = {
      {{"role", "user"}, {"content", "hi"}}};

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  TestableJinjaChatTemplate template_(args);

  // No kwargs → enable_thinking absent → "is not none" is false → "missing"
  // Plain text before the block is preserved verbatim (including "is undefined")
  {
    const nlohmann::ordered_json tools = nlohmann::json::array();
    const nlohmann::ordered_json kwargs = nlohmann::json::object();
    auto result = template_.apply(messages, tools, kwargs);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(),
              "plain text: value is undefined missinguser: hi");
  }

  // With enable_thinking=true → "is not none" is true → "has_value"
  {
    const nlohmann::ordered_json tools = nlohmann::json::array();
    const nlohmann::ordered_json kwargs = {{"enable_thinking", true}};
    auto result = template_.apply(messages, tools, kwargs);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(),
              "plain text: value is undefined has_valueuser: hi");
  }
}

// TC-02: Quoted strings inside Jinja blocks are not rewritten.
TEST(JinjaChatTemplate, QuotedStringsPreserved) {
  // The string literal 'value is undefined' should NOT be rewritten.
  const std::string template_str =
      "{% for message in messages %}"
      "{{ 'value is undefined' }}"
      "{% endfor %}";

  nlohmann::ordered_json messages = {
      {{"role", "user"}, {"content", "test"}}};

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  TestableJinjaChatTemplate template_(args);

  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = nlohmann::json::object();
  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());
  // The literal string is rendered verbatim — "is undefined" not rewritten.
  EXPECT_EQ(result.value(), "value is undefined");
}

}  // namespace xllm
