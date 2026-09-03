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

#include <glog/logging.h>
#include <unistd.h>

#include <optional>
#include <string>

namespace xllm {

namespace {

// ---------------------------------------------------------------------------
// Minja compatibility: rewrite "is undefined" / "is not undefined" tests.
//
// Qwen3.8-27B's official chat template uses Jinja2's "is undefined" test:
//   {% if enable_thinking is undefined %}...{% endif %}
// Minja (xLLM's Jinja engine) represents missing template arguments as null
// (None), not as truly undefined variables.  Minja supports "is none" but not
// "is undefined".  We rewrite the tests before handing the template to Minja.
//
// Commit: 9071dd22 · feat · PR #2246
// ---------------------------------------------------------------------------

/// Replace "is undefined" → "is none" and "is not undefined" → "is not none"
/// within a single Jinja expression/statement block.  Tracks quoted strings
/// so that literal text like "'value is undefined'" is left untouched.
std::string replace_undefined_tests(const std::string& block) {
  std::string result;
  result.reserve(block.size());

  char in_quote = 0;  // 0 = not in string, '\'' or '"' = current quote char
  size_t i = 0;

  while (i < block.size()) {
    // Track quote boundaries (skip escaped quotes).
    if (!in_quote && (block[i] == '\'' || block[i] == '"')) {
      in_quote = block[i];
      result += block[i];
      ++i;
      continue;
    }
    if (in_quote) {
      if (block[i] == '\\' && i + 1 < block.size()) {
        // Escaped character inside string — copy both.
        result += block[i];
        result += block[i + 1];
        i += 2;
        continue;
      }
      if (block[i] == in_quote) {
        in_quote = 0;
      }
      result += block[i];
      ++i;
      continue;
    }

    // Outside quotes — try to match "is not undefined" first (longer match).
    const std::string kIsNotUndefined = "is not undefined";
    const std::string kIsUndefined = "is undefined";

    if (block.compare(i, kIsNotUndefined.size(), kIsNotUndefined) == 0) {
      result += "is not none";
      i += kIsNotUndefined.size();
    } else if (block.compare(i, kIsUndefined.size(), kIsUndefined) == 0) {
      result += "is none";
      i += kIsUndefined.size();
    } else {
      result += block[i];
      ++i;
    }
  }

  return result;
}

/// Scan a Jinja template for {{ }} and {% %} blocks and apply
/// replace_undefined_tests() within each block.  Plain text outside
/// Jinja blocks is passed through unmodified.
std::string normalize_minja_tests(const std::string& tmpl) {
  std::string result;
  result.reserve(tmpl.size());

  size_t i = 0;
  while (i < tmpl.size()) {
    // Look for Jinja block openers: {{ or {%
    if (i + 1 < tmpl.size() && tmpl[i] == '{' &&
        (tmpl[i + 1] == '{' || tmpl[i + 1] == '%')) {
      const bool is_expr = (tmpl[i + 1] == '{');
      const std::string closer = is_expr ? "}}" : "%}";

      // Find the matching closer.
      size_t end = tmpl.find(closer, i + 2);
      if (end == std::string::npos) {
        // Unclosed block — copy the rest verbatim.
        result += tmpl.substr(i);
        break;
      }

      // Extract the block (including delimiters), rewrite only the interior.
      const std::string opener = tmpl.substr(i, 2);
      const std::string interior = tmpl.substr(i + 2, end - (i + 2));
      result += opener;
      result += replace_undefined_tests(interior);
      result += closer;
      i = end + 2;
    } else {
      result += tmpl[i];
      ++i;
    }
  }

  return result;
}

const std::unordered_map<std::string, std::string> type_to_modality = {
    {"video_url", "video"},
    {"image_url", "image"},
    {"audio_url", "audio"},
    {"image_embedding", "image"},
    {"video_embedding", "video"},
    {"audio_embedding", "audio"}};
}

JinjaChatTemplate::JinjaChatTemplate(const TokenizerArgs& args) : args_(args) {
  try {
    // Normalize "is undefined" → "is none" for Minja compatibility
    // (Qwen3.8-27B and other models that use Jinja2's "is undefined" test).
    const std::string normalized_template =
        normalize_minja_tests(args_.chat_template());
    template_ = std::make_unique<minja::chat_template>(
        normalized_template, args_.bos_token(), args_.eos_token());
    LOG(INFO) << "Jinja chat template init succeed.";

  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to parse jinja chat template, TokenizerArgs: "
               << args_ << std::endl
               << "Error message: " << e.what();
  }
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages) const {
  const std::vector<xllm::JsonTool> empty_tools;
  const nlohmann::ordered_json chat_template_kwargs;
  return apply(messages, empty_tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  const std::vector<xllm::JsonTool> empty_tools;
  return apply(messages, empty_tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    nlohmann::ordered_json& messages) const {
  // Call the overloaded method with empty tools
  nlohmann::ordered_json empty_tools = nlohmann::json::array();
  const nlohmann::ordered_json chat_template_kwargs = nlohmann::json::object();
  return apply(messages, empty_tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages,
    const std::vector<xllm::JsonTool>& json_tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  // convert the messages to json object
  nlohmann::ordered_json messages_json = nlohmann::json::array();
  for (const auto& message : messages) {
    nlohmann::ordered_json message_json;
    message_json["role"] = message.role;

    if (std::holds_alternative<std::string>(message.content)) {
      message_json["content"] = std::get<std::string>(message.content);
    } else if (std::holds_alternative<MMContentVec>(message.content)) {
      message_json["content"] =
          get_mm_content(std::get<MMContentVec>(message.content));
    }

    if (message.tool_call_id.has_value()) {
      message_json["tool_call_id"] = *message.tool_call_id;
    }

    if (message.reasoning_content.has_value()) {
      message_json["reasoning_content"] = *message.reasoning_content;
    }

    if (message.tool_calls.has_value()) {
      nlohmann::ordered_json tool_calls_json = nlohmann::json::array();
      const auto& tool_calls = *message.tool_calls;

      for (const auto& tool_call : tool_calls) {
        tool_calls_json.emplace_back(nlohmann::ordered_json{
            {"id", tool_call.id},
            {"type", tool_call.type},
            {"function",
             nlohmann::ordered_json{
                 {"name", tool_call.function.name},
                 {"arguments", tool_call.function.arguments}}}});
      }
      message_json["tool_calls"] = std::move(tool_calls_json);
    }

    messages_json.emplace_back(std::move(message_json));
  }

  nlohmann::ordered_json tools_json = nlohmann::json::array();

  for (const auto& json_tool : json_tools) {
    tools_json.emplace_back(nlohmann::ordered_json{
        {"type", json_tool.type},
        {"function",
         nlohmann::ordered_json{
             {"name", json_tool.function.name},
             {"description", json_tool.function.description},
             {"parameters", json_tool.function.parameters}}}});
  }
  // apply the template
  return apply(messages_json, tools_json, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    nlohmann::ordered_json& messages,
    const nlohmann::ordered_json& tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  try {
    minja::chat_template_inputs input;
    input.messages = messages;
    input.tools = tools;
    input.add_generation_prompt = true;
    input.extra_context = chat_template_kwargs;
    minja::chat_template_options options;

    return template_->apply(input, options);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to apply chat template: " << e.what();
    return std::nullopt;
  }
}

nlohmann::ordered_json JinjaChatTemplate::get_mm_content(
    const MMContentVec& vec) const {
  nlohmann::ordered_json content_json = nlohmann::json::array();

  for (const auto& item : vec) {
    nlohmann::ordered_json item_json;
    item_json["type"] = item.type;
    if (item.type == "text") {
      item_json["text"] = item.text;
    } else if (auto it = type_to_modality.find(item.type);
               it != type_to_modality.end()) {
      const std::string& modality = it->second;
      item_json[modality] = "mm place holder";
      item_json[item.type] = "mm place holder";
    } else {
      item_json[item.type] = "mm place holder";
    }

    content_json.emplace_back(item_json);
  }

  return content_json;
}

}  // namespace xllm
