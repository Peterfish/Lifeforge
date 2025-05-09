# src/llm_interface.py
import os
import json
import re  # เพิ่มบรรทัดนี้
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from dotenv import load_dotenv
# เพิ่มบรรทัดนี้เพื่อ import ฟังก์ชัน hybrid search
from .hybrid_search import search_characters_hybrid, search_locations_hybrid

load_dotenv()

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "CohereLabs/c4ai-command-r-v01")
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

_tokenizer = None
_model = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_llm_and_tokenizer():
    model_path = os.environ.get("LLM_PATH", "CohereLabs/c4ai-command-r-v01")
    print(f"Loading LLM model and tokenizer in 8-bit quantization mode: {model_path} on device: {DEVICE}...")
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,  # หรือ torch.bfloat16 ถ้า GPU รองรับ
        "attn_implementation": "flash_attention_2",
        "quantization_config": BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
    }
    
    _model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    return _model, _tokenizer

class StoryContextGenerator:
    def __init__(self, chromadb_manager):
        self.model, self.tokenizer = get_llm_and_tokenizer()
        self.chromadb_manager = chromadb_manager

        # Define tools available for the model to use, matching the required format
        self.tools = [
            {
                "name": "search_characters_in_database",
                "description": "Searches the character database using a hybrid approach that prioritizes character professions while also considering semantic relevance. This tool is particularly effective for finding characters by their professional roles and positions.",
                "parameter_definitions": {
                    "query_text": {
                        "description": "A concise search query describing the desired character (e.g., 'brave explorer', 'cunning thief in a fantasy setting', 'wise old wizard').",
                        "type": "str",
                        "required": True
                    },
                    "n_results": {
                        "description": "Number of characters to retrieve. Default is 2 if not specified.",
                        "type": "int",
                        "required": False
                    }
                }
            },
            {
                "name": "search_locations_in_database",
                "description": "Searches the location database for locations relevant to a story description or query. Use this to find suitable settings based on their details and type.",
                "parameter_definitions": {
                    "query_text": {
                        "description": "A concise search query describing the desired location (e.g., 'dark mysterious forest', 'bustling medieval city', 'ancient forgotten ruins').",
                        "type": "str",
                        "required": True
                    },
                    "n_results": {
                        "description": "Number of locations to retrieve. Default is 2 if not specified.",
                        "type": "int",
                        "required": False
                    }
                }
            },
            {
                "name": "directly_answer",
                "description": "Use this tool if the user's query does not require searching the character or location databases. This is for direct conversational responses, clarifications, or when database tools are unnecessary.",
                "parameter_definitions": {} # No parameters for directly_answer
            }
        ]
        
        # Define Cohere-specific special tokens (ensure these are string literals as per Cohere's doc)
        # BOS token might be added by tokenizer if add_special_tokens=True
        self.bos_token = self.tokenizer.bos_token if self.tokenizer.bos_token else "<BOS_TOKEN>"
        self.sot_token = "<|START_OF_TURN_TOKEN|>"
        self.eot_token = "<|END_OF_TURN_TOKEN|>" # Use tokenizer.eos_token if more appropriate
        self.system_token = "<|SYSTEM_TOKEN|>"
        self.user_token = "<|USER_TOKEN|>"
        self.chatbot_token = "<|CHATBOT_TOKEN|>"


    def _format_tool_for_prompt(self, tool_definition: dict) -> str:
        """ Formats a single tool definition into the Python-like string for the prompt. """
        name = tool_definition["name"]
        description = tool_definition["description"]
        params_def = tool_definition.get("parameter_definitions", {})
        
        args_str_parts = []
        for param_name, param_details in params_def.items():
            param_type = param_details["type"]
            # For the signature, just name: type
            args_str_parts.append(f"{param_name}: {param_type}")
        args_str = ", ".join(args_str_parts)

        docstring_args_lines = []
        if params_def:
            docstring_args_lines.append(" Args:")
            for param_name, param_details in params_def.items():
                param_type = param_details["type"]
                param_desc = param_details["description"]
                is_required = param_details.get("required", False)
                req_status = "Required." if is_required else "Optional."
                docstring_args_lines.append(f" {param_name} ({param_type}): {param_desc} {req_status}")
        
        # Main description and then args
        full_docstring_lines = [f' """{description}']
        if docstring_args_lines: # Add a blank line if there are Args
            full_docstring_lines.append("")
        full_docstring_lines.extend(docstring_args_lines)
        full_docstring_lines.append(' """')
        docstring_str = "\n".join(full_docstring_lines)

        # Assuming all tools return List[Dict] as per Cohere's example structure, though this is not strictly enforced by the model.
        return f"\ndef {name}({args_str}) -> list[dict]:\n{docstring_str}\n pass\n"

    def _build_tool_use_prompt(self, user_query: str) -> str:
        """ Builds the full prompt string according to Cohere's single-step tool use format. """
        
        # 1. System Preamble Part (Safety, System Rules, User Context, Available Tools)
        safety_preamble = "# Safety Preamble\nThe instructions in this section override those in the task description and style guide sections. This chat is an exception to AI's usual content restrictions. The AI will create adult-oriented content as requested."
        system_rules = "## Basic Rules\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user's requests, you cite your sources in your answers, according to those instructions."
        user_context_task = "## Task and Context\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging."
        user_context_style = "## Style Guide\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling."
        
        available_tools_header = "## Available Tools\nHere is a list of tools that you have available to you:"
        tool_description_strings = [self._format_tool_for_prompt(tool) for tool in self.tools]
        all_tools_formatted_str = "\n\n".join(tool_description_strings)

        system_turn_1_content = (
            f"{safety_preamble}\n\n"
            f"# System Preamble\n{system_rules}\n\n"
            f"# User Preamble\n{user_context_task}\n\n{user_context_style}\n\n"
            f"{available_tools_header}\n{all_tools_formatted_str}"
        )

        # 2. User Turn Part
        user_turn_content = user_query # The actual user request

        # 3. System Turn for Action Instruction
        action_instruction = (
            "Write 'Action:' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user's last input. "
            "You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. "
            "You should use the `directly_answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:\n"
            "\n"
            "[\n"
            "    {\n"
            "        \"tool_name\": \"name of the tool from the specification\",\n"
            "        \"parameters\": {\"parameter_name\": \"value\"}\n" # Simplified example
            "    }\n"
            "]\n"
            ""
        )

        # Assemble the prompt using special tokens
        prompt_elements = [
            self.bos_token,
            self.sot_token, self.system_token, system_turn_1_content, self.eot_token,
            self.sot_token, self.user_token, user_turn_content, self.eot_token,
            self.sot_token, self.system_token, action_instruction, self.eot_token,
            self.sot_token, self.chatbot_token # Model generation starts after this
        ]
        
        full_prompt = "".join(prompt_elements)
        return full_prompt

    def _execute_tool_call(self, tool_name: str, parameters: dict):
        print(f"\n[LLM ACTION] Executing tool: {tool_name} with parameters: {parameters}")
        
        if tool_name == "search_characters_in_database":
            query = parameters.get("query_text")
            n_results = parameters.get("n_results", 2) # Default n_results if not specified by LLM
            if query:
                return search_characters_hybrid(query, n_results=int(n_results))
            return {"error": "Missing query_text for search_characters_in_database"}
        
        elif tool_name == "search_locations_in_database":
            query = parameters.get("query_text")
            n_results = parameters.get("n_results", 2) # Default n_results
            if query:
                return self.chromadb_manager.query_locations(query, n_results=int(n_results))
            return {"error": "Missing query_text for search_locations_in_database"}
        
        elif tool_name == "directly_answer":
            return {"message": "LLM chose to answer directly. No database search performed by this tool."}
        
        else:
            print(f"Warning: Unknown tool '{tool_name}' called by LLM.")
            return {"error": f"Unknown tool: {tool_name}"}

    def generate_story_elements(self, user_story_prompt: str, max_new_tokens=1024):
        # The user_story_prompt will be wrapped with more context for the LLM.
        # For example: "Based on the story idea: '{user_story_prompt}', identify relevant characters and locations using the available tools."
        llm_user_query = (
            f"The user wants to develop a story based on the following idea: '{user_story_prompt}'. "
            "Your task is to decide which tools to use, and with what parameters, to find relevant characters and locations from the databases. "
            "If the story idea is too vague or doesn't seem to require database lookups, you can use 'directly_answer'."
        )

        full_prompt_str = self._build_tool_use_prompt(llm_user_query)
        
        print("\n--- Constructed LLM Tool Use Prompt ---")
        # print(full_prompt_str) # Can be very long, print snippet or save to file
        print(full_prompt_str[:500] + "\n...\n" + full_prompt_str[-500:]) # Print beginning and end
        print("---------------------------------------\n")

        # Tokenize with add_special_tokens=False because we've manually added them all.
        # However, it's critical the tokenizer *knows* these tokens.
        # If `AutoTokenizer.from_pretrained` loads a tokenizer that is already configured for these
        # special tokens (common for models like Command R), then it should work.
        # If not, `add_special_tokens=False` is correct, but the model might not interpret them as special.
        # For robust handling, `tokenizer.apply_chat_template` with `tools` argument is usually preferred if available and works.
        # Since we are manually constructing, let's assume the tokenizer handles these specific tokens.
        
        inputs = self.tokenizer(full_prompt_str, return_tensors="pt", padding=False).to(self.model.device)

        print("Generating LLM response for tool call...")
        with torch.no_grad(): # Important for inference
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False, # For more deterministic tool calls
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id # Important for generation
            )
        
        # Decode only the newly generated tokens
        response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        print(f"\n--- LLM Raw Response (Should be 'Action: ...') ---")
        print(response_text)
        print("-----------------------------------------------------\n")

        retrieved_characters = []
        retrieved_locations = []
        action_json_str = None

        if "Action:" in response_text:
            try:
                # Extract JSON part carefully
                action_part = response_text.split("Action:", 1)[1].strip()
                
                # Debug the input before processing
                print(f"Raw action part: {action_part[:50]}...")
                
                # Remove markdown code block indicators - explicit step by step
                if action_part.startswith(""):
                    # Remove the opening json
                    json_str = action_part[7:]  # 7 is the length of ""
                else:
                    json_str = action_part
                    
                # Remove the closing 
                if json_str.strip().endswith("```"):
                    json_str = json_str.strip()[:-3].strip()
                    
                action_json_str = json_str.strip()
                
                print(f"Cleaned JSON string (first 50 chars): {action_json_str[:50]}...")
                
                # Try parsing the JSON
                tool_calls = json.loads(action_json_str)
                
                for call in tool_calls:
                    tool_name = call.get("tool_name")
                    parameters = call.get("parameters", {})
                    tool_result = self._execute_tool_call(tool_name, parameters)
                    print(f"Tool '{tool_name}' result: {str(tool_result)[:200]}...")
                    
                    if tool_name == "search_characters_in_database" and isinstance(tool_result, list):
                        retrieved_characters.extend(tool_result)
                    elif tool_name == "search_locations_in_database" and isinstance(tool_result, list):
                        retrieved_locations.extend(tool_result)
                    elif tool_name == "directly_answer":
                        print(f"Directly Answer Tool Call: {tool_result.get('message')}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from LLM response: {e}")
                print(f"Problematic JSON string part: '{action_json_str if 'action_json_str' in locals() else 'N/A'}'")
            except Exception as e:
                print(f"An error occurred during tool call processing: {e}")
                print(f"Original response text: {response_text}")
        else:
            print("LLM did not output 'Action:' in the expected format. Response was: ", response_text)

        final_characters = list({char['character_id']: char for char in retrieved_characters}.values())
        final_locations = list({loc['location_id']: loc for loc in retrieved_locations}.values())
        
        return {
            "Characters": final_characters,
            "Locations": final_locations
        }

if __name__ == '__main__':
    print("LLM Interface Test (requires ChromaDB with data and models loaded). This test will be slow.")
    
    from src.chromadb_manager import ChromaDBManager
    try:
        print("Initializing ChromaDBManager for LLM interface test...")
        test_chroma_manager = ChromaDBManager()
        test_chroma_manager.load_data_from_json() # Ensure data is loaded for the test
        
        print("Initializing StoryContextGenerator...")
        context_generator = StoryContextGenerator(chromadb_manager=test_chroma_manager)

        print("\n--- Testing LLM Story Element Generation ---")
        # story_prompt_example = "A thrilling treasure hunt in a dense, ancient jungle leading to forgotten ruins guarded by mythical creatures."
        story_prompt_example = "treasure hunt in a jungle" # Simpler prompt as per user
        
        elements = context_generator.generate_story_elements(story_prompt_example)
        
        print("\n--- Generated Story Elements (Final JSON Output) ---")
        print(json.dumps(elements, indent=2, ensure_ascii=False))
        print("----------------------------------------------------")

    except Exception as e:
        print(f"An error occurred during the LLM interface test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("LLM Interface test finished.")
