#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import math
from typing import Optional, TypedDict, List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    temperature=1.0,
    max_retries=2,
    google_api_key=api_key,
)

checkpointer = InMemorySaver()

INIT_FEATURES = {
    "Module": None,
    "Teeth": None,
    "Face width": None,
    "Bore diameter": None
}


# Initialise graph state
class AgentState(TypedDict, total=False):
    user_input: str
    awaiting_user: bool
    question: Optional[str]
    ui_log: List[str]

    features: Dict[str, Optional[float]]
    missing: List[str]
    vague: bool
    mode: str

    valid_gear: bool
    gear_status: Optional[str]
    final_output: Optional[str]
    last_question: Optional[str]

    # Helper for confirmation flow
    pending_confirmation: bool  # True if we're waiting for y/n

# Ensure that all state variables are available 
def ensure_defaults(state):
    state.setdefault("user_input", "")
    state.setdefault("awaiting_user", False)
    state.setdefault("question", None)

    state.setdefault("ui_log", [])
    if not isinstance(state["ui_log"], list):
        state["ui_log"] = [str(state["ui_log"])]

    state.setdefault("features", INIT_FEATURES.copy())
    state.setdefault("missing", [])
    state.setdefault("vague", False)
    state.setdefault("mode", None)

    state.setdefault("valid_gear", False)
    state.setdefault("gear_status", None)
    state.setdefault("final_output", None)
    state.setdefault("last_question", None)

    state.setdefault("pending_confirmation", False)
    return state

# Send ui messages to be shown to frontend
def ui_log(state, *msg):
    state = ensure_defaults(state)
    state["ui_log"].append(" ".join(map(str, msg)))
    return state

# routing
# Checks if user input requires inference from description, or extraction of parameters
def routing(state):
    state = ensure_defaults(state)
    user_input = state["user_input"]

    prompt = PromptTemplate(
        template = """You are a prompt classifier. 
        Return 1 if user provides numbers with feature names (e.g "Set teeth to 20", "Make bore bigger", "module 1, teeth 10, face width 10, bore 1") only
        Return 0 if user describes requirements without numeric parameters for module, face width, teeth and bore diameter (e.g. "compact", "fit on 5mm shaft") even if parameters are given
        Return only 1 or 0, no additional text

        Rules:
        - Return 1 only if mentioned teeth count, bore diameter, module or face width
        - For other numerical values (e.g. 1000W, 200 RPM, 50 db), return 0
        - If feature names (teeth count, bore diameter, module or face width) and additional information about gear requirements are given, return 0

        Classify the following user input.
        User input: {user_input}
        """
    )

    response = llm.invoke(prompt.format(user_input=user_input))
    raw = response.content
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    if cleaned == "1":
        state["mode"] = "extract"
    else:
        state["mode"] = "infer"

    return state

# is_functional
# For conditional node: Checks if current user input uses extraction or inference for parameters
def is_functional(state):
    if state["mode"] == "infer":
        return 1
    else:
        return 0

# ask_routing
# For conditional edge: Placeholder to get graph state
def ask_routing(state):
    return state


# feature_extraction
# Idea: the json can return value, null or remove. if remove, delete value
def extract_features(state):
    state = ensure_defaults(state)

    user_input = state["user_input"]

    prompt = PromptTemplate(
        template = """Your role is to extract relevant gear parameters from the following user input.

        Extract the following parameters only. Only extract when they are explicitly mentioned in the user input:
        1. Module: Tooth's dimension.
        2. Teeth: Number of teeth on the edge of a gear wheel.
        3. Face width: Axial length of a gear tooth.
        4. Bore diameter: Measurement of the central hole in a gear

        Rules:
        - Extract numerical values. If not mentioned, return null.
        - Extract number only if user states the value for the field (e.g. "teeth is 64", "module=3", "bore diameter of 20mm", "face width equal 10mm")
        - If user expresses relative change without giving final, return null 
          Relative change include: Bigger, smaller, increase, decrease, add, subtract, raise, lower, more, less
        - Do not convert relative change into set values. Do not do any arithmetic.

        Examples (follow but do not copy, base your extract on user input):
        - "Make it bigger": All null
        - "Increase bore by 20%: Bore diameter null
        - "add 10 teeth": Teeth null
        - "Reduce face width by 10mm": Face width null
        - "Set module to 3 and teeth to 64": Module = 3, Teeth = 64

        User Input: {user_input}

        Return only JSON: {{
            "Module": <value or null>,
            "Teeth": <value or null>,
            "Face width": <value or null>,
            "Bore diameter": <value or null>
            }}
        """,
        input_variables=["user_input"]
    )

    response = llm.invoke(prompt.format(user_input=user_input))
    raw = response.content
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    extracted = json.loads(cleaned)

    state["vague"] = all(v is None for v in extracted.values())

    # Update state, keeping existing features if new ones are null
    updating = state["features"]
    for k, v in extracted.items():
        if v is not None:
            updating[k] = v

    state["features"] = updating
    return state

# is_vague 
# For conditional edge: Gets if user input requires more than just direct extraction (e.g. manipulation of parameter values)
def is_vague(state):
    return 1 if state.get("vague") else 0

# arithmetic_change
# Infer what changes to parameters are needed and execute them
def arithmetic_change(state):
    state = ensure_defaults(state)

    user_input = state["user_input"]
    features = state["features"]

    prompt = PromptTemplate(
        template = """Your role is to edit gear parameters based on the following current features and user input.

        Current Features: {features}

        User Input: {user_input}

        Return JSON only. Follow format: 
        {{"ops":[...]}}

        Each op must be one of the following formats:
        - set {{"op":"set", "feature":<feature>, "value":<value>}}
        - add {{"op":"add", "feature":<feature>, "change":<value>}}
        - scale {{"op":"scale", "feature":<feature>, "factor":<value>}}
        - noop {{"op": "noop"}}

        Rules:
        - "feature" must be one of: Module, Teeth, Face width, Bore diameter with exact spelling and capitalization
        - Only change features mentioned in user input
        - You may return multiple ops if user requests multiple changes
        - If field is null, skip change and do not include in ops
        - If user gives explicit number to change a feature to, use "set" op
        - If no number is given for scaling bigger or smaller, use 1.2 for bigger and 0.8 for smaller 
        - If nothing safe applies, return {{"ops":[{{"op": "noop"}}]}}
        - Never ask questions, never invent values
        """,
        input_variables=["features", "user_input"]
    )

    response = llm.invoke(prompt.format(features=features,user_input=user_input))
    raw = response.content
    cleaned = raw.replace("```json", "").replace("```", "").strip()

    try:
        inferred = json.loads(cleaned)
        operations = inferred["ops"]
    except Exception as e:
        operations = [{"op": "noop"}]

    for operation in operations:
        op_type = operation["op"]

        if op_type == "set":
            feature = operation["feature"]
            value = operation["value"]
            if feature == "Teeth" or "teeth":
                state["features"][feature] = int(value)
            else:
                state["features"][feature] = float(value)

        elif op_type == "add":
            feature = operation["feature"]
            if state["features"][feature] is None:
                continue
            change = operation["change"]
            if feature == "Teeth" or "teeth":
                state["features"][feature] += int(change)
            else: 
                state["features"][feature] += float(change)

        elif op_type == "scale":
            feature = operation["feature"]
            if state["features"][feature] is None:
                continue
            factor = operation["factor"]
            state["features"][feature] *= float(factor)
            if feature == "Teeth" or "teeth":
                state["features"][feature] = int(state["features"][feature])

    return state


# missing_features
# Check which parameters are missing and return them 
def check_missing_features(state):
    feature_dict = state["features"]
    state["missing"] = [k for k, v in feature_dict.items() if v is None]
    return state

# have_enough_features
# For conditional edge: Check how many parameters missing
def have_enough_features(state):
    count = len(state["missing"])
    filled = 4-count

    if filled >= 3:
        return 1 # Sufficient features
    else:
        return 0 # Insufficient features

# more_information
# Based on missing parameters, ask a single question to user for more information
def request_additional_info(state):
    state = ensure_defaults(state)

    missing = state.get("missing", [])
    prompt = PromptTemplate(
        template = """Your role is to request additional information, based on the following missing parameters: {missing}.
        Ask the user specifically for these missing parameters only.
        Ask a single concise question. If no parameters are given, respond with only "No".
        """,
        input_variables=["missing"]
    )

    question = llm.invoke(prompt.format(missing=", ".join(missing)))
    q = str(question.content).strip()

    if q.lower() == "no":
        state["awaiting_user"] = False
        state["question"] = None
        return state

    state["awaiting_user"] = True
    state["question"] = q
    return state

# functional_infer 
# Based on user description of gear, infer to relevant parameters
def functional_infer(state):
    state = ensure_defaults(state)\

    user_input = state["user_input"]
    last_question = state["last_question"] if state["last_question"] else ""
    prompt = PromptTemplate(
        template = """You infer gear parameters from functional requirements described by the user.

        If a follow-up question was asked, it is shown below. If it is empty, treat the latest user input as additional functional description.

        Follow-up Question (may be empty):
        {last_question}

        Lastest User Input: {user_input}

        Infer values for: 
        - Module: Tooth's dimension (float in mm)
        - Teeth: Number of teeth around gear wheel (integer)
        - Face width: Axial length of gear tooth (float in mm)
        - Bore diameter: Central hole in gear (float in mm)

        Rules:
        - If the user answers with an everyday object size (e.g. dime/bottle cap), convert it into an approximate outer diameter in mm and use it as a size constraint to infer Module/Teeth.
        - If the user answers with general size constraint (e.g. small/big/medium), convert it into an approximate outer diameter in mm and use it as a size constraint to infer Module/Teeth.
        - If the user answers with a thickness constraint, use it to infer face width.
        - If the user answers about load requirement (e.g. torque level, force, motor power) use it to infer module and face width
        - If the user answers about speed or noise, use it to infer module and teeth count
        - If the input does NOT contain enough information to support a field confidently, return null for that field.
        - Do not ask questions, do not make up values beyond reasonable inference
        - Think through your inferences step by step
        
        Return a JSON: {{
            "parameters":{{
                "Module": <value or null>,
                "Teeth": <value or null>,
                "Face width": <value or null>,
                "Bore diameter": <value or null>
                }},
            "reasoning": "<your step by step reasoning here>"
            }}

        and your inference reasoning as a text explanation in the JSON "reasoning" field. Return only one JSON and no text outside the JSON.
        """,
        input_variables=["last_question", "user_input"]
    )

    response = llm.invoke(prompt.format(last_question=last_question, user_input=user_input))
    cleaned = str(response.content).replace("```json", "").replace("```", "").strip()
    extracted = json.loads(cleaned)
    inferred = extracted.get("parameters", {})

    # Update state, keeping existing features if new ones are null
    updating = state["features"]
    for k, v in inferred.items():
        if v is not None:
            updating[k] = v
    state["features"] = updating
    state["vague"] = False
    ui_log(state, "features=", state["features"])
    return state

# ask_more_functional
# Based on missing parameters, ask question to get more information, but no asking directly for parameters unlike for extract pathway
def ask_more_functional(state):
    state = ensure_defaults(state)

    missing = state.get("missing", [])
    last_question = state["last_question"] if state["last_question"] else ""
    prompt = PromptTemplate(
        template = """You need more information to infer gear parameters from functional requirements. Do not ask directly for module/teeth/face width/bore diameter values.

        Missing parameters: {missing}

        A previous question was asked:
        {last_question}

        Ask one concise question that would most help infer parameters. The question must directly be able to help infer one of the missing parameters.

        Rules:
        - Ask only one thing
        - Use everyday wording
        - Do not repeat or rephrase the previous question. 
        - Do not ask for the same parameter as the previous question.
        - Return only the question, do not include any additional text

        """,
        input_variables=["missing", "last_question"]
    )

    question = llm.invoke(prompt.format(missing=", ".join(missing), last_question=last_question))
    q = question.content.strip()

    state["last_question"] = q
    state["awaiting_user"] = True
    state["question"] = q
    return state


# validate_gear
# Check if parameter values and their combination is feasible to create a gear
def validate_gear(state):
    state = ensure_defaults(state)
    features = state["features"]

    m = features["Module"]
    t = features["Teeth"]
    face_w = features["Face width"]
    bore_d = features["Bore diameter"]

    errors = []
    warnings = []
    suggestions = []

    if m is not None and m <= 0:
        errors.append("Module must be positive")

    if t is not None: 
        if t <= 0:
            errors.append("Teeth count must be positive")
            suggestions.append("Consider teeth count of at least 17")
        elif t < 6:
            warnings.append("Unusally low teeth count")
            suggestions.append("Consider teeth count of at least 17")
        elif t < 17:
            warnings.append("Low teeth count may cause undercutting")
            suggestions.append("Consider teeth count of at least 17")

    if face_w is not None and face_w <= 0:
        errors.append("Face width must be positive")

    if bore_d is not None and bore_d <= 0:
        errors.append("Bore diameter must be positive")

    if m is not None and t is not None and bore_d is not None:
        pitch_d = m * t
        outside_d = pitch_d + 2*m
        root_d = pitch_d - 2.5*m
        base_d = pitch_d * math.cos(math.radians(20))

        if root_d <= 0:
            errors.append("Root diameter is non-positive, gear not feasible")
            suggestions.append(f"Increase module or teeth count")

        if bore_d <= root_d:
            wall_t = (root_d - bore_d) / 2
            if wall_t < 2.0 * m:
                warnings.append("Weak wall thickness detected")
                suggestions.append("Reduce bore diameter or increase module")
        else:
            errors.append("Bore diameter too large for gear size")
            suggestions.append(f"Reduce bore diameter to below {root_d:.1f} or increase module/teeth count")

        if base_d >= outside_d:
            errors.append("Insufficient tooth height, gear not feasible")

    if face_w is not None and m is not None:
        if face_w < 8.0 * m:
            warnings.append("Face width too narrow relative to module")
            suggestions.append(f"Increase face width to at least {8.0*m:.1f}")
        elif face_w > 12.0 * m:
            warnings.append("Face width unusually wide relative to module")
            suggestions.append(f"Decrease face width to at most {12.0*m:.1f}")
        
        if t is not None:
            pitch_d = m * t
            if face_w > pitch_d:
                errors.append("Face width should not exceed pitch diameter")
                suggestions.append(f"Reduce face width or increase module/teeth count")

    if state["pending_confirmation"] == False:
        if errors:
            state["valid_gear"] = False
            ui_log(state, "Errors:")
            for e in errors:
                ui_log(state, "-", e)
        else:
            state["valid_gear"] = True

        if warnings: 
            ui_log(state, "\nWarnings found:")
            for warn in warnings:
                ui_log(state, "- ", warn)

        if suggestions:
            ui_log(state, "\nSuggestions to fix errors and warnings:")
            for sugg in suggestions:
                ui_log(state, "- ", sugg)

    return state

# confirmation (named is_valid_gear in notebook) 
# Confirm user's next step (proceed with parameters or changes required)
def confirmation(state):
    state = ensure_defaults(state)

    user_msg = (state.get("user_input") or "").strip().lower()

    # If invalid, ask for revision text
    if not state.get("valid_gear", False):
        q = "\nThere are errors in the parameters. Please give parameters to be changed."
        ui_log(state, "\nGear parameters:", state["features"])
        state["awaiting_user"] = True
        state["question"] = q
        state["gear_status"] = "revising"
        return state

    # Parse y/n if was waiting for it
    if state.get("pending_confirmation", False):
        if user_msg in ["y", "yes"]:
            state["gear_status"] = "confirmed"
            state["pending_confirmation"] = False
            state["awaiting_user"] = False
            state["question"] = None
            return state
        if user_msg in ["n", "no"]:
            q = "Please give parameters to be changed."
            state["gear_status"] = "revising"
            state["pending_confirmation"] = False
            state["awaiting_user"] = True
            state["question"] = q
            return state

        # Invalid input: ask again 
        q = "Please reply with y/n. Would you like to confirm these parameters? [y/n]"
        state["pending_confirmation"] = True
        state["awaiting_user"] = True
        state["question"] = q
        return state

    # Ask for y/n and pause
    ui_log(state, "\nGear parameters:", state["features"])
    q = "Would you like to confirm these parameters? [y/n]"
    state["pending_confirmation"] = True
    state["awaiting_user"] = True
    state["question"] = q
    return state

# next_step
# For conditional edge: Placeholder to get gear_status
def next_step(state):
    return state.get("gear_status") or "revising"

# rewrite_query
# Rewrite parameters from the dictionary into an input string for the text-to-3D model
def final_query(state):
    state = ensure_defaults(state)
    data = state["features"]

    # Few-shot prompting examples for LLM to learn from 
    examples = [
        {"input": {"module": None, "teeth": 76, "face_width": 9.6, "bore_diameter": 28.3}, 
         "output": "A gear with 76 teeth. The face width is 9.6 mm, and the central bore diameter is 28.3 mm."},
        {"input": {"module": 5, "teeth": None, "face_width": 13.7, "bore_diameter": 16.4}, 
         "output": "A gear with a module of 5. The face width is 13.7 mm, and the central bore diameter is 16.4 mm"}, 
        {"input": {"module": 6, "teeth": 36, "face_width": None, "bore_diameter": 19.7}, 
         "output": "A gear with a module of 6 and 36 teeth. The central bore diameter is 19.7 mm."},
        {"input": {"module": 2, "teeth": 69, "face_width": 11.0, "bore_diameter": None}, 
         "output": "A gear with a module of 2 and 69 teeth. The face width is 11.0 mm"},
        {"input": {"module": 3, "teeth": 48, "face_width": 14.9, "bore_diameter": 14.3}, 
         "output": "A gear with a module of 3 and 48 teeth evenly spaced around its circumference. The face width is 14.9 mm, and the central bore diameter is 14.3 mm."},
    ]

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("{input}"),
            ("{output}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a query writer."),
            few_shot_prompt,
            ("{user_input}"),
        ]
    )

    response = llm.invoke(final_prompt.format(user_input=data))
    out = response.content.strip()
    state["final_output"] = out
    return state

# pauses (for user input in frontend)
def end_if_awaiting(state):
    return "wait" if state.get("awaiting_user") else "go"

def route_confirmation(state):
    if state.get("awaiting_user"):
        return "wait"
    return next_step(state)

# Construct graph
graph = StateGraph(AgentState)

# routing (inference or extraction)
graph.add_node("routing", routing)
graph.add_node("ask_routing", ask_routing)

# extraction nodes
graph.add_node("extract", extract_features)
graph.add_node("arithmetic", arithmetic_change)
graph.add_node("missing", check_missing_features)
graph.add_node("ask_more", request_additional_info)

# inference nodes
graph.add_node("infer", functional_infer)
graph.add_node("ask_more_f", ask_more_functional)

# feature validation and finalization nodes
graph.add_node("validate", validate_gear)
graph.add_node("confirmation", confirmation)
graph.add_node("rewrite", final_query)

# Graph construction
graph.add_edge(START, "routing")
graph.add_conditional_edges("routing", is_functional, {1: "infer", 0: "extract"})

graph.add_conditional_edges("extract", is_vague, {1: "arithmetic", 0: "missing"})
graph.add_edge("infer", "missing")
graph.add_edge("arithmetic", "missing")

graph.add_conditional_edges("missing", have_enough_features, {1: "validate", 0: "ask_routing"})
graph.add_conditional_edges("ask_routing", is_functional, {1: "ask_more_f", 0: "ask_more"})

graph.add_conditional_edges("ask_more", end_if_awaiting, {"wait": END, "go": "extract"})
graph.add_conditional_edges("ask_more_f", end_if_awaiting, {"wait": END, "go": "infer"})

# Validation of features after >=3 features 
graph.add_edge("validate", "confirmation")
graph.add_conditional_edges(
    "confirmation",
    route_confirmation,
    {"wait": END, "confirmed": "rewrite", "revising": "arithmetic"},
)

# Get final query
graph.add_edge("rewrite", END)

# Compile graph
app = graph.compile(checkpointer=checkpointer)

