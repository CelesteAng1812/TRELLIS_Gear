#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
import imageio
from typing import *
from easydict import EasyDict as edict

import gradio as gr
from gradio_litmodel3d import LitModel3D
from gradio_graph import app as LANGGRAPH_APP
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

import uuid
from pathlib import Path

TMP_DIR = Path("./tmp")   # temp file to save sessions 
TMP_DIR.mkdir(parents=True, exist_ok=True)

# create seperate directory under ./tmp for each session
def _user_dir(session_id):
    d = TMP_DIR / str(session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d

# to get unique session token
def _new_token():
    return uuid.uuid4().hex[:10]

# CSS for slight website design
css = """
.app-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: 0.2px;
    margin: 4px 0 2px 0;
}

.top-links {
    display: flex;
    gap: 12px;
    align-items: center;
    justify-content: flex-end;
    flex-wrap: nowrap;
}

.word-btn button {
    background: #4b5563 !important;
    border: 1px solid #6b7280 !important;
    border-radius: 12px !important;
    padding: 10px 18px !important;
    min-width: 150px !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.16) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    cursor: pointer;
    transition: 0.2s ease;
}

.word-btn button:hover {
    background: #5b6472 !important;
    border-color: #94a3b8 !important;
    transform: translateY(-1px);
}

.inline-popup {
    border: 1px solid #dbe2ea;
    border-radius: 18px;
    padding: 18px 22px 16px 22px;
    margin: 12px 0 14px 0;
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
    box-shadow: 0 12px 28px rgba(0,0,0,0.10);
    color: #0f172a !important;
}

.inline-popup * {
    color: #0f172a !important;
} 

.popup-title {
    font-size: 1.2rem;
    font-weight: 800;
    text-align: center;
    margin: 0 !important;
}

.close-btn button {
    background: #4b5563 !important;
    border: none !important;
    border-radius: 10px !important;
    width: 44px !important;
    min-width: 44px !important;
    height: 44px !important;
    padding: 0 !important;
    box-shadow: none !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    color: white !important;
    cursor: pointer;
}

.close-btn button:hover {
    background: #374151 !important;
}

.param-list {
    line-height: 1.5;
    text-align: center;
    margin-top: 0px !important;
    color: #111827 !important;
}

.param-list b {
    color: #111827 !important;
}
"""

# Toggle info popup visbility
def toggle_info(info_open):
    new = not info_open
    return new, gr.Column(visible=new)

# Toggle param list popup visbility
def toggle_params(params_open):
    new = not params_open
    return new, gr.Column(visible=new)

# Close popups
def close_info():
    return False, gr.Column(visible=False)

# From TRELLIS: To store gaussian and mesh information
def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }
    
# From TRELLIS: To get gaussian and mesh information   
def unpack_state(state: dict) -> Tuple[Gaussian, edict, str]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')
    
    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )
    
    return gs, mesh

# After graph is run, generate 3D gear and video rendering of gear 
def text_to_3d(prompt: str, session_id: str):
    outputs = pipeline.run(
        prompt,
        formats=["gaussian", "mesh"],
        sparse_structure_sampler_params={"steps": 25, "cfg_strength": 7.5},
        slat_sampler_params={"steps": 25, "cfg_strength": 7.5},
    )

    # Render preview video
    video = render_utils.render_video(outputs["gaussian"][0], num_frames=120)["color"]

    token = _new_token()
    out_dir = _user_dir(session_id)
    video_path = str(out_dir / f"sample_{token}.mp4")
    imageio.mimsave(video_path, video, fps=15)

    # store packed state + token (same pattern as app_text’s output_buf)
    packed = pack_state(outputs["gaussian"][0], outputs["mesh"][0])
    out_buf = {"token": token, "state": packed}

    torch.cuda.empty_cache()
    return out_buf, video_path

# Get glb information during extraction
def extract_glb(out_buf: dict, session_id: str):
    if not out_buf or "state" not in out_buf:
        raise gr.Error("No generated asset found yet. Generate first.")

    token = out_buf.get("token", "latest")
    gs, mesh = unpack_state(out_buf["state"])

    glb = postprocessing_utils.to_glb(gs, mesh, simplify=0.95, texture_size=1024, verbose=False)
    out_dir = _user_dir(session_id)
    glb_path = str(out_dir / f"sample_{token}.glb")
    glb.export(glb_path)
    torch.cuda.empty_cache()
    return glb_path, glb_path

# Get ply information during extraction
def extract_gaussian(out_buf: dict, session_id: str):
    if not out_buf or "state" not in out_buf:
        raise gr.Error("No generated asset found yet. Generate first.")

    token = out_buf.get("token", "latest")
    gs, _ = unpack_state(out_buf["state"])

    out_dir = _user_dir(session_id)
    gaussian_path = str(out_dir / f"sample_{token}.ply")
    gs.save_ply(gaussian_path)
    torch.cuda.empty_cache()
    return gaussian_path, gaussian_path

def _extract_glb_with_session(out_buf, req: gr.Request):
    return extract_glb(out_buf, session_id=str(req.session_hash))

def _extract_ply_with_session(out_buf, req: gr.Request):
    return extract_gaussian(out_buf, session_id=str(req.session_hash))

# Ensure variables available in functions
def _ensure_defaults(state: dict):
    state.setdefault("user_input", "")
    state.setdefault("awaiting_user", False)
    state.setdefault("question", None)

    state.setdefault("ui_log", [])
    if not isinstance(state["ui_log"], list):
        state["ui_log"] = [str(state["ui_log"])]

    state.setdefault("features", {
        "Module": None,
        "Teeth": None,
        "Face width": None,
        "Bore diameter": None,
    })
    state.setdefault("missing", [])
    state.setdefault("vague", False)
    state.setdefault("mode", None)

    state.setdefault("valid_gear", False)
    state.setdefault("gear_status", None)
    state.setdefault("final_output", None)

    state.setdefault("pending_confirmation", False)
    return state

# Handle communication between graph and front end chat
def chat_send_stream(message: str, history: list, graph_state: dict, req: gr.Request):
    thread_id = str(req.session_hash)

    history = (history or []) + [[message, ""]]
    graph_state = _ensure_defaults(graph_state or {})
    graph_state.setdefault("ui_log", [])
    graph_state["ui_log"].clear()

    graph_state["user_input"] = message

    # user input handling
    if graph_state.get("awaiting_user"):
        graph_state["awaiting_user"] = False
        graph_state["question"] = None

    last_state = graph_state
    last_log_len = len(last_state.get("ui_log", []))

    status_md = "**Status:** starting graph..."

    done = False
    final_out = ""
    extract_interactive = False
    trellis_state_out = None
    video_path_out = None

    # FIRST YIELD: clears textbox immediately + updates UI
    yield (
        gr.update(value="", interactive=False),      # textbox (clear), disable textbox
        history,                                     # chatbot
        last_state,                                  # graph_state
        status_md,                                   # status markdown
        done,                                        # done_state
        final_out,                                   # final_out_state
        gr.update(interactive=extract_interactive),  # extract_glb_btn
        gr.update(interactive=extract_interactive),  # extract_ply_btn
        trellis_state_out,
        video_path_out,
    )

    # stream graph so information is sent to user at each stage required
    for event in LANGGRAPH_APP.stream(
        graph_state,
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 250,},
        stream_mode="updates",
    ):
        update = {}
        if isinstance(event, dict):
            if len(event) == 1 and isinstance(next(iter(event.values())), dict):
                update = next(iter(event.values()))
            else:
                update = event

        last_state = _ensure_defaults({**last_state, **update})

        # show ui_log updates
        ui_log = last_state.get("ui_log", [])
        if isinstance(ui_log, list) and len(ui_log) > last_log_len:
            new_lines = ui_log[last_log_len:]
            if history[-1][1]:
                history[-1][1] += "\n" + "\n".join(new_lines)
            else:
                history[-1][1] = "\n".join(new_lines)
            last_log_len = len(ui_log)

        # awaiting user -> stop early to get user info before continuing
        if last_state.get("awaiting_user"):
            q = last_state.get("question") or "I need more information."
            if history[-1][1]:
                history[-1][1] += "\n\n" + q
            else:
                history[-1][1] = q

            yield (
                gr.update(value="", interactive=True), # Enable typing
                history,
                last_state,
                "**Status:** Waiting for input...",
                False,
                "",
                gr.update(interactive=False),
                gr.update(interactive=False),
                None,
                None,
            )
            return

        yield (
            gr.update(value="", interactive=False), # Disable textbox after user input received
            history,
            last_state,
            "**Status:** Running graph...",
            False,
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
            None,
            None,
        )

    # Finished graph step  
    done = (last_state.get("gear_status") == "confirmed")
    final_out = last_state.get("final_output", "") if done else ""

    if not history[-1][1]:
        history[-1][1] = "Gear confirmed" if done else "OK."

    if not done:
        status_md = "**Status:** Graph complete (not confirmed)"
        extract_interactive = False

        yield (
            gr.update(value="", interactive=True), # Allow textbox for next turn
            history,
            last_state,
            status_md,
            False,
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
            None, 
            None,
        )

        return

    if done:
        history[-1][1] += "\n\n Gear confirmed. Generating now..."
        status_md = "**Status:** Graph complete"

        yield (
            gr.update(value="", interactive=False), # Allow textbox for next turn
            history,
            last_state,
            status_md,
            True,
            final_out,
            gr.update(interactive=False),
            gr.update(interactive=False),
            None, 
            None,
        )

        try:
            trellis_state, video_path = text_to_3d(
                prompt=final_out,
                session_id=thread_id,
            )
            trellis_state_out = trellis_state
            video_path_out = video_path  
            extract_interactive = True
            status_md = "**Status:** Generation complete"
            history[-1][1] += "\n\n You can now view and extract your generated gear"
        except Exception as e:
            extract_interactive = False
            status_md = "**Status:** Generation failed"
            history[-1][1] += f"\n\n Generation failed: {str(e)}"
            trellis_state_out = None
            video_path_out = None


    yield (
        gr.update(value="", interactive=True), # Allow textbox for next turn
        history,
        last_state,
        status_md,        
        True,
        final_out,
        gr.update(interactive=extract_interactive),
        gr.update(interactive=extract_interactive),
        trellis_state_out, 
        video_path_out,
    )

# Layout and linking functions to front-end containers
with gr.Blocks(css=css) as demo:
    info_open = gr.State(True)
    params_open = gr.State(False)
    graph_state = gr.State({})
    trellis_state = gr.State(None)
    done_state = gr.State(False)
    final_out_state = gr.State("")
    trellis_state_out = None
    video_path_out = None

    with gr.Row():
        with gr.Column(scale = 8):
            gr.Markdown('<div class="app-title"> TRELLISGear </div>')
        with gr.Column(scale = 2, min_width = 150):
            # Information of how to use system and the relevant parameters
            with gr.Row(elem_classes="top-links"):
                info_btn = gr.Button("Information", elem_classes="word-btn")
                params_btn = gr.Button("Parameters", elem_classes="word-btn")

    with gr.Column(visible=True, elem_classes="inline-popup") as info_popup:
        with gr.Row():
            with gr.Column(scale=20):
                gr.Markdown('<div class="popup-title">How to use the app?</div>')
            with gr.Column(scale=1, min_width=20):
                info_close_btn = gr.Button("X", elem_classes="close-btn")
        gr.Markdown("<center>Use Chatbot to give desired gear features, then wait for generation to complete after confirmation. Click <b>Parameters</b> to see supported parameters.</center>")

    with gr.Column(visible=False, elem_classes="inline-popup") as params_popup:
        with gr.Row():
            with gr.Column(scale=20):
                gr.Markdown('<div class="popup-title">Supported Gear Parameters</div>')
            with gr.Column(scale=1, min_width=20):
                params_close_btn = gr.Button("X", elem_classes="close-btn")
        with gr.Row():
            with gr.Column(scale=20):
                gr.Markdown('<div class="param-list"><b>Teeth count</b>: Number of teeth on the gear<br><b>Module</b>: Size of teeth <br><b>Face width</b>: Thickness of Gear <br><b>Bore diameter</b>: Size of Central Hole in Gear<br></div>')
            with gr.Column(scale=1, min_width=20):
                gr.Markdown('')
        
    with gr.Row(equal_height=True):
        # Left: Chatbot
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=675)
            textbox = gr.Textbox(placeholder="Type your gear characteristics here", container=False)

        # Right: Generation outputs
        with gr.Column(scale=1):
            status = gr.Markdown("**Status:** waiting for graph...")

            video_output = gr.Video(label="Generated 3D Asset", autoplay=True, loop=True, height=300)

            with gr.Row():
                extract_glb_btn = gr.Button("Extract GLB", interactive=False)
                extract_ply_btn = gr.Button("Extract PLY", interactive=False)

            model_output = LitModel3D(label="Extracted GLB/Gaussian", exposure=10.0, height=300)

            with gr.Row():
                download_glb = gr.DownloadButton("Download GLB", interactive=False)
                download_ply = gr.DownloadButton("Download PLY", interactive=False)

    # textbox submit -> streaming handler
    textbox.submit(
        chat_send_stream,
        inputs=[textbox, chatbot, graph_state],
        outputs=[
            textbox,
            chatbot,
            graph_state,
            status,
            done_state,
            final_out_state,
            extract_glb_btn,
            extract_ply_btn,
            trellis_state,
            video_output,
        ],
    )

    # Information and Parameter Description Popups
    info_btn.click(
        toggle_info,
        inputs=[info_open],
        outputs=[info_open, info_popup])

    info_close_btn.click(
        close_info,
        outputs=[info_open, info_popup])

    params_btn.click(
        toggle_params,
        inputs=[params_open],
        outputs=[params_open, params_popup])

    params_close_btn.click(
        close_info,
        outputs=[params_open, params_popup])

    # extract object files
    extract_glb_btn.click(
        _extract_glb_with_session,
        inputs=[trellis_state],
        outputs=[model_output, download_glb],
    ).then(
        lambda: gr.update(interactive=True), 
        outputs=[download_glb])

    extract_ply_btn.click(
        _extract_ply_with_session,
        inputs=[trellis_state],
        outputs=[model_output, download_ply],
    ).then(
        lambda: gr.update(interactive=True), 
        outputs=[download_ply])


if __name__ == "__main__":
    pipeline = TrellisTextTo3DPipeline.from_pretrained("new/test_models")
    pipeline.cuda()

    demo.launch(share=True, debug=True) # Share=True for shareable link (required for VM)

