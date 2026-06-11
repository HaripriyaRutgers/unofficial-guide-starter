"""
app.py — Gradio UI for the CS-careers RAG system.

A minimal gr.Blocks() interface: type a question, get a grounded answer plus
the list of sources the context was drawn from.

Run with:  python app.py
"""

import gradio as gr

from generate import ask  # the grounded generation function


def handle(question):
    """UI handler: run ask() and return (answer_text, sources_bullets).

    Returns two strings because the UI has two output textboxes.
    """
    # Guard against empty submissions so we don't waste an API call.
    if not question or not question.strip():
        return "Please enter a question.", ""

    result = ask(question)
    # Render the source list as a bullet list for the Sources textbox.
    sources_text = "\n".join(f"• {s}" for s in result["sources"])
    return result["answer"], sources_text


with gr.Blocks(title="CS Careers Beyond SWE") as demo:
    gr.Markdown("# CS Careers Beyond Software Engineering\n"
                "Ask about non-SWE career pathways for CS majors.")

    # Input row: question textbox + Ask button.
    inp = gr.Textbox(label="Your question", placeholder="e.g. What does a UX researcher do?")
    btn = gr.Button("Ask", variant="primary")

    # Output textboxes.
    answer_out = gr.Textbox(label="Answer", lines=8)
    sources_out = gr.Textbox(label="Sources", lines=4)

    # Wire BOTH the button click and the Enter key (inp.submit) to the handler.
    btn.click(fn=handle, inputs=inp, outputs=[answer_out, sources_out])
    inp.submit(fn=handle, inputs=inp, outputs=[answer_out, sources_out])


if __name__ == "__main__":
    # launch() starts a local web server (default http://127.0.0.1:7860).
    demo.launch()
