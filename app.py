import os
import re
from flask import Flask, render_template, request, Response
from google import genai
from google.genai import types


API_KEY = "AIzaSyC_E9bFV4ExKhptGYEbOxkgouHtOEIkN-M"
if not API_KEY:
    raise RuntimeError("🔑 No API key provided. Please set API_KEY in this file.")


client = genai.Client(api_key=API_KEY, vertexai=False)
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form.get("message", "").strip()

    prompt = f"""
User request: '{user_input}'

RULES:
- Output ONLY plain text in this exact order for each movie:
  Movie Name (Year)
  Director: <name>
  Synopsis: <short description>
  Available on: <comma-separated OTT platforms such as Netflix, Amazon Prime Video, Disney+ Hotstar, Hulu, Apple TV, Max, Paramount+>
- If OTT info is unknown/uncertain, write exactly: "OTT platform information not available".
- NEVER output the placeholder word "Platforms" (or "Platform") after "Available on:".
- No links, no HTML, no markdown.
- For category queries, return exactly 5 movies, numbered 1–5, each block formatted as above.
"""

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]
    config = types.GenerateContentConfig(
        system_instruction=[types.Part.from_text(
            text=(
                "You are a movie recommendation chatbot. "
                "Respond ONLY in the specified plain-text format. "
                "Do NOT include any links. "
                "After 'Available on:' list real OTT platforms (comma-separated). "
                "If unsure, output exactly 'OTT platform information not available'. "
                "Never output the literal word 'Platforms' as a placeholder."
            )
        )]
    )

    def generate_stream():
        full_text = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash-lite",
            contents=contents,
            config=config
        ):
            full_text += chunk.text or ""

        fixed_lines = []
        for line in full_text.splitlines():
            if line.strip().lower().startswith("available on:"):
                rhs = line.split(":", 1)[1].strip()
                rhs_clean = rhs.rstrip(" .,")

                if not rhs_clean or rhs_clean.lower() in {
                    "platform", "platforms", "n/a", "na", "unknown",
                    "not known", "none"
                }:
                    fixed_lines.append("Available on: OTT platform information not available")
                else:
                    fixed_lines.append(f"Available on: {rhs_clean}")
            else:
                fixed_lines.append(line)

        yield "\n".join(fixed_lines)

    return Response(generate_stream(), mimetype="text/html")

if __name__ == "__main__":
    app.run(debug=True)
