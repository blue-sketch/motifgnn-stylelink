import weasyprint
import markdown
import os

markdown_content = """
# MotifGNN-StyleLink

This project implements a hybrid recommendation system for dating applications, integrating stylometric Natural Language Processing (NLP) with Motif-based Graph Neural Networks (MGNN). By analyzing both textual communication styles and social interaction graphs, the system aims to predict long-term compatibility more accurately than traditional metadata-matching approaches.

## Architecture

The system operates through a two-stage pipeline:

### 1. StyleLink Encoder

The StyleLink Encoder processes user communication data to generate a dense representation of their conversational style.

*   **Input:** 387 stylometric and linguistic features extracted from user text data.
*   **Architecture:** A sequential neural network layer incorporating Batch Normalization (BatchNorm) and Rectified Linear Unit (ReLU) activations.
*   **Output:** A 64-dimensional feature vector, functioning as the user's stylistic profile.

### 2. VibeMatch MGNN

The VibeMatch MGNN utilizes the stylistic profiles within a graph structure to evaluate compatibility based on network topology.

*   **Input:** The 64-dimensional stylistic vectors mapped onto a user interaction graph (e.g., swipe and match history).
*   **Motif Convolutions:** The network calculates structural graph transformations across two specific topologies:
    *   **Reciprocal Motifs:** Bidirectional edges representing mutual interest.
    *   **Community Clusters:** Dense sub-graphs indicating shared social circles or affinity groups.
*   **Output:** 32-dimensional latent embeddings for each user. Compatibility is determined via a dot-product similarity calculation between two user embeddings.

## Data Provenance

The model is trained on the following datasets:

*   **Cornell Movie-Dialogs Corpus:** Utilized for extracting stylometric and linguistic features.
*   **Stanford Large Network Dataset Collection (SNAP):** Utilized to simulate the interaction network topology.

## Performance and Stability

*   **Stability Mechanisms:** The architecture implements gradient clipping and a specific final-layer adjustment to prevent the "ReLU Trap" during deep backpropagation.
*   **Accuracy:** The model achieves a Test Area Under the ROC Curve (AUC) of approximately 0.87 to 0.88.

## Project Files

*   `mgnn.ipynb`: The primary pipeline notebook containing data normalization, node mapping, training loops, and the match output calculation.
*   `vibematch_model.pth`: The optimized state dictionary containing the neural network weights and tensor parameters.
*   `vibe_embeddings.pt`: The pre-calculated tensor file containing the final processed user embeddings.
"""

html_content = markdown.markdown(markdown_content)

full_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  @page {{
      size: A4;
      margin: 20mm;
      background-color: #ffffff;
  }}
  body {{
      font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
      color: #333333;
      line-height: 1.6;
      font-size: 11pt;
  }}
  h1 {{
      font-size: 24pt;
      color: #2c3e50;
      border-bottom: 2px solid #ecf0f1;
      padding-bottom: 10px;
      margin-bottom: 20px;
  }}
  h2 {{
      font-size: 16pt;
      color: #2980b9;
      margin-top: 30px;
      margin-bottom: 15px;
  }}
  h3 {{
      font-size: 13pt;
      color: #34495e;
      margin-top: 20px;
      margin-bottom: 10px;
  }}
  p {{
      margin-bottom: 15px;
  }}
  ul {{
      margin-bottom: 15px;
      padding-left: 20px;
  }}
  li {{
      margin-bottom: 5px;
  }}
  code {{
      font-family: 'Courier New', Courier, monospace;
      background-color: #f8f9fa;
      padding: 2px 4px;
      border-radius: 3px;
      font-size: 0.9em;
  }}
</style>
</head>
<body>
{html_content}
</body>
</html>
"""

with open("readme.html", "w") as f:
    f.write(full_html)

weasyprint.HTML("readme.html").write_pdf("README_MotifGNN-StyleLink.pdf")
print("README_MotifGNN-StyleLink.pdf generated successfully.")
