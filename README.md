# MotifGNN-StyleLink

MotifGNN-StyleLink is a hybrid recommendation architecture designed for matchmaking and social discovery applications. It integrates stylometric Natural Language Processing (NLP) with Motif-based Graph Neural Networks (MGNN) to predict long-term compatibility. Instead of relying solely on static profile attributes, this system calculates compatibility by analyzing both a user's textual communication style and their structural position within a broader social interaction graph.

## Architecture Overview

The system operates through a two-stage sequential pipeline:

### 1. StyleLink Encoder

The first stage processes user communication data to generate a dense numerical representation of their conversational style.

* **Input:** 387 stylometric and linguistic features extracted from user text data.
* **Process:** A sequential neural network incorporates Batch Normalization (BatchNorm) and Rectified Linear Unit (ReLU) activations to compress these sparse features.
* **Output:** A dense 64-dimensional feature vector that serves as the user's stylistic fingerprint.

### 2. MotifGNN Engine

The second stage utilizes the stylistic profiles within a graph structure to evaluate compatibility based on network topology.

* **Input:** The 64-dimensional stylistic vectors mapped onto a dynamic user interaction graph (simulating historical interactions, connections, and passes).
* **Motif Convolutions:** The network calculates structural graph transformations across two specific topological micro-patterns:
* **Reciprocal Motifs:** Bidirectional edges representing direct mutual interest.
* **Community Clusters:** Dense sub-graphs indicating shared social circles or broader affinity groups.


* **Output:** 32-dimensional latent embeddings for each user. Final compatibility is determined via a highly efficient dot-product similarity calculation between two target user embeddings.

## Dataset and Provenance

The model's training pipeline is grounded in two primary research datasets:

* **Cornell Movie-Dialogs Corpus:** Utilized to derive the stylometric and linguistic features that form the conversational profiles.
* **Stanford Large Network Dataset Collection (SNAP):** Utilized to simulate the interaction network topology and graph structure required for the MGNN.

## Performance and Stability

* **Stability Mechanisms:** The architecture implements custom gradient clipping and a targeted final-layer adjustment to prevent the "ReLU Trap" (permanent node deactivation) during deep backpropagation.
* **Predictive Accuracy:** The model achieves a Test Area Under the ROC Curve (Test AUC) of approximately 0.87 to 0.88, indicating a strong predictive capacity for successful network pairings.

## Project Artifacts

* `mgnn.ipynb`: The complete end-to-end pipeline notebook handling feature normalization, node mapping, training loops, and recommendation generation.
* `vibematch_model.pth`: The trained state dictionary containing the neural network weights and tensor parameters for the MotifGNN.
* `vibe_embeddings.pt`: The pre-calculated tensor file containing the final processed user embeddings, structured for low-latency inference.
