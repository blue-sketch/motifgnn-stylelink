MotifGNN-StyleLink: VibeMatch Engine
MotifGNN-StyleLink is a hybrid recommendation architecture designed for dating applications. It combines stylometric NLP (capturing a user's "vibe") with Motif-based Graph Neural Networks (capturing social interaction patterns) to predict high-compatibility matches.

🧠 Architecture Overview
The system operates in two primary stages:

StyleLink Encoder (The "Vibe" Compressor):

Input: High-dimensional features (387 dimensions) extracted from user communication styles.

Process: A sequential neural network with BatchNorm and ReLU activation compresses these features into a dense, 64-dimensional "Vibe Vector".

Goal: To create a numerical fingerprint of a user's texting style and personality "vibe".

VibeMatch MGNN (The Social Matchmaker):

Input: The 64-dimensional Vibe Vectors and a graph of user interactions (swipes/matches).

Motif Convolutions: The model analyzes two distinct structural "motifs":

Reciprocal Matches: Signals direct mutual interest.

Community Clusters: Signals shared social circles or broader "types".

Output: Refined latent embeddings (32 dimensions) used to calculate compatibility scores via dot-product similarity.

📊 Dataset & Provenance
The training environment is grounded in two major research datasets:

Cornell Movie-Dialogs Corpus: Used to derive the stylometric and linguistic features that form the "vibe" profiles.

Stanford Social Profiles (SNAP): Used to simulate the network structure and interaction patterns (swipes, likes, and matches) required for the GNN.

🚀 Key Performance
Stability: Includes a "ReLU Trap" fix in the final layer and gradient clipping to ensure stable training.

Accuracy: Reaches a Test AUC of ~0.87-0.88, indicating strong predictive power for successful matches.

📁 Project Artifacts
vibematch_model.pth: The trained state dictionary for the MotifGNN.

vibe_embeddings.pt: The final processed user embeddings for inference.

mgnn.ipynb: The complete end-to-end pipeline from feature normalization to recommendation generation.
