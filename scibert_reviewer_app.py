import os
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

def pdf_to_text(path: str) -> str:
    try:
        reader = PdfReader(path)
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)
    except Exception as e:
        print(f"[WARN] Failed to read PDF {path}: {e}")
        return ""


@st.cache_resource
def load_scibert(model_name: str = "allenai/scibert_scivocab_uncased", use_safetensors: bool = False):
    print(f"[INFO] Loading SciBERT model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        if use_safetensors:
            model = AutoModel.from_pretrained(model_name, use_safetensors=True)
        else:
            model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        print(f"[WARN] initial model load failed ({e}). Retrying without use_safetensors...")
        model = AutoModel.from_pretrained(model_name)
    model.eval()
    print("[INFO] Model loading complete.")
    return tokenizer, model


def mean_pooling(outputs, attention_mask):
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def embed_text(text: str, tokenizer, model, device="cpu", max_length=512) -> np.ndarray:
    if not isinstance(text, str):
        text = str(text)
    if len(text.strip()) == 0:
        return None
    encoded = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        pooled = mean_pooling(outputs, attention_mask)
    emb = pooled.cpu().numpy().reshape(-1)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


@st.cache_data
def build_dataset_embeddings(dataset_dir: str,
                             _tokenizer,
                             _model,
                             device="cpu",
                             cache_path: str = "scibert_embeddings_cache.npz") -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]:
    print(f"[INFO] Building/Loading dataset embeddings for {dataset_dir}")
    dataset_dir = Path(dataset_dir)
    author_to_papers = {}
    for author_folder in sorted(dataset_dir.iterdir()):
        if author_folder.is_dir():
            papers = []
            for p in sorted(author_folder.glob("*.pdf")):
                papers.append(str(p.resolve()))
            if papers:
                author_to_papers[author_folder.name] = papers

    if os.path.exists(cache_path):
        try:
            print(f"[INFO] Loading embeddings cache from {cache_path}")
            cache = np.load(cache_path, allow_pickle=True)
            paper_embeddings = {k: cache[k].item() for k in cache.files}
            paper_embeddings = {p: (np.array(v) if not isinstance(v, np.ndarray) else v) for p, v in paper_embeddings.items()}
            print("[INFO] Cache loaded successfully.")
            return author_to_papers, paper_embeddings
        except Exception as e:
            print(f"[WARN] Failed loading cache ({e}), will recompute.")

    paper_embeddings = {}
    print("[INFO] Computing embeddings for dataset papers...")
    
    progress_bar = st.progress(0, text="Computing dataset embeddings (this may take time)...")
    total_papers = sum(len(p) for p in author_to_papers.values())
    paper_count = 0
    
    for author, papers in author_to_papers.items():
        for paper in papers:
            try:
                text = pdf_to_text(paper)
                if not text.strip():
                    print(f"[WARN] Empty text for {paper}, skipping.")
                    continue
                emb = embed_text(text, _tokenizer, _model, device=device)
                if emb is None:
                    continue
                paper_embeddings[paper] = emb
            except Exception as e:
                print(f"[WARN] Error embedding {paper}: {e}")
            
            paper_count += 1
            progress_bar.progress(paper_count / total_papers, text=f"Computing: {os.path.basename(paper)}")

    progress_bar.empty()

    try:
        np.savez_compressed(cache_path, **{p: paper_embeddings[p] for p in paper_embeddings})
        print(f"[INFO] Saved embeddings cache to {cache_path}")
    except Exception as e:
        print(f"[WARN] Failed to save cache ({e})")

    return author_to_papers, paper_embeddings


def compute_all_paper_similarities(query_emb: np.ndarray,
                                     author_to_papers: Dict[str, List[str]],
                                     paper_embeddings: Dict[str, np.ndarray]) -> Dict[str, List[Tuple[str, float]]]:
    author_sims = {}
    for author, papers in author_to_papers.items():
        sims = []
        for p in papers:
            if p not in paper_embeddings:
                continue
            emb = paper_embeddings[p]
            sim = float(cosine_similarity(query_emb.reshape(1, -1), emb.reshape(1, -1))[0, 0])
            sims.append((p, sim))
        if sims:
            author_sims[author] = sims
    return author_sims


def rank_authors_by_topN(author_sims: Dict[str, List[Tuple[str, float]]], topN: int = 3) -> List[Tuple[str, float, List[Tuple[str, float]]]]:
    author_scores = {}
    author_top_papers = {}
    for author, sims in author_s.items():
        sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
        top_n = sims_sorted[:min(topN, len(sims_sorted))]
        score = float(np.mean([s for (_, s) in top_n]))
        author_scores[author] = score
        author_top_papers[author] = top_n
    ranked = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)
    return [(a, score, author_top_papers[a]) for (a, score) in ranked]


def find_top_k_authors_topN(query_path: str,
                              tokenizer,
                              model,
                              author_to_papers: Dict[str, List[str]],
                              paper_embeddings: Dict[str, np.ndarray],
                              device="cpu",
                              topk: int = 5,
                              topN: int = 3) -> List[Tuple[str, float, List[Tuple[str, float]]]]:
    text = pdf_to_text(query_path)
    if not text.strip():
        raise ValueError("Query PDF produced no text or could not be read.")
    q_emb = embed_text(text, tokenizer, model, device=device)

    author_sims = compute_all_paper_similarities(q_emb, author_to_papers, paper_embeddings)
    ranked = rank_authors_by_topN(author_sims, topN=topN)
    return ranked[:topk]


def main_app():
    st.title("SciBERT Reviewer Recommender")
    st.markdown("Upload a query PDF to find the most relevant reviewers from the dataset.")

    st.sidebar.header("Configuration")
    dataset_dir = st.sidebar.text_input(
        "Dataset Directory",
        value=r"C:\Users\Aniketh\Documents\Applied AI\assignment 1 dataset",
        help="Path to the directory containing author subfolders."
    )
    cache_path = st.sidebar.text_input(
        "Embeddings Cache File",
        value="scibert_embeddings_cache.npz",
        help="Path to save/load the computed dataset embeddings."
    )
    topk = st.sidebar.slider("Top-K Reviewers", 1, 20, 5, help="Number of top reviewers to display.")
    topN = st.sidebar.slider("Top-N Papers per Reviewer", 1, 10, 3, help="Number of top papers to average for each reviewer's score.")
    
    device_option = "cuda" if torch.cuda.is_available() else "cpu"
    device = st.sidebar.radio("Device", ("cpu", "cuda"), index=(1 if device_option == "cuda" else 0), help="Select 'cuda' if you have an NVIDIA GPU.")
    if device == "cuda" and not torch.cuda.is_available():
        st.sidebar.error("CUDA not available. Falling back to CPU.")
        device = "cpu"

    use_safetensors = st.sidebar.checkbox("Use SafeTensors", value=True, help="Try loading model using safetensors (if available).")

    uploaded_file = st.file_uploader("Upload your Query PDF", type="pdf")

    if uploaded_file is not None:
        
        st.toast(f"Loading SciBERT model...")
        tokenizer, model = load_scibert(use_safetensors=use_safetensors)
        st.toast("Model loaded successfully!")
        
        if device.startswith("cuda"):
            model.to(device)

        st.toast(f"Loading dataset embeddings from '{cache_path}'...")
        author_to_papers, paper_embeddings = build_dataset_embeddings(
            dataset_dir, tokenizer, model, device=device, cache_path=cache_path
        )
        st.toast("Dataset embeddings loaded!")
        
        if not paper_embeddings:
            st.error("No paper embeddings were found or computed. Please check your dataset directory path and permissions.")
            return

        query_pdf_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                query_pdf_path = tmp_file.name

            with st.spinner(f"Analyzing '{uploaded_file.name}'..."):
                topk_authors = find_top_k_authors_topN(
                    query_pdf_path,
                    tokenizer,
                    model,
                    author_to_papers,
                    paper_embeddings,
                    device=device,
                    topk=topk,
                    topN=topN
                )
            
            st.success(f"Analysis complete! Found top {len(topk_authors)} reviewers.")
            st.header(f"Top {topk} Reviewers (using Top-{topN} paper aggregation)")
            
            for i, (author, score, top_papers) in enumerate(topk_authors, 1):
                with st.container(border=True):
                    st.markdown(f"### {i}. {author}")
                    st.metric(label="Average Similarity Score", value=f"{score:.4f}")
                    st.markdown("**Contributing Papers:**")
                    for (paper_path, sim) in top_papers:
                        fname = os.path.basename(paper_path)
                        st.markdown(f"      - `{fname}` (Similarity: **{sim:.4f}**)")
        
        except ValueError as e:
            st.error(f"Error processing PDF: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
        finally:
            if query_pdf_path and os.path.exists(query_pdf_path):
                os.unlink(query_pdf_path)
    else:
        st.info("Please upload a PDF file to begin analysis.")


if __name__ == "__main__":
    main_app()