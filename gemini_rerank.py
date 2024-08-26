import os
import gzip
import json
import re
from rank_gpt import permutation_pipeline
from env import GOOGLE_API_KEY
from tqdm import tqdm
import multiprocessing
from trec_eval import EvalFunction
import tempfile
from argparse import ArgumentParser


def get_document_wrapper(doc_id):
    try:
        document = get_document(doc_id)
        return {"content": document["segment"]}
    except Exception as e:
        print(f"Error retrieving document {doc_id}: {e}")
        return None


def get_document(doc_id, base_path="/root/data/msmarco_v2.1_doc_segmented/"):
    # Parse the doc_id
    match = re.match(r"msmarco_v2\.1_doc_(\d+)_(\d+)#(\d+)_(\d+)", doc_id)
    if not match:
        raise ValueError(f"Invalid doc_id format: {doc_id}")

    shard_number = int(match.group(1))
    byte_offset = int(match.group(4))
    # Construct the file path
    file_path = os.path.join(
        base_path, f"msmarco_v2.1_doc_segmented_{shard_number:02d}.json.gz"
    )

    # Open the file and seek to the byte offset
    with gzip.open(file_path, "rb") as f:
        f.seek(byte_offset)
        line = f.readline().decode("utf-8")

        # Parse the JSON line
        try:
            document = json.loads(line)
            if document["docid"] == doc_id:
                return document
            else:
                raise ValueError(
                    f"Document at offset does not match requested doc_id: {doc_id}"
                )
        except json.JSONDecodeError:
            raise ValueError(
                f"Invalid JSON at offset {byte_offset} in file {file_path}"
            )


def read_queries(topics_file):
    queries = {}
    with open(topics_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                query_id = parts[0]
                query_text = " ".join(parts[1:])
                queries[query_id] = query_text
    return queries


def process_input_file(input_file_path, topics_file_path, output_file_path, top_k=125):
    queries = read_queries(topics_file_path)

    # Read the input file and group documents by query
    query_docs = {}
    with open(input_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                query_id, _, doc_id, rank, score, _ = parts[:6]
                if int(rank) <= top_k:
                    if query_id not in query_docs:
                        query_docs[query_id] = []
                    query_docs[query_id].append((doc_id, int(rank), float(score)))

    # Process each query
    results = []
    for query_id, docs in tqdm(query_docs.items(), desc="Processing queries"):
        if query_id not in queries:
            print(f"Warning: Query ID {query_id} not found in topics file")
            continue

        print(f"Processing query {query_id}: {queries[query_id]}")

        # Retrieve document content
        with multiprocessing.Pool(processes=16) as pool:
            hits = list(
                tqdm(
                    pool.imap(get_document_wrapper, [doc_id for doc_id, _, _ in docs]),
                    total=len(docs),
                    desc="Retrieving documents",
                )
            )

        # Prepare item for reranking
        item = {"query": queries[query_id], "hits": []}
        for hit, (doc_id, rank, score) in zip(hits, docs):
            if hit is not None:
                item["hits"].append(
                    {
                        "content": hit["content"],
                        "qid": query_id,
                        "docid": doc_id,
                        "rank": rank,
                        "score": score,
                    }
                )

        # Rerank using Gemini
        try:
            new_item = permutation_pipeline(
                item,
                rank_start=0,
                rank_end=len(item["hits"]),
                model_name="gemini",
                api_key=GOOGLE_API_KEY,
            )
            results.append(new_item)
        except Exception as e:
            print(f"Error reranking query {query_id}: {e}")

    # Write reranked results to output file
    write_results(results, output_file_path)

    return results


def write_results(results, output_file_path):
    with open(output_file_path, "w") as out_f:
        for item in results:
            for hit in item["hits"]:
                out_f.write(
                    f"{hit['qid']} Q0 {hit['docid']} {hit['rank']} {hit['score']} gemini_rerank\n"
                )


if __name__ == "__main__":
    argparse = ArgumentParser()

    argparse.add_argument("--evaluate", action="store_true", default=False)
    argparse.add_argument("--input_file", type=str)
    argparse.add_argument("--topics_file", type=str)
    argparse.add_argument("--output_file", type=str)
    argparse.add_argument("--qrels_file", type=str)
    argparse.add_argument("--eval_output_file", type=str)
    # Usage
    # results_file_path = "raggy-dev_results.txt"
    # topics_file_path = "topics.rag24.raggy-dev.txt"
    # output_file_path = "gemini_raggy-dev_results.txt"
    # qrels_file_path = "qrels.rag24.raggy-dev.txt"
    # eval_output_file = "gemini_raggy-dev_eval.txt"

    args = argparse.parse_args()
    results = process_input_file(args.input_file, args.topics_file, args.output_file)

    # If you want to evaluate using trec_eval, you can add:
    # from trec_eval import EvalFunction
    # EvalFunction.main('your_qrels_file', output_file_path)
    if args.evaluate:
        temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt")
        EvalFunction.write_file(results, temp_file.name)
        temp_file.close()

        all_metrics = EvalFunction.main(args.qrels_file_path, temp_file.name)

        with open(args.eval_output_file, "w") as f:
            json.dump(all_metrics, f, indent=2)
