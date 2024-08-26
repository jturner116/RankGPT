import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
import os
import gzip
import json
import tqdm
import multiprocessing
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Rerank search results using a specified model."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["jina", "bge"],
        default="jina",
        help="Choose the model to use for reranking: 'jina' or 'bge'",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="test_results.txt",
        help="Input file containing search results",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="reranked_results.txt",
        help="Output file for reranked results",
    )
    return parser.parse_args()


def get_document(doc_id, base_path="/root/data/msmarco_v2.1_doc_segmented/"):
    match = re.match(r"msmarco_v2\.1_doc_(\d+)_(\d+)#(\d+)_(\d+)", doc_id)
    if not match:
        raise ValueError(f"Invalid doc_id format: {doc_id}")

    shard_number = int(match.group(1))
    byte_offset = int(match.group(4))
    file_path = os.path.join(
        base_path, f"msmarco_v2.1_doc_segmented_{shard_number:02d}.json.gz"
    )

    with gzip.open(file_path, "rb") as f:
        f.seek(byte_offset)
        line = f.readline().decode("utf-8")

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


def get_document_wrapper(doc_id):
    try:
        document = get_document(doc_id)
        return {"docid": doc_id, "content": document["segment"]}
    except Exception as e:
        print(f"Error retrieving document {doc_id}: {e}")
        return None


def rerank_results(query, results, model, tokenizer=None, batch_size=32):
    print(f"Reranking for query: {query}")
    print(f"Number of results: {len(results)}")

    with multiprocessing.Pool(processes=25) as pool:
        documents = list(
            tqdm.tqdm(
                pool.imap(
                    get_document_wrapper, [result["doc_id"] for result in results]
                ),
                total=len(results),
                desc="Gathering documents",
            )
        )

    documents = [doc for doc in documents if doc is not None]
    print(f"Number of valid documents: {len(documents)}")

    all_scores = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        sentence_pairs = [[query, doc["content"]] for doc in batch]

        with torch.inference_mode():
            if tokenizer:  # BGE model
                inputs = tokenizer(
                    sentence_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                scores = (
                    model(**inputs, return_dict=True)
                    .logits.view(
                        -1,
                    )
                    .float()
                )

                all_scores.extend(scores.cpu().tolist())
            else:  # Jina model
                scores = model.compute_score(sentence_pairs, max_length=1024)
                all_scores.extend(scores)

    print(f"Total scores: {len(all_scores)}")

    # Combine scores with original results and sort
    for result, score in zip(results[: len(all_scores)], all_scores):
        result["new_score"] = score

    reranked_results = sorted(
        results[: len(all_scores)], key=lambda x: x["new_score"], reverse=True
    )
    return reranked_results


# ... (keep the existing parse_results function)
def parse_results(results_string):
    parsed_results = []
    for line in results_string.strip().split("\n"):
        parts = line.split()
        parsed_results.append(
            {
                "query_id": parts[0],
                "q0": parts[1],
                "doc_id": parts[2],
                "rank": int(parts[3]),
                "score": float(parts[4]),
                "run_id": parts[5],
            }
        )
    return parsed_results


def rerank_and_save(input_file, output_file, model, tokenizer=None):
    with open(input_file, "r") as f:
        input_data = f.read()

    results = parse_results(input_data)

    # Group results by query_id
    grouped_results = {}
    for result in results:
        query_id = result["query_id"]
        if query_id not in grouped_results:
            grouped_results[query_id] = []
        grouped_results[query_id].append(result)

    # Rerank queries and write results incrementally
    with open(output_file, "w") as f:
        for query_id, group in tqdm.tqdm(
            list(grouped_results.items()), desc="Reranking queries"
        ):
            print(f"\nProcessing query_id: {query_id}")
            query = f"Query for {query_id}"  # Replace with actual query if available
            reranked_results = rerank_results(query, group, model, tokenizer)

            # Write reranked results for this query
            for new_rank, result in enumerate(reranked_results, start=1):
                f.write(
                    f"{query_id} Q0 {result['doc_id']} {new_rank} {result['new_score']:.8f} jina-splade\n"
                )

            # Flush the file to ensure writing
            f.flush()


if __name__ == "__main__":
    args = parse_arguments()

    if args.model == "jina":
        model = AutoModelForSequenceClassification.from_pretrained(
            "jinaai/jina-reranker-v2-base-multilingual",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tokenizer = None
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            "BAAI/bge-reranker-v2-m3",
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    model.to("cuda")  # Use 'cpu' if no GPU is available
    model.eval()

    model = torch.compile(model)

    rerank_and_save(args.input_file, args.output_file, model, tokenizer)
