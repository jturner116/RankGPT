import os
import gzip
import json
import re
from tqdm import tqdm
import logging
from rank_gpt import sliding_windows, write_eval_file
from env import GOOGLE_API_KEY
import multiprocessing
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


def load_topics(topics_file):
    topics = {}
    with open(topics_file, "r") as f:
        for line in f:
            qid, query = line.strip().split("\t")
            topics[qid] = query
    return topics


def load_results(results_file):
    results = {}
    with open(results_file, "r") as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            if qid not in results:
                results[qid] = []
            results[qid].append(
                {"docid": docid, "rank": int(rank), "score": float(score)}
            )
    return results


def process_topic(qid, query, hits, max_results=100, num_processes=25):
    hits = hits[:max_results]

    with multiprocessing.Pool(processes=num_processes) as pool:
        documents = list(
            tqdm(
                pool.imap(get_document_wrapper, [hit["docid"] for hit in hits]),
                total=len(hits),
                desc=f"Retrieving documents for query {qid}",
            )
        )

    item = {"query": query, "hits": []}
    for hit, doc in zip(hits, documents):
        if doc:
            item["hits"].append(
                {
                    "content": doc["content"],
                    "qid": qid,
                    "docid": hit["docid"],
                    "rank": hit["rank"],
                    "score": hit["score"],
                }
            )

    return item


def save_failed_topics(failed_topics, filename="gemini_test_fails.json"):
    with open(filename, "w") as f:
        json.dump(failed_topics, f)


def append_to_results_file(new_item, output_file):
    with open(output_file, "a") as f:
        for hit in new_item["hits"]:
            f.write(
                f"{hit['qid']} Q0 {hit['docid']} {hit['rank']} {hit['score']} rank\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerank topics using Gemini 2")

    parser.add_argument("--topics_file", type=str, help="Path to the topics file")
    parser.add_argument("--results_file", type=str, help="Path to the results file")
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    args = parser.parse_args()

    # topics_file = "topics.rag24.raggy-dev.txt"
    # results_file = "raggy-dev_results.txt"
    # output_file = "gemini_rerank_test.txt"
    api_key = GOOGLE_API_KEY

    # Clear the output file if it exists
    open(args.output_file, "w").close()

    topics = load_topics(args.topics_file)
    results = load_results(args.results_file)

    failed_topics = {}
    processed_count = 0

    for qid, query in tqdm(topics.items(), desc="Processing topics"):
        if qid in results:
            try:
                item = process_topic(qid, query, results[qid], max_results=100)
                new_item = sliding_windows(
                    item,
                    rank_start=0,
                    rank_end=100,
                    window_size=100,
                    model_name="gemini",
                    api_key=api_key,
                )
                append_to_results_file(new_item, args.output_file)
                processed_count += 1
            except Exception as e:
                error_message = str(e)
                logging.error(f"Error processing topic {qid}: {error_message}")
                failed_topics[qid] = {"query": query, "error": error_message}

    # Save failed topics
    if failed_topics:
        save_failed_topics(failed_topics)
        logging.info(f"Saved {len(failed_topics)} failed topics to failed_topics.json")

    print(f"Reranked results have been written to {args.output_file}")
    print(f"Number of successfully processed topics: {processed_count}")
    print(f"Number of failed topics: {len(failed_topics)}")
