import torch
import argparse
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from semantic_cleaning import deduplicate_dataset

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset to be deduplicated.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model to compute embeddings.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer.")
    parser.add_argument("--epsilon", type=float, default=1e-2, help="Threshold for cosine similarity to consider embeddings as duplicates.")
    parser.add_argument("--model_batch_size", type=int, default=64, help="Batch size for the model.")
    parser.add_argument("--deduplication_batch_size", type=int, default=20000, help="Batch size for deduplication process.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of worker processes for data loading.")
    parser.add_argument("--dataset_feature", type=str, default="", help="Feature in the dataset to use for deduplication.")
    parser.add_argument("--output_path", type=str, default="./deduplicated_dataset", help="Path where the deduplicated dataset will be saved.")
    parser.add_argument("--hub_repo", type=str, default=None, help="Repository on the Hugging Face hub to push the dataset.")
    parser.add_argument("--hub_token", type=str, default=None, help="Hugging Face hub token.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computations (e.g., 'cpu', 'cuda:0', 'cuda:1').")
    
    return parser.parse_args()

def main(args):
    device = torch.device(args.device)
    auth_token = args.hub_token if args.hub_token else None
    dataset = load_dataset(args.dataset_path, use_auth_token=auth_token)
    model = AutoModel.from_pretrained(args.model_path, use_auth_token=auth_token).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_auth_token=auth_token)

    deduplicated_dataset = deduplicate_dataset(
        dataset=dataset, 
        model=model, 
        tokenizer=tokenizer,
        epsilon=args.epsilon, 
        model_batch_size=args.model_batch_size, 
        deduplication_batch_size=args.deduplication_batch_size, 
        num_workers=args.num_workers,
        dataset_feature=args.dataset_feature
    )

    if args.hub_repo is not None:
        # Push dataset to Hugging Face hub
        deduplicated_dataset.push_to_hub(
            repository=args.hub_repo,
            token=args.hub_token if args.hub_token else None
        )
    else:
        # Save deduplicated dataset locally
        deduplicated_dataset.save_to_disk(args.output_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
