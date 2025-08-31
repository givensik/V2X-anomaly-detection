import pandas as pd
import os
import argparse
from pathlib import Path

def split_csv_file(input_csv_path: str, output_dir: str, chunk_size: int = 100000):
    """
    대용량 CSV 파일을 작은 파일들로 분할하여 저장하는 함수.
    
    Args:
        input_csv_path (str): 분할할 원본 CSV 파일 경로.
        output_dir (str): 분할된 파일들을 저장할 디렉토리.
        chunk_size (int): 각 분할 파일에 포함될 행의 수.
    """
    input_path = Path(input_csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading large file: {input_path}")
    
    # 원본 파일 이름에서 확장자를 제거하여 새 파일 이름의 접두사로 사용
    file_prefix = input_path.stem
    
    chunk_count = 0
    
    # pandas의 read_csv chunksize 인자를 사용하여 파일을 청크 단위로 읽음
    for chunk in pd.read_csv(input_csv_path, chunksize=chunk_size):
        output_file_name = f"{file_prefix}_chunk_{chunk_count}.csv"
        output_file_path = output_path / output_file_name
        
        print(f"Saving chunk {chunk_count} to {output_file_path}")
        
        chunk.to_csv(output_file_path, index=False)
        chunk_count += 1
    
    print(f"\nSuccessfully split {chunk_path.name} into {chunk_count} chunks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a large CSV file into smaller chunks.")
    parser.add_argument("--input_csv_path", type=str, required=True,
                        help="Path to the large CSV file to be split.")
    parser.add_argument("--output_dir", type=str, default="out/chunks",
                        help="Directory to save the smaller chunk files.")
    parser.add_argument("--chunk_size", type=int, default=100000,
                        help="Number of rows per chunk file.")
    
    args = parser.parse_args()
    
    split_csv_file(
        input_csv_path=args.input_csv_path,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size
    )