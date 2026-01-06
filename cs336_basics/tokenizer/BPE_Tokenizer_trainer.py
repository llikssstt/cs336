"""BPE Tokenizer Trainer - Optimized Version

Optimizations:
1. Incremental update + Priority Queue for _compute_merges
2. Parallel pretokenization with multiprocessing
3. Fixed duplicate imports and type annotations
"""

from collections import defaultdict
from typing import List, Dict, BinaryIO, Tuple
from multiprocessing import Pool, cpu_count
import heapq
import os
import regex as re


# ============= Parallel pretokenization worker (module-level for pickle) =============
def _process_chunk_worker(args: Tuple[str, int, int, str, str]) -> List[str]:
    """Worker function for parallel chunk processing."""
    filepath, start, end, pattern, PAT = args
    with open(filepath, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        chunk = chunk.replace('\r\n', '\n').replace('\r', '\n')
        chunk_parts = re.split(pattern, chunk)
        return [match.group() for chunk_item in chunk_parts 
                for match in re.finditer(PAT, chunk_item)]


class MaxHeapPair:
    """Helper class to reverse pair comparison for min-heap to act as max-heap tie-breaker."""
    def __init__(self, pair):
        self.pair = pair
    
    def __lt__(self, other):
        # We want larger pairs to be popped first (to match max() behavior),
        # so larger pairs must be considered "smaller" in the min-heap.
        return self.pair > other.pair
    
    def __eq__(self, other):
        return self.pair == other.pair
    
    def __repr__(self):
        return f"MaxHeapPair({self.pair})"




class RegexTokenizer:
    def __init__(self, filepath:str, PAT=None, pattern=None, special_tokens: List[str]=None ):
        """
        初始化正则分词器。
        
        Args:
            filepath: 输入文本文件路径
            PAT: 预留参数（未使用）
            pattern: 预留参数（未使用）
            special_tokens: 特殊 token 列表，默认为 ["<|endoftext|>"]
        """
        if special_tokens is None:
            special_tokens = ["<|endoftext|>"]
        self.filepath = filepath
        self.special_tokens = special_tokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pattern = "|".join(map(re.escape, self.special_tokens))

        
    def find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        将文件分割成多个独立处理的块。
        按特殊 token 位置对齐块边界，以便并行处理。
        如果边界重叠，返回的块数可能少于 desired_num_chunks。
        
        Args:
            file: 二进制文件对象
            desired_num_chunks: 期望的块数
            split_special_token: 用于对齐块边界的特殊 token（字节形式）
        
        Returns:
            块边界位置列表（字节偏移量）
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))


    def pretokenize(self, num_workers: int = None) -> List[str]:
        """
        预分词：将文本文件分块读取并进行正则匹配分词。
        支持并行处理。
        
        Args:
            num_workers: 并行进程数，默认为 CPU 核心数
        
        Returns:
            预分词后的 token 列表（字符串形式）
        """
        tokens = []
        if num_workers is None:
            num_workers = min(cpu_count(), 4)
            
        with open(self.filepath, "rb") as f:
            # Use default special token for splitting if not specified, 
            # assuming special_tokens[0] is the separator like <|endoftext|>
            boundary_token = self.special_tokens[0].encode("utf-8") if self.special_tokens else b""
            boundaries = self.find_chunk_boundaries(f, num_workers, boundary_token)

            # Parallel execution
            # Prepare arguments for each chunk
            chunk_args = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                chunk_args.append((self.filepath, start, end, self.pattern, self.PAT))
            
            with Pool(processes=num_workers) as pool:
                results = pool.map(_process_chunk_worker, chunk_args)
                
            for res in results:
                tokens.extend(res)
                
        return tokens
    
    def potokenize_text(self, text:str) -> List[str]:
        """
        对单个文本字符串进行预分词（未在 train 中使用，提供额外接口）。
        
        Args:
            text: 输入文本字符串
        
        Returns:
            预分词后的 token 列表
        """
        text_parts = re.split(self.pattern, text)
        tokens = [match.group() for chunk_item in text_parts for match in re.finditer(self.PAT, chunk_item)]
        return tokens
    
class BPE_Tokenizer_Trainer:
    def __init__(self, input_path:str, vocab_size:int, special_tokens:list[str]):
        """
        初始化 BPE 分词器训练器。
        
        Args:
            input_path: 输入文本文件路径
            vocab_size: 目标词表大小
            special_tokens: 特殊 token 列表
        """
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens else []

        self.word_freqs = {}  # 预分词后的词频
        self.merges:List[tuple[bytes, bytes]] = []  # BPE 合并规则列表
        self.vocab:Dict[int, bytes] = {}  # token_id -> 字节序列
        self.inverse_vocab: Dict[bytes, int] = {}  # 字节序列 -> token_id


    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        训练 BPE 分词器的主函数。
        执行流程：初始化词表 -> 预分词 -> 计算合并规则 -> 更新词表。
        
        Returns:
            (vocab, merges) 元组
            - vocab: token_id 到 byte 序列的映射
            - merges: 所有 BPE 合并规则的列表
        """
        min_vocab_size = 256 + len(self.special_tokens)
        self.vocab, self.inverse_vocab = self._initialize_vocab()

        assert self.vocab_size >= min_vocab_size, f"Vocab size must be at least {min_vocab_size}"

        word_freqs = self._pretokenize_corpus()  # 预分词并计算词频

        self.merges = self._compute_merges(word_freqs)  # 计算 BPE 合并规则

        self._update_vocab_with_merges()  # 更新词表

        return self.vocab, self.merges
    
    def _initialize_vocab(self) -> Dict[int, bytes]:
        """
        初始化词表，包含特殊 token 和所有 256 个单字节 token。
        
        Returns:
            (vocab, inverse_vocab) 元组
            - vocab: token_id -> bytes
            - inverse_vocab: bytes -> token_id
        """
        vocab = {}
        inverse_vocab = {}

        # 添加特殊 token
        for i, token in enumerate(self.special_tokens):
            token_bytes = token.encode('utf-8')
            vocab[i] = token_bytes
            inverse_vocab[token_bytes] = i

        # 添加 256 个单字节 token (0x00 - 0xFF)
        for i in range(256):
            token_id = len(self.special_tokens) + i
            byte_token = bytes([i])
            vocab[token_id] = byte_token
            inverse_vocab[byte_token] = token_id
    
        return vocab, inverse_vocab
    
    def _pretokenize_corpus(self) -> Dict[Tuple[bytes, ...], int]:
        """
        预分词语料库：分割文本 -> 编码为字节 -> 统计频率。
        
        处理流程：
        1. 用 RegexTokenizer 进行预分词（得到字符串 token）
        2. 将每个 token 编码为 UTF-8 字节序列
        3. 将字节序列拆分为单字节 token 的元组
        4. 统计每种 token 序列的出现频率
        
        Returns:
            词频字典，键为字节元组，值为出现次数
            例如：{(b'c', b'a', b't'): 5, (b'd', b'o', b'g'): 3}
        """
        word_freqs = defaultdict(int)
        regex_tokenizer = RegexTokenizer(
            filepath = self.input_path,
            special_tokens = self.special_tokens
        )

        # 进行正则预分词
        tokens = regex_tokenizer.pretokenize()
        for token in tokens:
            # 将字符串 token 编码为 UTF-8 字节序列
            # 例如："你好" -> b'\xe4\xbd\xa0\xe5\xa5\xbd'
            word_bytes = token.encode('utf-8')

            # 将字节序列拆分为单字节 token 的元组
            # 例如：b'cat' -> (b'c', b'a', b't')
            word_tuple = tuple(bytes([b]) for b in word_bytes)

            # 统计该 byte-level token 序列在语料中的出现频率
            word_freqs[word_tuple] += 1

        # 返回普通 dict，便于后续处理
        return dict(word_freqs)
    
    def _compute_merges(
            self,
            word_freqs: dict[tuple[bytes, ...], int]
    ) -> List[tuple[bytes, bytes]]:
        """
        计算 BPE 合并规则（优化版：增量更新 + 优先队列）。
        """
        merges = []
        target_merges = self.vocab_size - len(self.vocab)
        
        # 1. Build initial counts and inverted index
        pair_counts = defaultdict(int)
        # pair -> set of words containing it
        pair_to_words = defaultdict(set)
        
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += freq
                pair_to_words[pair].add(word)
        
        # 2. Build max heap (-count, MaxHeapPair(pair))
        pq = []
        for pair, count in pair_counts.items():
            heapq.heappush(pq, (-count, MaxHeapPair(pair)))
            
        for _ in range(target_merges):
            # Find best pair
            best_pair = None
            while pq:
                count, max_heap_pair = heapq.heappop(pq)
                pair = max_heap_pair.pair
                if -count == pair_counts[pair]:
                    best_pair = pair
                    break
            
            if not best_pair or pair_counts[best_pair] < 1:
                break
                
            merges.append(best_pair)
            p0, p1 = best_pair
            new_token = p0 + p1
            
            # Words to update (copy set to avoid modification during iteration)
            words_to_update = list(pair_to_words[best_pair])
            
            # Collect deltas for bulk update
            # word -> freq_change
            word_updates = defaultdict(int)
            
            for word in words_to_update:
                if word not in word_freqs: continue
                freq = word_freqs[word]
                
                # Create merged word
                new_word_list = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == p0 and word[i+1] == p1:
                        new_word_list.append(new_token)
                        i += 2
                    else:
                        new_word_list.append(word[i])
                        i += 1
                new_word = tuple(new_word_list)
                
                if new_word != word:
                    word_updates[word] -= freq
                    word_updates[new_word] += freq
                    
            # Apply updates
            for word, freq_change in word_updates.items():
                if freq_change == 0: continue
                
                current_freq = word_freqs.get(word, 0)
                new_freq = current_freq + freq_change
                
                if new_freq <= 0:
                    if word in word_freqs: del word_freqs[word]
                    # Clean up indices if needed (optional for correctness but good for memory)
                    if new_freq == 0:
                        for i in range(len(word) - 1):
                            p = (word[i], word[i+1])
                            if p in pair_to_words: pair_to_words[p].discard(word)
                else:
                    word_freqs[word] = new_freq
                    # Ensure new word is in indices
                    for i in range(len(word) - 1):
                        p = (word[i], word[i+1])
                        pair_to_words[p].add(word)
                        
                # Update pair counts
                for i in range(len(word) - 1):
                    p = (word[i], word[i+1])
                    pair_counts[p] += freq_change
                    # Push updated count to heap
                    heapq.heappush(pq, (-pair_counts[p], MaxHeapPair(p)))

        return merges

    def _merge_pair(
            self,
            word_freqs: dict[tuple[bytes, ...], int],
            pair: tuple[bytes, bytes]
    ) -> dict[tuple[bytes, ...], int]:
        """
        在所有 token 序列中将指定的 byte pair 合并为单个 token。
        
        Args:
            word_freqs: 词频字典，键为 byte 元组，值为出现次数
            pair: 要合并的 byte pair，例如 (b'c', b'a')
        
        Returns:
            new_word_freqs: 合并后的词频字典
                            例如：{(b'ca', b't'): 5} （原来是 {(b'c', b'a', b't'): 5}）
        """
        new_word_freqs: dict[tuple[bytes, ...], int] = {}
        token_a, token_b = pair
        merged_token = token_a + token_b  # 合并后的新 token

        for word_tuple, freq in word_freqs.items():
            # 如果序列长度 < 2，无法合并，直接保留
            if len(word_tuple) < 2:
                new_word_freqs[word_tuple] = freq
                continue

            # 构建新的 token 序列，遇到 pair 就合并
            new_word: list[bytes] = []
            i = 0
            while i < len(word_tuple):
                # 检查当前位置是否匹配 pair
                if i < len(word_tuple) - 1 and word_tuple[i] == token_a and word_tuple[i + 1] == token_b:
                    new_word.append(merged_token)
                    i += 2  # 跳过两个 token
                else:
                    new_word.append(word_tuple[i])
                    i += 1

            new_word_tuple = tuple(new_word)
            # 相同序列的频率累加（不同原始序列可能合并成相同结果）
            new_word_freqs[new_word_tuple] = new_word_freqs.get(new_word_tuple, 0) + freq

        return new_word_freqs
    
    def _count_pairs(
            self,
            word_freqs: dict[tuple[bytes, ...], int]
    ) -> dict[tuple[bytes, bytes], int]:
        """
        统计所有 token 序列中相邻 byte pair 的出现频率。
        
        Args:
            word_freqs: 词频字典，键为 byte 元组，值为出现次数
                        例如：{(b'c', b'a', b't'): 5, (b'd', b'o', b'g'): 3}
        
        Returns:
            pair_counts: 相邻 pair 频率字典
                         例如：{(b'c', b'a'): 5, (b'a', b't'): 5, (b'd', b'o'): 3, (b'o', b'g'): 3}
        """
        pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)

        for word_tuple, freq in word_freqs.items():
            # 序列长度小于 2 时无法构成 pair
            if len(word_tuple) < 2:
                continue

            # 遍历相邻的 byte pair
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                # 累加频率（该 pair 在此序列出现 1 次，乘以序列频率）
                pair_counts[pair] += freq

        return dict(pair_counts)

    def _update_vocab_with_merges(self):
        """
        根据合并规则更新词表。
        
        将每个 merge 规则转换为 (token_id, 合并的字节序列) 的映射。
        token_id 从 base_vocab_size 开始连续递增，顺序与 merge 规则一致。
        这个顺序在编码/解码时很重要。
        """
        # 基础词表大小：= 特殊 token 数量 + 256 个单字节 token
        # 新生成的 BPE token 将从该索引之后依次编号
        base_vocab_size = 256 + len(self.special_tokens)

        # 依次遍历所有 merge 规则（顺序非常重要）
        for i, (token_a, token_b) in enumerate(self.merges):
            # 将 byte pair (a, b) 合并成一个新的 byte token
            # 例如：(b'h', b'e') -> b'he'
            merged_token = token_a + token_b

            # 为该合并 token 分配新的 token_id
            # token_id 连续递增，保证与 merge 顺序一致
            token_id = base_vocab_size + i

            # 更新正向词表：token_id -> byte 序列
            self.vocab[token_id] = merged_token

            # 更新反向词表：byte 序列 -> token_id
            self.inverse_vocab[merged_token] = token_id