import ast
import regex as re
from typing import Iterator, Iterable
class BPE_Tokenizer:
    def __init__(
      self,
      vocab: dict[int, bytes],                 # 词表：token_id -> token(bytes)
      merges: list[tuple[bytes, bytes]],       # BPE 合并规则列表，每一项是 (token1, token2)
      special_tokens: list[str] | None = None  # 特殊 token（如 <|endoftext|>），以字符串形式给出
  ):
      super().__init__()

      # 复制一份 vocab，避免修改外部传入的对象
      self.vocab = vocab.copy()

      # 保存 BPE 的 merge 规则（有序，顺序即优先级）
      self.merges = merges

      # 构建反向词表：token(bytes) -> token_id
      # 用于从字节 token 快速查 id
      self.inverse_vocab = {v: k for k, v in vocab.items()}

      # merge_priority[(token1, token2)] = 合并优先级
      # 数字越小，优先级越高
      self.merge_priority = {}
      for i, (token1, token2) in enumerate(merges):
          self.merge_priority[(token1, token2)] = i

      # 特殊 token 列表（若未提供则为空列表）
      self.special_tokens = special_tokens or []

      # 特殊 token 字符串 -> token_id 的映射
      self.special_tokens_ids = {}

      # 新 token 的起始 id（接在原 vocab 最大 id 后面）
      next_id = max(self.vocab.keys()) + 1 if self.vocab else 0

      for special_token in self.special_tokens:
          # 将特殊 token 从 str 编码为 utf-8 bytes
          # BPE 内部统一使用 bytes 作为 token 表示
          special_bytes = special_token.encode('utf-8')

          if special_bytes not in self.inverse_vocab:
              # 如果该特殊 token 不在原始词表中，则新增
              self.vocab[next_id] = special_bytes
              self.inverse_vocab[special_bytes] = next_id
              self.special_tokens_ids[special_token] = next_id
              next_id += 1
          else:
              # 如果已经存在，直接复用已有 token_id
              self.special_tokens_ids[special_token] = self.inverse_vocab[special_bytes]

      if self.special_tokens:
          # 对特殊 token 进行正则转义，避免 < | > 等字符被当作正则符号
          escaped_tokens = [re.escape(token) for token in self.special_tokens]

          # 按长度从大到小排序，保证最长匹配优先
          # 例如 "<|endoftext|>" 要先于 "<|" 被匹配
          escaped_tokens.sort(key=len, reverse=True)

          # 构造形如 "<\|endoftext\|>|<\|pad\|>" 的正则
          self.special_pattern = '|'.join(escaped_tokens)
      else:
          # 没有特殊 token，则不使用特殊匹配
          self.special_pattern = None
          
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,                     # vocab 文件路径（通常是 token_id -> bytes 的字典）
        merges_filepath: str,                    # merges 文件路径（BPE 合并规则）
        special_tokens: list[str] | None = None  # 特殊 token（如 <|endoftext|>）
    ):
        try:
            # 以 UTF-8 编码读取 vocab 文件
            with open(vocab_filepath, 'r', encoding='utf-8') as f:
                vocab_text = f.read()

                # 使用 ast.literal_eval 将字符串安全地解析为 Python 对象
                # 这里期望得到的是一个 dict[int, bytes]
                vocab = ast.literal_eval(vocab_text)

        except FileNotFoundError:
            # 如果 vocab 文件不存在，抛出明确的错误信息
            raise FileNotFoundError(f"Vocab file not found: {vocab_filepath}")

        merges = []
        try:
            # merges 文件通常每一行是两个 token（以空格分隔）
            with open(merges_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    # 去掉首尾空白字符
                    line = line.strip()

                    # 跳过空行和注释行（以 # 开头）
                    if not line or line.startswith("#"):
                        continue

                    # 每一行应包含两个 token
                    a, b = line.split()

                    # 将字符串 token 编码为 UTF-8 bytes
                    # 与 BPE 内部统一的 bytes 表示保持一致
                    merges.append((a.encode('utf-8'), b.encode('utf-8')))

        except FileNotFoundError:
            # 如果 merges 文件不存在，抛出明确的错误信息
            raise FileNotFoundError(f"Merges file not found: {merges_filepath}")


        # 调用构造函数，返回一个 BpeTokenizer 实例
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """
        将输入的字符串 text 编码为 token id 序列
        """

        # 如果输入为空字符串，直接返回空列表
        if not text:
            return []

        # 最终输出的 token id 列表
        token_ids = []

        if self.special_pattern:
            # 使用正则对文本进行切分
            # 注意：这里给 pattern 加了括号
            #   re.split(pattern, text)        → 会丢弃匹配到的内容
            #   re.split(f'({pattern})', text) → 会保留匹配到的内容
            parts = re.split(f'({self.special_pattern})', text)

            for part in parts:
                # 跳过空字符串片段
                if not part:
                    continue

                # 如果当前片段正好是一个特殊 token
                if part in self.special_tokens_ids:
                    # 直接映射为对应的 token id
                    token_ids.append(self.special_tokens_ids[part])
                else:
                    # 否则按普通文本处理（BPE 编码）
                    token_ids.extend(self._encode_ordinary(part))

        else:
            token_ids = self._encode_ordinary(text)

        # 返回最终的 token id 序列
        return token_ids
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        对一个字符串可迭代对象进行编码，返回一个惰性生成的 token id 流

        适用场景：
        - 大文本 / 多文档数据集
        - 希望边读取、边编码、边消费，避免一次性占用大量内存
        """

        # 逐个读取 iterable 中的字符串
        for text in iterable:
            # 对每个字符串调用 encode
            # yield from 会将 encode(text) 生成的 token id
            # 逐个“转发”给外层迭代器
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
      if not self.vocab or not self.merges or not self.inverse_vocab:
          raise ValueError("Tokenizer has not been trained yet.")

      byte_array = bytearray()
      for token_id in ids:
          if token_id in self.vocab:
              byte_array+=self.vocab[token_id]

      return byte_array.decode('utf-8', errors='replace')


    def _encode_ordinary(self, text: str) -> list[int]:
        """
        Encoding that ignores special tokens (uses regex splitting + BPE).
        """
        # GPT-4 style regex pattern
        pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        token_ids = []
        # findall captures the matches
        for chunk in re.findall(pat, text):
            token_ids.extend(self._bpe_encode(chunk))
            
        return token_ids

    def _bpe_encode(self, text: str) -> list[int]:
        """
        Apply BPE to a single text chunk (no regex splitting).
        """
        # 1. Convert to initial byte tokens
        word = [self.inverse_vocab[bytes([b])] for b in text.encode("utf-8")]
        
        if not word:
            return []

        while len(word) >= 2:
            # Find the pair with the lowest merge index (highest priority)
            min_idx = float('inf')
            best_pair_pos = -1
            best_pair_val = None
            
            # Identify candidates
            for i in range(len(word) - 1):
                # Retrieve bytes for current tokens
                token1_bytes = self.vocab[word[i]]
                token2_bytes = self.vocab[word[i+1]]
                pair = (token1_bytes, token2_bytes)
                
                if pair in self.merge_priority:
                    rank = self.merge_priority[pair]
                    if rank < min_idx:
                        min_idx = rank
                        best_pair_pos = i
                        best_pair_val = pair
            
            # If no merge possible, stop
            if best_pair_pos == -1:
                break
                
            # Apply the best merge globally in the word (or just first? Standard BPE usually merges all occurrences)
            # But the loop above just found *an* occurrence with min_rank.
            # To be efficient and correct, usually we merge *all* occurrences of the best pair found.
            
            # Let's find the absolute best pair available in the word
            # (The loop above found the best pair that *exists* in the word)
            
            target_pair = best_pair_val
            target_rank = min_idx
            
            # Merge all occurrences of target_pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1:
                    t1 = self.vocab[word[i]]
                    t2 = self.vocab[word[i+1]]
                    if (t1, t2) == target_pair:
                        # Merge!
                        merged_bytes = t1 + t2
                        new_word.append(self.inverse_vocab[merged_bytes])
                        i += 2
                        continue
                new_word.append(word[i])
                i += 1
            word = new_word
            
        return word
