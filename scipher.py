"""
==============================================================================
DSP lab: stream cipher
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

POLY_LFSR_1: list[int] = [1, 0, 0, 1] + [0] * 16 + [1]     # x^20 + x^3 + 1
POLY_LFSR_2: list[int] = [1, 1, 1, 0, 1] + [0] * 14 + [1]  # x^19 + x^5 + x^2 + x + 1
POLY_LFSR_3: list[int] = [1, 0, 1] + [0] * 18 + [1]        # x^21 + x^2 + 1

SIZE_LFSR_1 = len(POLY_LFSR_1) - 1
SIZE_LFSR_2 = len(POLY_LFSR_2) - 1
SIZE_LFSR_3 = len(POLY_LFSR_3) - 1


def read_string_from_file(file_path: str) -> str:
    """
    Reads the content of a file and returns it as a string.

    This function takes the file path as input, opens the file in read mode,
    and returns the entire content of the file as a string.

    Args:
        file_path (str): The path to the file that needs to be read.

    Returns:
        str: The content of the file as a string.
    """
    with open(file_path, "r") as file:
        return file.read()


def hex_to_bits(hex_str: str) -> list[int]:
    """
    Converts a hex string (ciphertext) to a list of bits.

    The function interprets the input hexadecimal string, converts each hex
    character into its binary equivalent, and returns a list of individual bits,
    where each bit is represented as an integer (0 or 1).

    Args:
        hex_str (str): Hexadecimal string to be converted (ciphertext).

    Returns:
        list[int]: A list of bits representing the binary equivalent of the
        hexadecimal input.
    """
    data = bytes.fromhex(hex_str)
    bits = []
    for byte in data:
        for b in f"{byte:08b}":
            bits.append(int(b))
    return bits


def str_to_bits(text: str) -> list[int]:
    """
    Converts a given string (plaintext) into a list of its binary
    representation as integers.

    This function takes an input string and encodes it to binary
    form using UTF-8 encoding. For each byte in the encoded data,
    it generates its binary representation, splits it into individual
    bits, and converts them into a list of integers.

    Args:
        text (str): The input string to be converted to bits (plaintext).

    Returns:
        list[int]: A list of integers where each integer is a bit
        (0 or 1) representing the binary encoding of the input string.
    """
    data = text.encode("utf-8")
    bits = []
    for byte in data:
        for b in f"{byte:08b}":
            bits.append(int(b))
    return bits


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


@dataclass
class LFSR:
    """
    LFSR with:
    - state: list[int] of length n, MSB-first (state[0] outputs)
    - poly: list[int] of length n+1, binary coefficients for x^0..x^n,
        with poly[-1]==1

    One clock:
        out = state[0]
        feedback = XOR(state[n-1-k] for k where poly[k]==1 and k<n)
        state = state[1:] + [feedback]
    """

    state: list[int]
    poly: list[int]

    def __post_init__(self) -> None:
        n = len(self.state)
        if n == 0 or any(b not in (0, 1) for b in self.state):
            raise ValueError("state must be a non-empty list of bits (0/1).")
        if len(self.poly) != n + 1:
            raise ValueError("poly length must be len(state)+1 (degree n).")
        if self.poly[-1] != 1:
            raise ValueError("poly[-1] (x^n) must be 1.")
        if all(b == 0 for b in self.state):
            raise ValueError("state cannot be all zeros.")

    def clock(self) -> int:
        """
        Calculates the output of the Linear Feedback Shift Register (LFSR) and
        updates its state based on feedback computed from the polynomial
        coefficients.

        Returns:
            int: The bit output from the LFSR before the state update.
        """
        out = self.state[0]
        n = len(self.state)
        feedback = 0
        for k, coeff in enumerate(self.poly[:-1]):  # exclude x^n
            if coeff == 1:
                idx = n - 1 - k
                feedback ^= self.state[idx]
        self.state = self.state[1:] + [feedback]
        return out


class Cipher:
    """
    Stream cipher with three polynomial-defined LFSRs.

    The *key* is a tuple of three bit-lists (initial states for LFSR1, LFSR2,
    LFSR3):
        key_bits = (state1_bits, state2_bits, state3_bits)
    """

    poly1 = POLY_LFSR_1
    poly2 = POLY_LFSR_2
    poly3 = POLY_LFSR_3

    deg1 = len(poly1) - 1
    deg2 = len(poly2) - 1
    deg3 = len(poly3) - 1

    def __init__(self,
                 key_bits: tuple[list[int], list[int], list[int]]) -> None:
        s1, s2, s3 = key_bits
        if len(s1) != self.deg1 or len(s2) != self.deg2 or len(s3) != self.deg3:
            raise ValueError(
                f"key_bits lengths must be ({self.deg1}, {self.deg2}, {self.deg3}).")
        if any(b not in (0, 1) for b in s1 + s2 + s3):
            raise ValueError("key_bits must contain only 0/1.")
        if (all(b == 0 for b in s1) or all(b == 0 for b in s2) or all(
                b == 0 for b in s3)):
            raise ValueError("Each LFSR state must be non-zero.")

        self._key_bits: tuple[list[int], list[int], list[int]] = (s1[:], s2[:],
                                                                  s3[:],)
        self.reset()

    def reset(self) -> None:
        """Reset L1, L2, L3 to the initial key states."""
        s1, s2, s3 = (self._key_bits[0][:], self._key_bits[1][:],
                      self._key_bits[2][:],)
        self.l1 = LFSR(state=s1, poly=self.poly1)
        self.l2 = LFSR(state=s2, poly=self.poly2)
        self.l3 = LFSR(state=s3, poly=self.poly3)

    def _next_keystream_bit(self) -> int:
        x = self.l1.clock()
        y = self.l2.clock()
        z = self.l3.clock()
        return (x & y) ^ ((~x & 1) & z)

    # ---------- Keystream ----------
    def keystream_bits(self, n_bits: int) -> list[int]:
        """
        Generates a specified number of bits from the keystream.

        Args:
            n_bits (int): The number of keystream bits to generate.

        Returns:
            list[int]: A list containing the generated keystream bits.
        """
        return [self._next_keystream_bit() for _ in range(n_bits)]

    def keystream_bytes(self, n_bytes: int) -> bytes:
        bits = self.keystream_bits(n_bytes * 8)
        out = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for b in bits[i: i + 8]:
                byte = (byte << 1) | (b & 1)
            out.append(byte)
        return bytes(out)

    # ---------- Encrypt / Decrypt (bytes!) ----------
    def encrypt(self, plaintext: bytes) -> bytes:
        ks = self.keystream_bytes(len(plaintext))
        return _xor_bytes(plaintext, ks)

    def decrypt(self, ciphertext: bytes) -> bytes:
        ks = self.keystream_bytes(len(ciphertext))
        return _xor_bytes(ciphertext, ks)

    # ---------- String helpers ----------
    def encrypt_to_hex(self, text: str, reset: bool = True) -> str:
        """
        Encrypts a given text (plaintext) to its hexadecimal representation
        (ciphertext).

        Args:
            text (str): The input string to be encrypted (plaintext).
            reset (bool): A boolean flag that determines whether the internal
                          state should be reset before encryption. Defaults to
                          True.

        Returns:
            The hexadecimal string representation of the encrypted input
            (ciphertext).
        """
        if reset:
            self.reset()
        return self.encrypt(text.encode("utf-8")).hex()

    def decrypt_from_hex(self, hex_text: str, reset: bool = True) -> str:
        """
        Decrypt a given hexadecimal string (ciphertext) and return the plaintext
        representation.

        Args:
            hex_text (str): A string containing the hexadecimal input to be
                            decrypted.
            reset (bool): A boolean flag that determines whether the internal
                          state should be reset before encryption. Defaults to
                          True.

        Returns:
            str: The UTF-8 decoded plaintext string obtained after decryption.
        """
        if reset:
            self.reset()
        return self.decrypt(bytes.fromhex(hex_text)).decode("utf-8",
                                                            errors="strict")
