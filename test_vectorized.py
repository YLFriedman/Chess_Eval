import torch
import numpy as np
from utils import batch_fen_to_tensor
# ==========================================
# 1. ORIGINAL CODE (For baseline comparison)
# ==========================================
channels = {'P' : 0, 'N': 1, 'B' : 2, 'R' : 3, 'K' : 4,
            'Q' : 5, 'p' : 6, 'n' : 7, 'b' : 8, 'r' : 9,
            'k' : 10, 'q': 11, 'turn' : 12, 'Kc' : 13,
           'Qc' : 14, 'kc' : 15, 'qc' : 16,'en_passant' : 17}

def original_fen_to_tensor(fen):
    fields = fen.split(' ')
    board_part, turn, castling, en_passant = fields[:4]
    tens = torch.zeros(18, 8, 8, dtype=torch.float32) 
    rows = board_part.split('/')
    for row_num, curr_row in enumerate(rows):
        col_num = 0
        for char in curr_row:
            if char in channels:
                tens[channels[char], row_num, col_num] = 1
                col_num += 1
            else:
                col_num += int(char)
    if turn == 'w':
        tens[12, :, :] = 1
    if castling != '-':
        for char in castling:
            channel = char + 'c'
            if channel in channels: 
                tens[channels[channel], :, :] = 1
    if en_passant != '-':
        row_num = 8 - int(en_passant[1])
        col_num = ord(en_passant[0]) - ord('a')
        tens[17, row_num, col_num] = 1
    return tens

# ==========================================
# 2. NEW VECTORIZED CODE
# ==========================================
PIECES = ['P', 'N', 'B', 'R', 'K', 'Q', 'p', 'n', 'b', 'r', 'k', 'q']

# ==========================================
# 3. TEST SUITE
# ==========================================
if __name__ == "__main__":
    test_fens = [
        # Standard starting position
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        # Black to move, no castling rights
        "8/8/8/4k3/8/8/4K3/8 b - - 0 1",
        # White to move, partial castling rights (White Kingside, Black Queenside)
        "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1",
        # Black to move, en passant target on e3
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        # White to move, en passant target on c6
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        # Complex midgame with random numbers and pieces
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    ]

    print(f"Testing {len(test_fens)} distinct FEN configurations...")

    # 1. Process sequentially with original function
    original_tensors = []
    for fen in test_fens:
        original_tensors.append(original_fen_to_tensor(fen))
    original_stacked = torch.stack(original_tensors)

    # 2. Process simultaneously with vectorized function
    vectorized_stacked = batch_fen_to_tensor(test_fens)

    # 3. Compare outputs
    if torch.equal(original_stacked, vectorized_stacked):
        print("✅ SUCCESS: Vectorized outputs perfectly match original outputs.")
        print(f"Shape verified: {vectorized_stacked.shape}")
    else:
        print("❌ FAILURE: Tensors do not match!")
        
        # Find which specific FEN failed for easier debugging
        for i, fen in enumerate(test_fens):
            if not torch.equal(original_stacked[i], vectorized_stacked[i]):
                print(f"\nMismatch found at index {i}")
                print(f"FEN: {fen}")
                
                # Check which channel failed
                for channel in range(18):
                    if not torch.equal(original_stacked[i, channel], vectorized_stacked[i, channel]):
                        print(f"-> Channel {channel} differs.")