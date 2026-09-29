import torch
import numpy as np

# Channels map sequentially matching the original dictionary
PIECES = ['P', 'N', 'B', 'R', 'K', 'Q', 'p', 'n', 'b', 'r', 'k', 'q']

def batch_fen_to_tensor(fens):
    B = len(fens)
    boards, turns, castles, eps = [], [], [], []
    
    # Fast Python sequential split
    for fen in fens:
        parts = fen.split(' ')
        boards.append(parts[0])
        turns.append(parts[1])
        castles.append(parts[2])
        eps.append(parts[3])

    # Expand RLE digits (1-8) into empty spaces (represented by dots)
    for i in range(1, 9):
        repl = '.' * i
        boards = [b.replace(str(i), repl) for b in boards]
        
    # Strip slashes to yield exactly 64 characters per board
    boards = [b.replace('/', '') for b in boards]
    
    # Convert to a 2D matrix of single-byte characters: (B, 64) -> (B, 8, 8)
    board_arr = np.array(boards, dtype='S64').view('S1').reshape(B, 8, 8)
    
    tens = np.zeros((B, 18, 8, 8), dtype=np.float32)
    
    # Vectorized Piece Channels (0-11)
    for channel, piece in enumerate(PIECES):
        # NumPy broadcast boolean comparison automatically casts True/False to 1/0
        tens[:, channel, :, :] = (board_arr == piece.encode())

    # Vectorized Turn Channel (12)
    turn_arr = np.array(turns) == 'w'
    tens[:, 12, :, :] = turn_arr[:, None, None]

    # Vectorized Castling Channels (13-16)
    for i, c in enumerate(['K', 'Q', 'k', 'q']):
        has_castle = np.array([c in cast for cast in castles])
        tens[:, 13 + i, :, :] = has_castle[:, None, None]

    # Vectorized En Passant Channel (17)
    ep_arr = np.array(eps)
    has_ep = ep_arr != '-'
    valid_ep_idx = np.where(has_ep)[0]
    
    if len(valid_ep_idx) > 0:
        valid_eps = ep_arr[has_ep]
        ep_bytes = valid_eps.astype('S2').view('S1').reshape(-1, 2)
        
        # ASCII math to find row and col indices
        cols = ep_bytes[:, 0].view(np.uint8) - ord('a')
        rows = 8 - (ep_bytes[:, 1].view(np.uint8) - ord('0'))
        
        tens[valid_ep_idx, 17, rows, cols] = 1

    return torch.from_numpy(tens)