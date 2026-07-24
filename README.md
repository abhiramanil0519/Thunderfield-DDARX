# Thunderfield-DDARX
not more than 350 chars  DDARX by Thunderfield rethinks paged attention by splitting VRAM allocation from cross request sharing. Allocation uses CUDA virtual memory for contiguous, on demand VRAM per sequence, so kernels run without lookup overhead. Sharing is handled by a lightweight host side index. Experimental, unbenchmarked.
