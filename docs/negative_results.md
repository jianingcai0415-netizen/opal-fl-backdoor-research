# Negative results

Several branches were intentionally not promoted.

- Early OGM/SCD branches were useful for diagnosing the problem, but they did not reliably preserve ASR after aggregation and recovery.
- APD-style virtual objectives were finite and measurable, but proxy success did not consistently transfer to the actual submitted update.
- Stage5p and Stage5q preserved attack-window ASR but failed to improve ASR-t20 over the Stage5o baseline.

These negative results are still valuable because they narrow the paper direction: the next useful variable is not more attack strength, more malicious clients, or a broad parameter sweep. The next useful question is whether submitted-update write-in should be placed in a more persistent layer/subspace.

